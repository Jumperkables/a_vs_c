# Standard imports
import os, sys
import shutil
import random
import argparse
import h5py
import string
from tqdm import tqdm


# Complex imports
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#from multimodal.datasets import VQA, VQA2, VQACP, VQACP2, VQACPDataModule, VQACP2DataModule
from transformers import LxmertConfig, LxmertForQuestionAnswering, LxmertModel, LxmertTokenizer, BertTokenizer, BertModel, BertConfig
from transformers.models.lxmert.modeling_lxmert import LxmertVisualAnswerHead
import pytorch_lightning as pl
import torchmetrics
import wandb
import spacy

# Local imports
import myutils, dset_utils
from misc.multimodal_pip_vqa_utils import process_annotations
from models.bidaf import BidafAttn
import models.hopfield_layers.modules as hpf
#from models.assoc_vs_ctgrcl import VQA_AvsC

class vqa_dummy_args():
    """
    Dummy class to contain vqa args
    """
    def __init__(self, topk=-1, mao=-1):
        self.topk = topk
        self.mao = mao

def pad_obj_img_collate(data):
    def pad_images(images):
        """
        Images are of different sizes
        """
        maxh = max([ img.shape[1] for img in images])
        maxw = max([ img.shape[2] for img in images])
        padded_images = torch.zeros(len(images), images[0].shape[0], maxh, maxw)
        for idx, img in enumerate(images):
            h_excess = maxh - img.shape[1]
            h_start = random.randint(0, max(h_excess-1,0)) # Becuase randint includes the MAX range also
            w_excess = maxw - img.shape[2]
            w_start = random.randint(0, max(w_excess-1,0))
            #print(f"{idx}, h_start {h_start}, w_start {w_start}")
            padded_images[idx][:,h_start:h_start+img.shape[1],w_start:w_start+img.shape[2]] = img # Random pad of 0s in both dims
        return padded_images
    column_data = list(zip(*data))
    #keys = ["question", "answer", "bboxes", "features", "image", "return_norm", "abs_answer_tens", "conc_answer_tens"]
    return torch.stack(column_data[0]), torch.stack(column_data[1]), torch.stack(column_data[2]), torch.stack(column_data[3]), pad_images(column_data[4]), torch.stack(column_data[5]), torch.stack(column_data[6]), torch.stack(column_data[7])

def pad_question_collate(data):
    def pad_sequences(question):
        #max_len = max(map(lambda x: x.shape[1], question))
        #question = torch.stack([qu. for qu in question])
        question = nn.utils.rnn.pad_sequence(question, batch_first=True)
        return question
    column_data = list(zip(*data))
    #keys = ["question", "answer", "bboxes", "features", "image", "return_norm", "abs_answer_tens", "conc_answer_tens"]
    return pad_sequences(column_data[0]), torch.stack(column_data[1]), torch.stack(column_data[2]), torch.stack(column_data[3]), torch.stack(column_data[4]), torch.stack(column_data[5]).squeeze(1), torch.stack(column_data[6]), torch.stack(column_data[7]), torch.stack(column_data[8]), torch.stack(column_data[9]), torch.stack(column_data[10])


def set_avsc_loss_tensor(args, ans2idx): # loads norm_dict
    norm_dict = myutils.load_norms_pickle( os.path.join(os.path.dirname(__file__),"misc/all_norms.pickle"))
    idx2BCE_assoc_tensor = {}  # Associative 'abstract' relations
    idx2BCE_ctgrcl_tensor = {} # Categorical 'concrete' relations
    answers = ans2idx.keys()
    print("avsc loss, generating answer tensors")
    for ans, idx in tqdm(ans2idx.items()):    # Get the relevant word pairs in each answer 
        BCE_assoc_tensor = []
        BCE_ctgrcl_tensor = []
        for answer in answers:
            if ans == answer:
                BCE_assoc_tensor.append(1)
                BCE_ctgrcl_tensor.append(1)
            else:
                # For assoc and SimLex999-m, i have saved the word pairs commutatively, order is unimportant
                try:
                    assoc_score = norm_dict.word_pairs[f"{ans}|{answer}"]['assoc']['sources']['USF']['scaled']
                except KeyError:
                    assoc_score = 0
                try:
                    simlex_score = norm_dict.word_pairs[f"{ans}|{answer}"]['simlex999-m']['sources']['SimLex999']['scaled']
                except KeyError:
                    simlex_score = 0
                BCE_assoc_tensor.append(assoc_score)
                BCE_ctgrcl_tensor.append(simlex_score)
        # Final unknown token if needed
        if args.dataset in ["VQACP", "VQACP2"]:
            BCE_assoc_tensor.append(0)
            BCE_ctgrcl_tensor.append(0)
        idx2BCE_assoc_tensor[idx] = torch.Tensor(BCE_assoc_tensor)
        idx2BCE_ctgrcl_tensor[idx] = torch.Tensor(BCE_ctgrcl_tensor)
    # Final unknown token if needed
    if args.dataset in ["VQACP", "VQACP2"]:
        idx2BCE_assoc_tensor[len(answers)] = torch.Tensor([0]*len(answers)+[1])
        idx2BCE_ctgrcl_tensor[len(answers)] = torch.Tensor([0]*len(answers)+[1])
    return idx2BCE_assoc_tensor, idx2BCE_ctgrcl_tensor

def make_idx2norm(args, ans2idx):
    idx2norm = {}
    norm_dict = myutils.load_norms_pickle( os.path.join(os.path.dirname(__file__),"misc/all_norms.pickle"))
    for ans, idx in ans2idx.items():
        try:    #TODO Speedily developing this code, comeback later to replace with .get
            ans_norm = norm_dict.words[ans][args.norm]["sources"]["MT40k"]["scaled"] #TODO generalise this norm
            idx2norm[idx] = ans_norm
        except KeyError:
            ans = myutils.remove_stopwords(myutils.clean_word(ans)) # Try to remove stopwords and special characters
            try:
                ans_norm = norm_dict.words[ans][args.norm]["sources"]["MT40k"]["scaled"] #TODO generalise this norm
                idx2norm[idx] = ans_norm
            except KeyError:
                idx2norm[idx] = 0.5 # Set unknown norms to 0.5
    if args.dataset in ["VQACP", "VQACP2"]:
        idx2norm[len(idx2norm)] = 0.5  # Add one final 0.5 for the unknown token
    return idx2norm



######################################################
######################################################
# Datasets
######################################################
######################################################
# Dataset classes
class VQA(Dataset):
    """
    The VQA Changing Priors Dataset
    """
    def __init__(self, args, version="cp-v1", split="train", images=False, resnet=False, spatial=False, objects=False, obj_names=False, return_norm=False, return_avsc=False, n_objs=10, max_q_len=30):
        # Feature flags
        self.images_flag = images
        self.spatial_flag = spatial
        self.objects_flag = objects
        self.resnet_flag = resnet
        self.obj_names_flag = obj_names
        self.return_norm_flag = return_norm # The output of the answer norm algorithm
        self.return_avsc_flag = return_avsc # Output the avsc tensor between answers in answer vocab
        self.n_objs = n_objs
        self.max_q_len = max_q_len
        self.split = split
        self.args = args
        self.topk_flag = not (args.topk == -1) # -1 -> set flag to False
        self.min_ans_occ_flag = not (self.topk_flag) # -1 -> set flag to False
        # Answer2Idx
        if version == "cp-v1":
            data_root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/vqa/datasets/vqacp")
            if self.topk_flag:
                anno_prepro_path = os.path.join(data_root_dir, f"top{args.topk}_answers.json")
            else: # min_ans_occ
                anno_prepro_path = os.path.join(data_root_dir, f"occ_gt{args.min_ans_occ}_answers.json")
            if os.path.exists(anno_prepro_path):
                self.ans2idx = myutils.load_json(anno_prepro_path)
            else:
                self.create_ans2idx("cp-v1")
                self.ans2idx = myutils.load_json(anno_prepro_path)
        elif version == "cp-v2":
            data_root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/vqa/datasets/vqacp2")
            if self.topk_flag:
                anno_prepro_path = os.path.join(data_root_dir, f"top{args.topk}_answers.json")
            else: # min_ans_occ
                anno_prepro_path = os.path.join(data_root_dir, f"occ_gt{args.min_ans_occ}_answers.json")
            if os.path.exists(anno_prepro_path):
                self.ans2idx = myutils.load_json(anno_prepro_path)
            else:
                self.create_ans2idx("cp-v2")
                self.ans2idx = myutils.load_json(anno_prepro_path)
        if self.min_ans_occ_flag:
            self.ans2idx = {ans:ans_idx for ans_idx, ans in enumerate(self.ans2idx)}
        else:   # topk_flag
            self.ans2idx = {ans[0]:ans_idx for ans_idx, ans in enumerate(self.ans2idx)}
        self.tokeniser = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        # Questions and Answers
        ## TODO Tidy all these up with fstrings
        if version == "cp-v1":
            if split == "train":
                self.qs = myutils.load_json(os.path.join(data_root_dir, "train", "vqacp_v1_train_questions.json"))
                self.ans = myutils.load_json(os.path.join(data_root_dir, "train", "processed_vqacp_v1_train_annotations.json"))
            elif split == "test":
                self.qs = myutils.load_json(os.path.join(data_root_dir, "test", "vqacp_v1_test_questions.json"))
                self.ans = myutils.load_json(os.path.join(data_root_dir, "test", "processed_vqacp_v1_test_annotations.json"))
        elif version == "cp-v2":
            if split == "train":
                self.qs = myutils.load_json(os.path.join(data_root_dir, "train", "vqacp_v2_train_questions.json"))
                self.ans = myutils.load_json(os.path.join(data_root_dir, "train", "processed_vqacp_v2_train_annotations.json"))
            elif split == "test":
                self.qs = myutils.load_json(os.path.join(data_root_dir, "test", "vqacp_v2_test_questions.json"))
                self.ans = myutils.load_json(os.path.join(data_root_dir, "test", "processed_vqacp_v2_test_annotations.json"))
        # Print the percentage of questions with valid answer
        have_ans = 0
        for ans in self.ans:
            scores = ans["scores"]
            answer = max(scores, key=scores.get)
            answer = self.ans2idx.get(answer, len(self.ans2idx))
            if answer != len(self.ans2idx):
                have_ans += 1
        print(f"Number of Questions with answers in ans2idx: {have_ans*100/len(self.ans):.2f}% (should be very very high)")
        print(f"There are {len(self.ans2idx)} answers in this {'topk='+str(args.topk) if self.topk_flag else 'min_ans_occ='+str(args.min_ans_occ)} scheme")
        # Objects
        if self.objects_flag:
            object_root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/vqa/features/coco-bottom-up/trainval")
            h5_path = os.path.join(object_root_dir, "features.h5")
            self.h5_path = h5_path
            if not os.path.exists(h5_path):
                print(f"No features/bbox files. Generating them at {h5_path}. This'll take a while...")
                dset_utils.vqa_tsv_to_h5( os.path.join(object_root_dir, "karpathy_val_resnet101_faster_rcnn_genome.tsv"), h5_path )
                dset_utils.vqa_tsv_to_h5( os.path.join(object_root_dir, "karpathy_test_resnet101_faster_rcnn_genome.tsv"), h5_path )
                dset_utils.vqa_tsv_to_h5( os.path.join(object_root_dir, "karpathy_train_resnet101_faster_rcnn_genome.tsv.0"), h5_path )
                dset_utils.vqa_tsv_to_h5( os.path.join(object_root_dir, "karpathy_train_resnet101_faster_rcnn_genome.tsv.1"), h5_path )
                print("Created h5 file! Continuing...")
                #self.feats = h5py.File(h5_path, "r", driver=None)                
            else:
                pass
                #self.feats = h5py.File(h5_path, "r", driver=None)# MOVED to __getitem__ to avoid num_workers>0 error with h5
        if self.images_flag:
            raise NotImplementedError(f"This is implemented and working, but shouldnt be used right now until needed")
            self.images_root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/vqa/images")
        # Pre-extracted resnet features
        if self.resnet_flag:
            resnet_h5_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/vqa/resnet", "resnet.h5")
            if not os.path.exists(resnet_h5_path):
                # Preprocess resnet features
                dset_utils.frames_to_resnet_h5("VQACP", resnet_h5_path)
            pass # Once again this will be handled in __getitem__ becuase of h5 parallelism problem
        # Return norm
        if self.return_norm_flag:
            # TODO DEPRECATED?? self.idx2norm = make_idx2norm(args, self.ans2idx)  
            if args.norm_gt == "nsubj": # If you get norms for answers from the subject of the question
                self.lxmert_tokeniser = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
                self.nlp = spacy.load('en_core_web_sm')
            self.norm_dict = myutils.load_norms_pickle( os.path.join(os.path.dirname(__file__),"misc/all_norms.pickle")) 
        # Return avsc tensor
        if self.return_avsc_flag:   # If using the avsc loss, generate answer tensor
            self.idx2BCE_assoc_tensor, self.idx2BCE_ctgrcl_tensor = set_avsc_loss_tensor(args, self.ans2idx) # loads norm_dict
        self.features = []
        self.features += ['images' if images else '']
        self.features += ['resnet' if resnet else '']
        self.features += ['spatial' if spatial else '']
        self.features += ['objects' if objects else '']
        self.features += ['obj_names' if obj_names else '']
        self.features += ['return_norm' if return_norm else '']
        self.features += ['return_avsc' if return_avsc else '']
        nl = "\n"
        print(f"{split}{nl}Features:{nl}{nl.join(self.features)}")
        # VQA-CP, remove all questions that don't have an answer given the answer scheme
        # TODO Deprecated? Decide if to remove the now redundant no-answer token. It could be useful later
        print(f"Keeping {have_ans*100/len(self.ans):.2f}% of {split} questions ({len(self.ans)}) and ignoring the rest")
        for q_idx in range(len(self.qs)-1, -1, -1): # Using range in reverse means we shift our start and end points by -1 to get the right values
            scores = self.ans[q_idx]["scores"]
            answer = max(scores, key=scores.get)
            answer = self.ans2idx.get(answer, len(self.ans2idx)) # The final key is the designated no answer token 
            if answer == len(self.ans2idx): # If this answer is not in ans2idx
                del self.ans[q_idx]
                del self.qs[q_idx]
                assert len(self.qs) == len(self.ans), "Somehow the answer removal failed"

    def __len__(self):
        return len(self.qs)

    def __getitem__(self, idx):
        if self.objects_flag:
            if not hasattr(self, 'feats'):
                self.feats = h5py.File(self.h5_path, "r", driver=None)
        if self.resnet_flag:
            if not hasattr(self, "resnet_h5"):
                self.resnet_h5 = h5py.File(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/vqa/resnet", "resnet.h5"), "r", driver="core")   # File is small enough to fit in memory
        #TODO deprecated? question = torch.LongTensor(self.tokeniser(self.qs[idx]['question'], padding="max_length", truncation=True, max_length=self.max_q_len)["input_ids"])
        question = torch.LongTensor(self.tokeniser(self.qs[idx]['question'])["input_ids"])
        scores = self.ans[idx]["scores"]
        answer = max(scores, key=scores.get)
        answer_text = max(scores, key=scores.get)
        answer = self.ans2idx.get(answer, len(self.ans2idx)) # The final key is the designated no answer token 
        answer = torch.LongTensor([ answer ])            # i.e. len(ans2idx) == 3000 => 0-2999 answer ids and 3000 is the unknown token
        img_id = self.qs[idx]['image_id']
        if self.objects_flag:
            bboxes = torch.from_numpy(self.feats[str(img_id)]['bboxes'][:self.n_objs]).round()
            features = torch.from_numpy(self.feats[str(img_id)]['features'][:self.n_objs])
        else:   # Create dummy inputs
            bboxes = torch.zeros(self.n_objs, 4)
            features = torch.zeros(self.n_objs, 2048)
        if self.images_flag:
            # TODO finish implementing VQACP images input
            split = self.qs[idx]["coco_split"]
            padded_img_id = f"{self.qs[idx]['image_id']:012}"
            image = cv2.imread(f"{self.images_root_dir}/{split}/COCO_{split}_{padded_img_id}.jpg")
            image = torch.from_numpy(image).permute(2,0,1) # (channels, height, width)
        else:
            image = torch.zeros(3,244,244)
        # ResNet
        if self.resnet_flag:
            image = torch.from_numpy(self.resnet_h5[str(img_id)]["resnet"][:2048])
        else:
            image = torch.zeros(2048)
        # The average norm considered of the question/answer pair
        if self.return_norm_flag:
            if self.args.norm_gt == "answer":
                try:
                    return_norm = self.norm_dict.words[answer_text][self.args.norm]["sources"]["MT40k"]["scaled"] #TODO generalise this norm
                except KeyError:
                    return_norm = 0.5
            elif self.args.norm_gt == "nsubj":
                return_norm = []
                qu = myutils.clean_word(self.qs[idx]['question']) # Adapted to clean entire sentences
                decoded_qu = [ str(tok) for tok in self.nlp(qu) if tok.dep_ == "nsubj" ] 
                for nsubj in decoded_qu:
                    try:
                        norm = self.norm_dict.words[nsubj][self.args.norm]["sources"]["MT40k"]["scaled"] #TODO generalise this norm
                        return_norm.append(norm)
                    except KeyError:
                        pass
                if return_norm == []: # If there is no norm for the subject of question, try the norm of the answer
                    try:
                        return_norm = self.norm_dict.words[answer_text][self.args.norm]["sources"]["MT40k"]["scaled"]
                    except KeyError:
                        return_norm = 0.5 # If no norm from answer, set to 0.5 (halfway)
                else:
                    return_norm = myutils.list_avg(return_norm)
            return_norm = torch.Tensor([return_norm])
        else:
            return_norm = torch.Tensor([-1])
        # Return the avsc loss tensor for assoc/ctgrcl relations between answers
        if self.return_avsc_flag:
            abs_answer_tens = self.idx2BCE_assoc_tensor[self.ans2idx[answer_text]]
            conc_answer_tens = self.idx2BCE_ctgrcl_tensor[self.ans2idx[answer_text]]
        else:
            abs_answer_tens, conc_answer_tens = torch.Tensor([0]), torch.Tensor([0])
        # Return the image_id: [0/1, img_id] where 0 => train and 1 => val
        # This is because the VQA dataset images are split between train/val folders
        if self.qs[idx]["coco_split"] == "train2014":
            ret_img_id= torch.Tensor([0, img_id]).long()
        elif self.qs[idx]["coco_split"] == "val2014":
            ret_img_id= torch.Tensor([1, img_id]).long()
        else:
            raise ValueError("You got the split wrong Tom")# TODO remove this after works???
        return question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens, ret_img_id


    # UTILITY FUNCTIONS
    def create_ans2idx(self, version):
        answers = []
        if version == "cp-v1":
            data_root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/vqa/datasets/vqacp")
            train_path = os.path.join(data_root_dir, "train", "vqacp_v1_train_annotations.json")
            valid_path = os.path.join(data_root_dir, "test", "vqacp_v1_test_annotations.json")
            train_annotations = myutils.load_json(train_path)
            valid_annotations = myutils.load_json(valid_path) 
            train_path = os.path.join(data_root_dir, "train", "processed_vqacp_v1_train_annotations.json")
            valid_path = os.path.join(data_root_dir, "test", "processed_vqacp_v1_test_annotations.json")
        elif version == "cp-v2":
            data_root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/vqa/datasets/vqacp2")
            train_path = os.path.join(data_root_dir, "train", "vqacp_v2_train_annotations.json")
            valid_path = os.path.join(data_root_dir, "test", "vqacp_v2_test_annotations.json")
            train_annotations = myutils.load_json(train_path)
            valid_annotations = myutils.load_json(valid_path)
            train_path = os.path.join(data_root_dir, "train", "processed_vqacp_v2_train_annotations.json")
            valid_path = os.path.join(data_root_dir, "test", "processed_vqacp_v2_test_annotations.json")
        answers_path = os.path.join(data_root_dir)
        # Process annotations
        process_annotations(
            train_annotations, 
            valid_annotations, 
            train_path,
            valid_path,
            answers_path,
            self.args
        )
        #ans2idx = {answer:a_idx for a_idx, answer in enumerate(answers)}
        #myutils.save_pickle(ans2idx, save_path)





class GQA(Dataset):
    """
    The GQA Dataset: https://cs.stanford.edu/people/dorarad/gqa/download.html
    """
    def __init__(self, args, split="train", images=False, resnet=False, spatial=False, objects=False, obj_names=False, return_norm=False,return_avsc=False , n_objs=10, max_q_len=30):
        # Feature flags
        self.args = args
        self.images_flag = images
        self.spatial_flag = spatial
        self.objects_flag = objects
        self.resnet_flag = resnet
        self.obj_names_flag = obj_names
        self.return_norm_flag = return_norm # The output of the answer norm algorithm
        self.return_avsc_flag = return_avsc # Output the avsc tensor between answers in answer vocab
        self.n_objs = n_objs
        self.max_q_len = max_q_len
        # Loading Dataset
        data_root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/gqa")
        self.data_root_dir = data_root_dir
        # Tokeniser
        self.tokeniser = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        # Questions and Answers
        if split == "train":
            self.q_as = myutils.load_json(os.path.join(data_root_dir, "train_balanced_questions.json"))
            ans2idxFile = "ans2idx.pickle"
        elif split == "valid":
            self.q_as = myutils.load_json(os.path.join(data_root_dir, "val_balanced_questions.json"))
            ans2idxFile = "ans2idx.pickle"
        elif split == "train-absMixed":
            self.q_as = myutils.load_json(os.path.join(data_root_dir, "absMixed_train_questions.json"))
            ans2idxFile = "ans2idx-absMixed.pickle"
        elif split == "valid-absMixed":
            self.q_as = myutils.load_json(os.path.join(data_root_dir, "absMixed_val_questions.json"))
            ans2idxFile = "ans2idx-absMixed.pickle"
        # Ans2Idx
        #if split == "train-absMixed":
        #    print("ALLOW FULL GQA")
        #    sub_keys = list(self.q_as.keys())[:100]
        #    self.q_as = {key:self.q_as[key] for key in sub_keys}
        ans2idx_path = os.path.join(data_root_dir, ans2idxFile)
        if os.path.exists(ans2idx_path):
            self.ans2idx = myutils.load_pickle(ans2idx_path)
            self.idx2ans = {value:key for key,value in self.ans2idx.items()}
        else:
            print(f"{ans2idxFile} for this dataset split not found. generating...")
            if ans2idxFile == "ans2idx.pickle":
                # GQA
                train_path = os.path.join(data_root_dir, "train_balanced_questions.json")
                valid_path = os.path.join(data_root_dir, "val_balanced_questions.json")
            elif ans2idxFile == "ans2idx-absMixed.pickle":
                # GQA-ABSMIXED
                train_path = os.path.join(data_root_dir, "absMixed_train_questions.json")
                valid_path = os.path.join(data_root_dir, "absMixed_val_questions.json")
            self.create_ans2idx(train_path=train_path, valid_path=valid_path, save_path=ans2idx_path)
            print(f"{ans2idxFile} created! Continuing...")
            self.ans2idx = myutils.load_pickle(ans2idx_path)
            self.idx2ans = {value:key for key,value in self.ans2idx.items()}

        self.idx_2_q = {q_idx:key for q_idx, key in enumerate(self.q_as.keys())}
        # Objects
        if self.objects_flag:
            # This will be handled in __getitem__ because of h5py parallelism problem
            pass
            # TODO keep to allow analysis of objects or DEPRECATED
            #if split == "train":
            #    self.scene_graph = myutils.load_json(os.path.join(data_root_dir, "train_sceneGraphs.json"))
            #if split == "valid":
            #    self.scene_graph = myutils.load_json(os.path.join(data_root_dir, "val_sceneGraphs.json"))
        # Images
        if self.images_flag:
            raise NotImplementedError(f"This is implemented and working, but shouldnt be used right now until needed")
            self.images_root_dir = os.path.join(data_root_dir, "images")
        # Pre-extracted resnet features
        if self.resnet_flag:
            resnet_h5_path = os.path.join(data_root_dir, "resnet", "resnet.h5")
            if not os.path.exists(resnet_h5_path):
                # Preprocess resnet features
                dset_utils.frames_to_resnet_h5("GQA", resnet_h5_path)
            pass # Once again this will be handled in __getitem__ becuase of h5 parallelism problem
        # Return norm
        if self.return_norm_flag:
            # TODO DEPRECATED?? self.idx2norm = make_idx2norm(args, self.ans2idx)  
            if args.norm_gt == "nsubj": # If you get norms for answers from the subject of the question
                self.lxmert_tokeniser = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
                self.nlp = spacy.load('en_core_web_sm')
            self.norm_dict = myutils.load_norms_pickle( os.path.join(os.path.dirname(__file__),"misc/all_norms.pickle")) 
        # Return avsc tensor
        if self.return_avsc_flag:   # If using the avsc loss, generate answer tensor
            self.idx2BCE_assoc_tensor, self.idx2BCE_ctgrcl_tensor = set_avsc_loss_tensor(args, self.ans2idx) # loads norm_dict

        self.features = []
        self.features += ['images' if images else '']
        self.features += ['resnet' if resnet else '']
        self.features += ['spatial' if spatial else '']
        self.features += ['objects' if objects else '']
        self.features += ['obj_names' if obj_names else '']
        self.features += ['return_norm' if return_norm else '']
        self.features += ['return_avsc' if return_avsc else '']
        nl = "\n"
        print(f"{split}{nl}Features:{nl}{nl.join(self.features)}")

    def load_obj_h5(self):
        data_root_dir = self.data_root_dir
        self.objects_json = myutils.load_json(os.path.join(data_root_dir, "objects", "gqa_objects_info.json"))
        self.objects_h5s = {
            0:h5py.File(os.path.join(data_root_dir, "objects", "gqa_objects_0.h5"), "r", driver=None),
            1:h5py.File(os.path.join(data_root_dir, "objects", "gqa_objects_1.h5"), "r", driver=None),
            2:h5py.File(os.path.join(data_root_dir, "objects", "gqa_objects_2.h5"), "r", driver=None),
            3:h5py.File(os.path.join(data_root_dir, "objects", "gqa_objects_3.h5"), "r", driver=None),
            4:h5py.File(os.path.join(data_root_dir, "objects", "gqa_objects_4.h5"), "r", driver=None),
            5:h5py.File(os.path.join(data_root_dir, "objects", "gqa_objects_5.h5"), "r", driver=None),
            6:h5py.File(os.path.join(data_root_dir, "objects", "gqa_objects_6.h5"), "r", driver=None),
            7:h5py.File(os.path.join(data_root_dir, "objects", "gqa_objects_7.h5"), "r", driver=None),
            8:h5py.File(os.path.join(data_root_dir, "objects", "gqa_objects_8.h5"), "r", driver=None),
            9:h5py.File(os.path.join(data_root_dir, "objects", "gqa_objects_9.h5"), "r", driver=None),
            10:h5py.File(os.path.join(data_root_dir, "objects", "gqa_objects_10.h5"), "r", driver=None),
            11:h5py.File(os.path.join(data_root_dir, "objects", "gqa_objects_11.h5"), "r", driver=None),
            12:h5py.File(os.path.join(data_root_dir, "objects", "gqa_objects_12.h5"), "r", driver=None),
            13:h5py.File(os.path.join(data_root_dir, "objects", "gqa_objects_13.h5"), "r", driver=None),
            14:h5py.File(os.path.join(data_root_dir, "objects", "gqa_objects_14.h5"), "r", driver=None),
            15:h5py.File(os.path.join(data_root_dir, "objects", "gqa_objects_15.h5"), "r", driver=None)
        }

    def __len__(self):
        return len(self.q_as)

    def __getitem__(self, idx):
        if self.objects_flag:
            if not hasattr(self, 'objects_h5s'):
                self.load_obj_h5()
        if self.resnet_flag:
            if not hasattr(self, "resnet_h5"):
                self.resnet_h5 = h5py.File(os.path.join(self.data_root_dir, "resnet", "resnet.h5"), "r", driver="core") # small enough
        # Question
        q_idx = self.idx_2_q[idx]
        question = torch.LongTensor(self.tokeniser(self.q_as[q_idx]['question'])["input_ids"])
        # Answer
        answer = torch.LongTensor([ self.ans2idx[self.q_as[q_idx]['answer']] ])
        img_id = self.q_as[q_idx]['imageId']
        # Objects
        img_dims = torch.tensor([self.objects_json[img_id]['width'],self.objects_json[img_id]['height']]).long()
        if self.objects_flag:
            ih5_file, ih5_idx = self.objects_json[img_id]['file'], self.objects_json[img_id]['idx']
            bboxes = torch.from_numpy(self.objects_h5s[ih5_file]['bboxes'][ih5_idx][:self.n_objs]).round()
            bboxes[:,0]/=img_dims[0]
            bboxes[:,1]/=img_dims[1]
            bboxes[:,2]/=img_dims[0]
            bboxes[:,3]/=img_dims[1]
            features = torch.from_numpy(self.objects_h5s[ih5_file]['features'][ih5_idx][:self.n_objs])
        else:   # Create dummy inputs
            bboxes = torch.zeros(self.n_objs, 4)
            features = torch.zeros(self.n_objs, 2048)
        # Images
        if self.images_flag:
            image_path = os.path.join(self.images_root_dir, f"{img_id}.jpg")
            image = torch.from_numpy(cv2.imread(image_path)).permute(2,0,1) # (channels, height, width)
            # TODO finish images loading
        else:
            image = torch.zeros(3,244,244)
        # ResNet
        if self.resnet_flag:
            image = torch.from_numpy(self.resnet_h5[img_id]["resnet"][:2048])
        else:
            #image = torch.zeros(2048)
            image = torch.zeros(2)
        # The average norm considered of the question/answer pair
        if self.return_norm_flag:
            if self.args.norm_gt == "answer":
                try:
                    return_norm = self.norm_dict.words[self.q_as[q_idx]['answer']][self.args.norm]["sources"]["MT40k"]["scaled"] #TODO generalise this norm
                except KeyError:
                    return_norm = 0.5
            elif self.args.norm_gt == "nsubj":
                return_norm = []
                qu = myutils.clean_word(self.q_as[q_idx]['question']) # Adapted to clean entire sentences
                decoded_qu = [ str(tok) for tok in self.nlp(qu) if tok.dep_ == "nsubj" ] 
                for nsubj in decoded_qu:
                    try:
                        norm = self.norm_dict.words[nsubj][self.args.norm]["sources"]["MT40k"]["scaled"] #TODO generalise this norm
                        return_norm.append(norm)
                    except KeyError:
                        pass
                if return_norm == []: # If there is no norm for the subject of question, try the norm of the answer
                    try:
                        return_norm = self.norm_dict.words[self.q_as[q_idx]['answer']][self.args.norm]["sources"]["MT40k"]["scaled"]
                    except KeyError:
                        return_norm = 0.5 # If no norm from answer, set to 0.5 (halfway)
                else:
                    return_norm = myutils.list_avg(return_norm)
            elif self.args.norm_gt == "qtype":
                if self.q_as[q_idx]['types']['semantic'] == "global":
                    return_norm = 0.2
                else:
                    return_norm = 0.8
            elif self.args.norm_gt == "qtype-full":
                if self.q_as[q_idx]['types']['semantic'] == "global":
                    return_norm = 0.01
                else:
                    return_norm = 0.99
            return_norm = torch.Tensor([return_norm])
        else:
            return_norm = torch.Tensor([-1])
        # Return the avsc loss tensor for assoc/ctgrcl relations between answers
        if self.return_avsc_flag:
            abs_answer_tens = self.idx2BCE_assoc_tensor[self.ans2idx[self.q_as[q_idx]["answer"]]]
            conc_answer_tens = self.idx2BCE_ctgrcl_tensor[self.ans2idx[self.q_as[q_idx]["answer"]]]
        else:
            abs_answer_tens, conc_answer_tens = torch.Tensor([0]), torch.Tensor([0])
        img_id = self.q_as[q_idx]['imageId']
        if img_id[0] == "n":
            ret_img_id= torch.Tensor([0, int(img_id[1:])]).long()
        elif img_id.isnumeric():
            ret_img_id= torch.Tensor([1, int(img_id)]).long()
        else:
            raise ValueError("Something went wrong you dingus")# TODO remove this after works???
        # Question ID
        ## Give length of string to regenerate original
        q_id_ret = torch.tensor([int(q_idx), len(q_idx)]).long()
        return question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens, ret_img_id, q_id_ret, img_dims


    # UTILITY FUNCTIONS
    def create_ans2idx(self, train_path, valid_path, save_path):
        answers = []
        train_questions = myutils.load_json(train_path)
        valid_questions = myutils.load_json(valid_path) 
        for idx, key in tqdm(enumerate(train_questions.keys()), total=len(train_questions)):
            answers.append(train_questions[key]['answer'])
        for idx, key in tqdm(enumerate(valid_questions.keys()), total=len(valid_questions)):
            answers.append(valid_questions[key]['answer'])
        answers = list(set(answers))
        ans2idx = {answer:a_idx for a_idx, answer in enumerate(answers)}
        myutils.save_pickle(ans2idx, save_path)


"""
Pytorch_Lightning Model handling system
"""
######################################################
######################################################
# Models
######################################################
######################################################
# Pytorch_Lightning is a package that cleanly handles training and testing pytorch models. Check their website https://www.pytorchlightning.ai/
#class Basic(pl.LightningModule):
#    def __init__(self, args, n_answers):
#        super().__init__()
#        self.args = args
#        self.lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
#        fc_intermediate = ((n_answers-768)//2)+768
#        self.classifier_fc = nn.Sequential(
#            nn.Dropout(0.2),
#            nn.Linear(768, fc_intermediate),
#            nn.BatchNorm1d(fc_intermediate),
#            nn.GELU(),
#            nn.Dropout(0.2),
#            nn.Linear(fc_intermediate, n_answers+1)   # n+1 (includes unknown answer token)
#        )
#        if args.unfreeze == "all":
#            pass
#        elif args.unfreeze == "heads":
#            for param in self.lxmert.base_model.parameters():
#                param.requires_grad = False
#        elif args.unfreeze == "none":
#            for param in self.lxmert.parameters():
#                param.requires_grad = False
#        self.criterion = nn.CrossEntropyLoss()
#        self.valid_acc = pl.metrics.Accuracy()
#        self.train_acc = pl.metrics.Accuracy()
#
#
#    def forward(self, question, bboxes, features):
#        out = self.lxmert(question, features, bboxes)[2]    #['language_output', 'vision_output', 'pooled_output']
#        out = self.classifier_fc(out)
#        return out
#
#    def configure_optimizers(self):
#        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
#        return optimizer
#
#    def training_step(self, train_batch, batch_idx):
#        # Prepare data
#        question, answer, bboxes, features, image = train_batch
#        out = self(question, bboxes, features)
#        train_loss = self.criterion(out, answer.squeeze(1))
#        out = F.softmax(out, dim=1)
#        self.log("train_loss", train_loss, prog_bar=True, on_step=False, on_epoch=True)
#        self.log("train_acc", self.train_acc(out, answer.squeeze(1)), prog_bar=True, on_step=False, on_epoch=True)
#        return train_loss
#
#    def validation_step(self, val_batch, batch_idx):
#        question, answer, bboxes, features, image = val_batch
#        out = self(question, bboxes, features)
#        valid_loss = self.criterion(out, answer.squeeze(1))
#        out = F.softmax(out, dim=1)
#        raise NotImplementedError("Check if you should use softmax or not use softmax for loss calculation")
#        self.log("valid_loss", valid_loss, on_step=False, on_epoch=True)
#        self.log("valid_acc", self.valid_acc(out, answer.squeeze(1)), prog_bar=True, on_step=False, on_epoch=True)
#        return valid_loss
#
#
#class LxLSTM(pl.LightningModule):
#    def __init__(self, args, n_answers, ans2idx):   # Pass ans2idx from relevant dataset object
#        super().__init__()
#        self.args = args
#        self.lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
#        # Language/Vision LSTM
#        self.lng_lstm = nn.LSTM(768, 1024, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
#        self.vis_lstm = nn.LSTM(768, 1024, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
#        fc_intermediate = ((n_answers-8960)//2)+8960
#        self.classifier_fc = nn.Sequential(
#            nn.Dropout(0.2),
#            nn.Linear(8960, fc_intermediate),
#            nn.BatchNorm1d(fc_intermediate),
#            nn.GELU(),
#            nn.Dropout(0.2),
#            nn.Linear(fc_intermediate, n_answers+1)
#        )
#        for name, param in self.lxmert.named_parameters():
#            param.requires_grad = True
#        if args.unfreeze == "all":
#            pass
#        elif args.unfreeze == "heads":
#            for name, param in self.lxmert.named_parameters():
#                if not("attention" in name):
#                    param.requires_grad = False
#        elif args.unfreeze == "none":
#            for name, param in self.lxmert.named_parameters():
#                param.requires_grad = False
#        if args.loss == "default":
#            self.criterion = nn.CrossEntropyLoss()#reduction='none')
#        elif args.loss == "avsc":
#            self.criterion = nn.BCEWithLogitsLoss()#reduction='none')
#        else:
#            raise NotImplementedError(f"Loss {args.loss} not implement for Hopfield_3 net")
#        self.valid_acc = pl.metrics.Accuracy()
#        self.train_acc = pl.metrics.Accuracy()
#
#
#    def forward(self, question, bboxes, features, image):
#        # Process language
#        out = self.lxmert(question, features, bboxes)       #['language_output', 'vision_output', 'pooled_output']
#        lng_out, vis_out, x_out = out['language_output'], out['vision_output'], out['pooled_output']
#        # x stands for 'cross', see naming scheme in documentation
#        # Language/Vision LSTM processing
#        _, (_, lng_out) = self.lng_lstm(lng_out)
#        _, (_, vis_out) = self.vis_lstm(vis_out)
#        lng_out = lng_out.permute(1,0,2).contiguous().view(self.args.bsz, -1)
#        vis_out = vis_out.permute(1,0,2).contiguous().view(self.args.bsz, -1)
#        out = torch.cat((lng_out, vis_out, x_out), dim=1)
#        out = self.classifier_fc(out)
#        return out
#
#    def configure_optimizers(self):
#        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
#        return optimizer
#
#    def training_step(self, train_batch, batch_idx):
#        # Prepare data
#        question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens, _ = train_batch
#        out = self(question, bboxes, features, image)
#        train_loss = self.criterion(out, answer.squeeze(1))
#        out = F.softmax(out, dim=1)
#        self.log("train_loss", train_loss, prog_bar=True, on_step=False, on_epoch=True)
#        self.log("train_acc", self.train_acc(F.softmax(out, dim=1), answer.squeeze(1)), prog_bar=True, on_step=False, on_epoch=True)
#        return train_loss
#
#    def validation_step(self, val_batch, batch_idx):
#        question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens, _ = val_batch
#        out = self(question, bboxes, features, image)
#        valid_loss = self.criterion(out, answer.squeeze(1))
#        out = F.softmax(out, dim=1)
#        self.log("valid_loss", valid_loss, prog_bar=True, on_step=False, on_epoch=True)
#        self.log("valid_acc", self.valid_acc(F.softmax(out, dim=1), answer.squeeze(1)), prog_bar=True, on_step=False, on_epoch=True)
#        return valid_loss
#
#
#
#class BERTLSTM(pl.LightningModule):
#    def __init__(self, args, n_answers):
#        super().__init__()
#        self.args = args
#        self.bert = BertModel.from_pretrained('bert-base-uncased')       
#        self.vis_lstm = nn.LSTM(2048, 1024, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
#        fc_intermediate = ((n_answers-4864)//2)+4864
#        self.classifier_fc = nn.Sequential(
#            nn.Dropout(0.2),
#            nn.Linear(4864, fc_intermediate),
#            nn.BatchNorm1d(fc_intermediate),
#            nn.GELU(),
#            nn.Dropout(0.2),
#            nn.Linear(fc_intermediate, n_answers+1)   # n+1 (includes unknown answer token)
#        )
#        if args.unfreeze == "all":
#            pass
#        elif args.unfreeze == "heads":
#            for param in self.bert.base_model.parameters():
#                param.requires_grad = False
#        elif args.unfreeze == "none":
#            for param in self.bert.parameters():
#                param.requires_grad = False
#        self.criterion = nn.CrossEntropyLoss()
#        self.valid_acc = pl.metrics.Accuracy()
#        self.train_acc = pl.metrics.Accuracy()
#
#
#    def forward(self, question, bboxes, features):
#        lng_out = self.bert(question)
#        lng_out = lng_out[1]
#        _, (_, vis_out) = self.vis_lstm(features)    # output, (hn, cn)
#        vis_out = vis_out.permute(1,0,2)
#        vis_out = vis_out.contiguous().view(self.args.bsz, -1)
#        combined_out = torch.cat((lng_out, vis_out), 1) # 8092
#        out = self.classifier_fc(combined_out)
#        return out
#
#    def configure_optimizers(self):
#        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
#        return optimizer
#
#    def training_step(self, train_batch, batch_idx):
#        # Prepare data
#        question, answer, bboxes, features, image = train_batch
#        out = self(question, bboxes, features)
#        train_loss = self.criterion(out, answer.squeeze(1))
#        out = F.softmax(out, dim=1)
#        self.log("train_loss", train_loss, prog_bar=True, on_step=False, on_epoch=True)
#        self.log("train_acc", self.train_acc(out, answer.squeeze(1)), prog_bar=True, on_step=False, on_epoch=True)
#        return train_loss
#
#    def validation_step(self, val_batch, batch_idx):
#        question, answer, bboxes, features, image = val_batch
#        out = self(question, bboxes, features)
#        valid_loss = self.criterion(out, answer.squeeze(1))
#        out = F.softmax(out, dim=1)
#        self.log("valid_loss", valid_loss, on_step=False, on_epoch=True)
#        self.log("valid_acc", self.valid_acc(out, answer.squeeze(1)), prog_bar=True, on_step=False, on_epoch=True)
#        return valid_loss
#
#
#
## k / (1-k) induction
#class Induction(pl.LightningModule):
#    def __init__(self, args, n_answers, ans2idx):
#        super().__init__()
#        self.args = args
#        if args.loss == "avsc":
#            raise NotImplementedError(f"Not implemented this with avsc loss")
#        fc_intermediate = ((n_answers-768)//2)+768
#         # High-norm / low-norm may mean high abstract/concrete. But generalised for other norms
#        self.lxmert_lownorm = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
#        self.lxmert_highnorm = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
#        self.low_classifier_fc = nn.Sequential(
#            nn.Dropout(0.2),
#            nn.Linear(768, fc_intermediate),
#            nn.BatchNorm1d(fc_intermediate),
#            nn.GELU(),
#            nn.Dropout(0.2),
#            nn.Linear(fc_intermediate, n_answers+1)   #GQA has 1842 unique answers, so we pass in 1841
#        )
#        self.high_classifier_fc = nn.Sequential(
#            nn.Dropout(0.2),
#            nn.Linear(768, fc_intermediate),
#            nn.BatchNorm1d(fc_intermediate),
#            nn.GELU(),
#            nn.Dropout(0.2),
#            nn.Linear(fc_intermediate, n_answers+1)
#        )
#        if args.unfreeze == "all":
#            pass
#        elif args.unfreeze == "heads":
#            for param in self.lxmert_highnorm.base_model.parameters():
#                param.requires_grad = False
#            for param in self.lxmert_lownorm.base_model.parameters():
#                param.requires_grad = False
#        elif args.unfreeze == "none":
#            for param in self.lxmert_highnorm.parameters():
#                param.requires_grad = False
#            for param in self.lxmert_lownorm.parameters():
#                param.requires_grad = False
#        self.criterion = nn.CrossEntropyLoss(reduction='none')
#        self.valid_acc = pl.metrics.Accuracy()
#        self.valid_low_acc = pl.metrics.Accuracy()
#        self.valid_high_acc = pl.metrics.Accuracy()
#        self.train_acc = pl.metrics.Accuracy()
#        self.train_low_acc = pl.metrics.Accuracy()
#        self.train_high_acc = pl.metrics.Accuracy()
#
#        # TODO Deprecated self.idx2norm = make_idx2norm(args, ans2idx)  
#
#
#    def forward(self, question, bboxes, features):
#        out_low = self.lxmert_lownorm(question, features, bboxes)[2]    #['language_output', 'vision_output', 'pooled_output']
#        out_high = self.lxmert_highnorm(question, features, bboxes)[2]    #['language_output', 'vision_output', 'pooled_output']
#        out_low = self.low_classifier_fc(out_low)
#        out_high = self.high_classifier_fc(out_high)
#        return out_low, out_high
#
#    def configure_optimizers(self):
#        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
#        return optimizer
#
#    def training_step(self, train_batch, batch_idx):
#        # Prepare data
#        raise NotImplementedError("Check this works")
#        question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens = train_batch
#        high_norms = return_norm
#        low_norms = torch.ones(len(high_norms)).to(self.device)
#        low_norms = low_norms - high_norms
#        out_low, out_high = self(question, bboxes, features)
#        low_loss = self.criterion(out_low, answer.squeeze(1))
#        high_loss = self.criterion(out_high, answer.squeeze(1))
#        low_loss = torch.dot(low_norms, low_loss) / len(low_loss)
#        high_loss = torch.dot(high_norms, high_loss) / len(high_loss)
#        train_loss = low_loss + high_loss
#        out_high = F.softmax(out_high, dim=1)
#        out_low = F.softmax(out_low, dim=1)
#        self.log("train_loss", train_loss, prog_bar=True, on_step=False, on_epoch=True)
#        self.log("train_low_loss", low_loss, on_step=False, on_epoch=True)
#        self.log("train_high_loss", high_loss, on_step=False, on_epoch=True)
#        self.log("train_acc", self.train_acc(F.softmax(out_high+out_low, dim=1), answer.squeeze(1)), prog_bar=True, on_step=False, on_epoch=True)
#        self.log("train_low_acc", self.train_acc(out_low, answer.squeeze(1)), on_step=False, on_epoch=True)
#        self.log("train_high_acc", self.train_acc(out_high, answer.squeeze(1)), on_step=False, on_epoch=True)
#        return train_loss
#
#    def validation_step(self, val_batch, batch_idx):
#        raise NotImplementedError("Check this works")
#        question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens = val_batch
#        high_norms = return_norm
#        low_norms = torch.ones(len(high_norms)).to(self.device)
#        low_norms = low_norms - high_norms
#        out_low, out_high = self(question, bboxes, features)
#        low_loss = self.criterion(out_low, answer.squeeze(1))
#        high_loss = self.criterion(out_high, answer.squeeze(1))
#        low_loss = torch.dot(low_norms, low_loss) / len(low_loss)
#        high_loss = torch.dot(high_norms, high_loss) / len(high_loss)
#        valid_loss = low_loss + high_loss
#        out_high = F.softmax(out_high, dim=1)
#        out_low = F.softmax(out_low, dim=1)
#        self.log("valid_loss", valid_loss, prog_bar=True, on_step=False, on_epoch=True)
#        self.log("valid_low_loss", low_loss, on_step=False, on_epoch=True)
#        self.log("valid_high_loss", high_loss, on_step=False, on_epoch=True)
#        self.log("valid_acc", self.valid_acc(F.softmax(out_high+out_low, dim=1), answer.squeeze(1)), prog_bar=True, on_step=False, on_epoch=True)
#        self.log("valid_low_acc", self.valid_acc(out_low, answer.squeeze(1)), on_step=False, on_epoch=True)
#        self.log("valid_high_acc", self.valid_acc(out_high, answer.squeeze(1)), on_step=False, on_epoch=True)
#        return valid_loss
#
#
#
#
#class Hopfield_0(pl.LightningModule):
#    def __init__(self, args, n_answers, ans2idx):   # Pass ans2idx from relevant dataset object
#        super().__init__()
#        self.args = args
#        # Concrete: Higher scaling beta to assert more discrete store states
#        self.high_hopfield = hpf.Hopfield(input_size = 4864, hidden_size = 1024, output_size = 1024, pattern_size = 1, num_heads = 7, scaling = args.hopfield_beta_high, update_steps_max = 3, update_steps_eps = 1e-4, dropout = 0.2)
#        # Abstract: lower scaling beta to allow more metastable/global state
#        self.low_hopfield = hpf.Hopfield(input_size = 4864, hidden_size = 1024, output_size = 1024, pattern_size = 1, num_heads = 7, scaling = args.hopfield_beta_low, update_steps_max = 3, update_steps_eps = 1e-4, dropout = 0.2)
#
#        self.vis_lstm = nn.LSTM(2048, 1024, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
#        self.bert = BertModel.from_pretrained('bert-base-uncased')       
#        fc_intermediate = ((n_answers-1024)//2)+1024
#
#         # High-norm / low-norm may mean high abstract/concrete. But generalised for other norms
#        self.low_classifier_fc = nn.Sequential(
#            nn.Dropout(0.2),
#            nn.Linear(1024, fc_intermediate),
#            nn.BatchNorm1d(fc_intermediate),
#            nn.GELU(),
#            nn.Dropout(0.2),
#            nn.Linear(fc_intermediate, n_answers+1)   #GQA has 1842 unique answers, so we pass in 1841
#        )
#        self.high_classifier_fc = nn.Sequential(
#            nn.Dropout(0.2),
#            nn.Linear(1024, fc_intermediate),
#            nn.BatchNorm1d(fc_intermediate),
#            nn.GELU(),
#            nn.Dropout(0.2),
#            nn.Linear(fc_intermediate, n_answers+1)
#        )
#        if args.unfreeze == "all":
#            pass
#        elif args.unfreeze == "heads":
#            for param in self.bert.base_model.parameters():
#                param.requires_grad = False
#        elif args.unfreeze == "none":
#            for param in self.bert.parameters():
#                param.requires_grad = False
#        if args.loss == "default":
#            self.criterion = nn.CrossEntropyLoss(reduction='none')
#        elif args.loss == "avsc":
#            self.criterion = nn.BCEWithLogitsLoss(reduction='none')
#        else:
#            raise NotImplementedError(f"Loss {args.loss} not implement for Hopfield_0 net")
#        self.valid_acc = pl.metrics.Accuracy()
#        self.valid_low_acc = pl.metrics.Accuracy()
#        self.valid_high_acc = pl.metrics.Accuracy()
#        self.train_acc = pl.metrics.Accuracy()
#        self.train_low_acc = pl.metrics.Accuracy()
#        self.train_high_acc = pl.metrics.Accuracy()
#
#
#    def forward(self, question, bboxes, features):
#        lng_out = self.bert(question)
#        lng_out = lng_out[1]
#        _, (_, vis_out) = self.vis_lstm(features)    # output, (hn, cn)
#        vis_out = vis_out.permute(1,0,2)
#        vis_out = vis_out.contiguous().view(self.args.bsz, -1)
#        combined_out = torch.cat((lng_out, vis_out), 1).unsqueeze(1) # 4864
#        out_low = self.low_hopfield(combined_out)
#        out_high = self.high_hopfield(combined_out)
#        out_low = out_low.squeeze(1)
#        out_high = out_high.squeeze(1)
#        out_low = self.low_classifier_fc(out_low)
#        out_high = self.high_classifier_fc(out_high)
#        return out_low, out_high
#
#    def configure_optimizers(self):
#        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
#        return optimizer
#
#    def training_step(self, train_batch, batch_idx):
#        # Prepare data
#        raise NotImplementedError("Check this work")
#        question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens = train_batch
#        high_norms = return_norm
#        low_norms = torch.ones(len(high_norms)).to(self.device)
#        low_norms = low_norms - high_norms
#        out_low, out_high = self(question, bboxes, features, image)
#        if self.args.loss == "default":
#            low_loss = self.criterion(out_low, answer.squeeze(1))
#            high_loss = self.criterion(out_high, answer.squeeze(1))
#        elif self.args.loss == "avsc":
#            low_loss = torch.mean(self.criterion(out_low, abs_answer_tens), 1)
#            high_loss = torch.mean(self.criterion(out_high, conc_answer_tens), 1)
#        low_loss = torch.dot(low_norms, low_loss) / len(low_loss)
#        high_loss = torch.dot(high_norms, high_loss) / len(high_loss)
#        train_loss = low_loss + high_loss
#        out_high = F.softmax(out_high, dim=1)
#        out_low = F.softmax(out_low, dim=1)
#        self.log("train_loss", train_loss, prog_bar=True, on_step=False, on_epoch=True)
#        self.log("train_low_loss", low_loss, on_step=False, on_epoch=True)
#        self.log("train_high_loss", high_loss, on_step=False, on_epoch=True)
#        self.log("train_acc", self.train_acc(F.softmax(out_high+out_low, dim=1), answer.squeeze(1)), prog_bar=True, on_step=False, on_epoch=True)
#        self.log("train_low_acc", self.train_acc(out_low, answer.squeeze(1)), on_step=False, on_epoch=True)
#        self.log("train_high_acc", self.train_acc(out_high, answer.squeeze(1)), on_step=False, on_epoch=True)
#        return train_loss
#
#    def validation_step(self, val_batch, batch_idx):
#        raise NotImplementedError("Check this work")
#        question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens = val_batch
#        high_norms = return_norm
#        low_norms = torch.ones(len(high_norms)).to(self.device)
#        low_norms = low_norms - high_norms
#        out_low, out_high = self(question, bboxes, features, image)
#        if self.args.loss == "default":
#            low_loss = self.criterion(out_low, answer.squeeze(1))
#            high_loss = self.criterion(out_high, answer.squeeze(1))
#        elif self.args.loss == "avsc":
#            low_loss = torch.mean(self.criterion(out_low, abs_answer_tens), 1)
#            high_loss = torch.mean(self.criterion(out_high, conc_answer_tens), 1)
#        low_loss = torch.dot(low_norms, low_loss) / len(low_loss)
#        high_loss = torch.dot(high_norms, high_loss) / len(high_loss)
#        valid_loss = low_loss + high_loss
#        out_high = F.softmax(out_high, dim=1)
#        out_low = F.softmax(out_low, dim=1)
#        self.log("valid_loss", valid_loss, prog_bar=True, on_step=False, on_epoch=True)
#        self.log("valid_low_loss", low_loss, on_step=False, on_epoch=True)
#        self.log("valid_high_loss", high_loss, on_step=False, on_epoch=True)
#        self.log("valid_acc", self.valid_acc(F.softmax(out_high+out_low, dim=1), answer.squeeze(1)), prog_bar=True, on_step=False, on_epoch=True)
#        self.log("valid_low_acc", self.valid_acc(out_low, answer.squeeze(1)), on_step=False, on_epoch=True)
#        self.log("valid_high_acc", self.valid_acc(out_high, answer.squeeze(1)), on_step=False, on_epoch=True)
#        return valid_loss
#
#
#
#
#class Hopfield_1(pl.LightningModule):
#    def __init__(self, args, n_answers, ans2idx):   # Pass ans2idx from relevant dataset object
#        super().__init__()
#        self.args = args
#        # Concrete: Higher scaling beta to assert more discrete store states
#        self.high_hopfield = hpf.Hopfield(input_size = 4096, hidden_size = 1024, output_size = 1024, pattern_size = 1, num_heads = 7, scaling = args.hopfield_beta_high, update_steps_max = 3, update_steps_eps = 1e-4, dropout = 0.2)
#        # Abstract: lower scaling beta to allow more metastable/global state
#        self.low_hopfield = hpf.Hopfield(input_size = 4096, hidden_size = 1024, output_size = 1024, pattern_size = 1, num_heads = 7, scaling = args.hopfield_beta_low, update_steps_max = 3, update_steps_eps = 1e-4, dropout = 0.2)
#        #bert_config = BertConfig(hidden_size=2048, num_attention_heads=8)  # Upsize to match visual features for BiDaf
#        self.bert = BertModel.from_pretrained('bert-base-uncased')
#        self.bert_fc = nn.Linear(768, 2048)
#        self.bidaf = BidafAttn(None, method="dot")
#        self.lstm = nn.LSTM(2048, 1024, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
#        fc_intermediate = ((n_answers-1024)//2)+1024
#
#         # High-norm / low-norm may mean high abstract/concrete. But generalised for other norms
#        self.low_classifier_fc = nn.Sequential(
#            nn.Dropout(0.2),
#            nn.Linear(1024, fc_intermediate),
#            nn.BatchNorm1d(fc_intermediate),
#            nn.GELU(),
#            nn.Dropout(0.2),
#            nn.Linear(fc_intermediate, n_answers+1)   #GQA has 1842 unique answers, so we pass in 1841
#        )
#        self.high_classifier_fc = nn.Sequential(
#            nn.Dropout(0.2),
#            nn.Linear(1024, fc_intermediate),
#            nn.BatchNorm1d(fc_intermediate),
#            nn.GELU(),
#            nn.Dropout(0.2),
#            nn.Linear(fc_intermediate, n_answers+1)
#        )
#        if args.unfreeze == "all":
#            pass
#        elif args.unfreeze == "heads":
#            for param in self.bert.base_model.parameters():
#                param.requires_grad = False
#        elif args.unfreeze == "none":
#            for param in self.bert.parameters():
#                param.requires_grad = False
#        if args.loss == "default":
#            self.criterion = nn.CrossEntropyLoss(reduction='none')
#        elif args.loss == "avsc":
#            self.criterion = nn.BCEWithLogitsLoss(reduction='none')
#        else:
#            raise NotImplementedError(f"Loss {args.loss} not implement for Hopfield_1 net")
#        self.valid_acc = pl.metrics.Accuracy()
#        self.valid_low_acc = pl.metrics.Accuracy()
#        self.valid_high_acc = pl.metrics.Accuracy()
#        self.train_acc = pl.metrics.Accuracy()
#        self.train_low_acc = pl.metrics.Accuracy()
#        self.train_high_acc = pl.metrics.Accuracy()
#
#
#    def forward(self, question, bboxes, features):
#        lng_out = self.bert(question)
#        lng_out = lng_out[0]
#        lng_out = self.bert_fc(lng_out)
#        lng_out_l = lng_out.shape[1]
#        features_l = features.shape[1]
#        lng_out_l = torch.LongTensor([lng_out_l]*self.args.bsz)
#        features_l = torch.LongTensor([features_l]*self.args.bsz)
#        bidaf_out = self.bidaf(lng_out, lng_out_l, features, features_l)
#        bidaf_out = bidaf_out[0]
#        _, (_, lstm_out) = self.lstm(bidaf_out) # output, (hn, cn)
#        lstm_out = lstm_out.permute(1,0,2)
#        lstm_out = lstm_out.contiguous().view(self.args.bsz, -1).unsqueeze(1)
#        out_low = self.low_hopfield(lstm_out)
#        out_high = self.high_hopfield(lstm_out)
#        out_low = out_low.squeeze(1)
#        out_high = out_high.squeeze(1)
#        out_low = self.low_classifier_fc(out_low)
#        out_high = self.high_classifier_fc(out_high)
#        return out_low, out_high
#
#    def configure_optimizers(self):
#        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
#        return optimizer
#
#    def training_step(self, train_batch, batch_idx):
#        raise NotImplementedError("Check this works")
#        # Prepare data
#        question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens = train_batch
#        high_norms = return_norm
#        low_norms = torch.ones(len(high_norms)).to(self.device)
#        low_norms = low_norms - high_norms
#        out_low, out_high = self(question, bboxes, features, image)
#        if self.args.loss == "default":
#            low_loss = self.criterion(out_low, answer.squeeze(1))
#            high_loss = self.criterion(out_high, answer.squeeze(1))
#        elif self.args.loss == "avsc":
#            low_loss = torch.mean(self.criterion(out_low, abs_answer_tens), 1)
#            high_loss = torch.mean(self.criterion(out_high, conc_answer_tens), 1)
#        low_loss = torch.dot(low_norms, low_loss) / len(low_loss)
#        high_loss = torch.dot(high_norms, high_loss) / len(high_loss)
#        train_loss = low_loss + high_loss
#        out_high = F.softmax(out_high, dim=1)
#        out_low = F.softmax(out_low, dim=1)
#        self.log("train_loss", train_loss, prog_bar=True, on_step=False, on_epoch=True)
#        self.log("train_low_loss", low_loss, on_step=False, on_epoch=True)
#        self.log("train_high_loss", high_loss, on_step=False, on_epoch=True)
#        self.log("train_acc", self.train_acc(F.softmax(out_high+out_low, dim=1), answer.squeeze(1)), prog_bar=True, on_step=False, on_epoch=True)
#        self.log("train_low_acc", self.train_acc(out_low, answer.squeeze(1)), on_step=False, on_epoch=True)
#        self.log("train_high_acc", self.train_acc(out_high, answer.squeeze(1)), on_step=False, on_epoch=True)
#        return train_loss
#
#    def validation_step(self, val_batch, batch_idx):
#        question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens = val_batch
#        high_norms = return_norm
#        low_norms = torch.ones(len(high_norms)).to(self.device)
#        low_norms = low_norms - high_norms
#        out_low, out_high = self(question, bboxes, features, image)
#        if self.args.loss == "default":
#            low_loss = self.criterion(out_low, answer.squeeze(1))
#            high_loss = self.criterion(out_high, answer.squeeze(1))
#        elif self.args.loss == "avsc":
#            low_loss = torch.mean(self.criterion(out_low, abs_answer_tens), 1)
#            high_loss = torch.mean(self.criterion(out_high, conc_answer_tens), 1)
#        low_loss = torch.dot(low_norms, low_loss) / len(low_loss)
#        high_loss = torch.dot(high_norms, high_loss) / len(high_loss)
#        valid_loss = low_loss + high_loss
#        out_high = F.softmax(out_high, dim=1)
#        out_low = F.softmax(out_low, dim=1)
#        self.log("valid_loss", valid_loss, prog_bar=True, on_step=False, on_epoch=True)
#        self.log("valid_low_loss", low_loss, on_step=False, on_epoch=True)
#        self.log("valid_high_loss", high_loss, on_step=False, on_epoch=True)
#        self.log("valid_acc", self.valid_acc(F.softmax(out_high+out_low, dim=1), answer.squeeze(1)), prog_bar=True, on_step=False, on_epoch=True)
#        self.log("valid_low_acc", self.valid_acc(out_low, answer.squeeze(1)), on_step=False, on_epoch=True)
#        self.log("valid_high_acc", self.valid_acc(out_high, answer.squeeze(1)), on_step=False, on_epoch=True)
#        return valid_loss
#
#
#
#
#class Hopfield_2(pl.LightningModule):
#    def __init__(self, args, n_answers, ans2idx):   # Pass ans2idx from relevant dataset object
#        super().__init__()
#        self.args = args
#        # Bert question processing
#        self.bert = BertModel.from_pretrained('bert-base-uncased')
#        self.bert_fc = nn.Linear(768, 2048)
#        # Torchvision ResNet
#        # TODO DEPRECATED??
#        #raise NotImplementedError(f"Process the h5 file for GQA and VQA-CP. Update the dataloader. ")
#        #self.img_cnn = resnet101(pretrained=True)
#        #self.img_cnn = resnet50(pretrained=True)
#        #self.img_cnn.fc = myutils.Identity() # Really cool trick, read myutils for explanation
#        #for param in self.img_cnn.parameters():
#        #    param.requires_grad = False
#        # Concrete: Higher scaling beta to assert more discrete store states
#        self.high_hopfield = hpf.Hopfield(input_size = 4096, hidden_size = 1024, output_size = 1024, pattern_size = 1, num_heads = 7, scaling = args.hopfield_beta_high, update_steps_max = 3, update_steps_eps = 1e-4, dropout = 0.2)
#        self.high_bidaf = BidafAttn(None, method="dot")
#        self.high_lstm = nn.LSTM(2048, 1024, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
#        # Abstract: lower scaling beta to allow more metastable/global state
#        self.low_hopfield = hpf.Hopfield(input_size = 4096, hidden_size = 1024, output_size = 1024, pattern_size = 1, num_heads = 7, scaling = args.hopfield_beta_low, update_steps_max = 3, update_steps_eps = 1e-4, dropout = 0.2)
#        self.low_bidaf = BidafAttn(None, method="dot")
#        self.low_lstm = nn.LSTM(2048, 1024, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
#        fc_intermediate = ((n_answers-1024)//2)+1024
#
#         # High-norm / low-norm may mean high abstract/concrete. But generalised for other norms
#        self.low_classifier_fc = nn.Sequential(
#            nn.Dropout(0.2),
#            nn.Linear(1024, fc_intermediate),
#            nn.BatchNorm1d(fc_intermediate),
#            nn.GELU(),
#            nn.Dropout(0.2),
#            nn.Linear(fc_intermediate, n_answers+1)   #GQA has 1842 unique answers, so we pass in 1841
#        )
#        self.high_classifier_fc = nn.Sequential(
#            nn.Dropout(0.2),
#            nn.Linear(1024, fc_intermediate),
#            nn.BatchNorm1d(fc_intermediate),
#            nn.GELU(),
#            nn.Dropout(0.2),
#            nn.Linear(fc_intermediate, n_answers+1)
#        )
#        if args.unfreeze == "all":
#            pass
#        elif args.unfreeze == "heads":
#            for param in self.bert.base_model.parameters():
#                param.requires_grad = False
#        elif args.unfreeze == "none":
#            for param in self.bert.parameters():
#                param.requires_grad = False
#        if args.loss == "default":
#            self.criterion = nn.CrossEntropyLoss(reduction='none')
#        elif args.loss == "avsc":
#            self.criterion = nn.BCEWithLogitsLoss(reduction='none')
#        else:
#            raise NotImplementedError(f"Loss {args.loss} not implement for Hopfield_2 net")
#        self.valid_acc = pl.metrics.Accuracy()
#        self.valid_low_acc = pl.metrics.Accuracy()
#        self.valid_high_acc = pl.metrics.Accuracy()
#        self.train_acc = pl.metrics.Accuracy()
#        self.train_low_acc = pl.metrics.Accuracy()
#        self.train_high_acc = pl.metrics.Accuracy()
#
#
#    def forward(self, question, bboxes, features, image):
#        # Process language
#        lng_out = self.bert(question)
#        lng_out = lng_out[0]
#        lng_out = self.bert_fc(lng_out)
#        lng_out_l = lng_out.shape[1]
#        # Process image
#        # TODO Deprecated? image_feat = self.img_cnn(image).unsqueeze(1)
#        image_feat = image.unsqueeze(1)
#        image_l = torch.LongTensor([1]*self.args.bsz)
#        #Get lengths
#        features_l = features.shape[1]
#        lng_out_l = torch.LongTensor([lng_out_l]*self.args.bsz)
#        features_l = torch.LongTensor([features_l]*self.args.bsz)
#        # High stream (objects)
#        high_bidaf_out = self.high_bidaf(lng_out, lng_out_l, features, features_l)
#        high_bidaf_out = high_bidaf_out[0]
#        _, (_, high_lstm_out) = self.high_lstm(high_bidaf_out) # output, (hn, cn)
#        high_lstm_out = high_lstm_out.permute(1,0,2)
#        high_lstm_out = high_lstm_out.contiguous().view(self.args.bsz, -1).unsqueeze(1)
#        # Low Stream (image)
#        low_bidaf_out = self.low_bidaf(lng_out, lng_out_l, image_feat, image_l)
#        low_bidaf_out = low_bidaf_out[0]
#        _, (_, low_lstm_out) = self.low_lstm(low_bidaf_out)
#        low_lstm_out = low_lstm_out.permute(1,0,2)
#        low_lstm_out = low_lstm_out.contiguous().view(self.args.bsz, -1).unsqueeze(1)
#        # Hopfields and FC
#        out_low = self.low_hopfield(low_lstm_out)
#        out_high = self.high_hopfield(high_lstm_out)
#        out_low = out_low.squeeze(1)
#        out_high = out_high.squeeze(1)
#        out_low = self.low_classifier_fc(out_low)
#        out_high = self.high_classifier_fc(out_high)
#        return out_low, out_high
#
#    def configure_optimizers(self):
#        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
#        return optimizer
#
#    def training_step(self, train_batch, batch_idx):
#        # Prepare data
#        question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens = train_batch
#        high_norms = return_norm
#        low_norms = torch.ones(len(high_norms)).to(self.device)
#        low_norms = low_norms - high_norms
#        out_low, out_high = self(question, bboxes, features, image)
#        if self.args.loss == "default":
#            low_loss = self.criterion(out_low, answer.squeeze(1))
#            high_loss = self.criterion(out_high, answer.squeeze(1))
#        elif self.args.loss == "avsc":
#            low_loss = torch.mean(self.criterion(out_low, abs_answer_tens), 1)
#            high_loss = torch.mean(self.criterion(out_high, conc_answer_tens), 1)
#        low_loss = torch.dot(low_norms, low_loss) / len(low_loss)
#        high_loss = torch.dot(high_norms, high_loss) / len(high_loss)
#        train_loss = low_loss + high_loss
#        out_high = F.softmax(out_high, dim=1)
#        out_low = F.softmax(out_low, dim=1)
#        self.log("train_loss", train_loss, prog_bar=True, on_step=False, on_epoch=True)
#        self.log("train_low_loss", low_loss, on_step=False, on_epoch=True)
#        self.log("train_high_loss", high_loss, on_step=False, on_epoch=True)
#        self.log("train_acc", self.train_acc(F.softmax(out_high+out_low, dim=1), answer.squeeze(1)), prog_bar=True, on_step=False, on_epoch=True)
#        self.log("train_low_acc", self.train_acc(out_low, answer.squeeze(1)), on_step=False, on_epoch=True)
#        self.log("train_high_acc", self.train_acc(out_high, answer.squeeze(1)), on_step=False, on_epoch=True)
#        return train_loss
#
#    def validation_step(self, val_batch, batch_idx):
#        raise NotImplementedError("Check this is implemented properly")
#        question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens = val_batch
#        high_norms = return_norm
#        low_norms = torch.ones(len(high_norms)).to(self.device)
#        low_norms = low_norms - high_norms
#        out_low, out_high = self(question, bboxes, features, image)
#        if self.args.loss == "default":
#            low_loss = self.criterion(out_low, answer.squeeze(1))
#            high_loss = self.criterion(out_high, answer.squeeze(1))
#        elif self.args.loss == "avsc":
#            low_loss = torch.mean(self.criterion(out_low, abs_answer_tens), 1)
#            high_loss = torch.mean(self.criterion(out_high, conc_answer_tens), 1)
#        low_loss = torch.dot(low_norms, low_loss) / len(low_loss)
#        high_loss = torch.dot(high_norms, high_loss) / len(high_loss)
#        valid_loss = low_loss + high_loss
#        out_high = F.softmax(out_high, dim=1)
#        out_low = F.softmax(out_low, dim=1)
#        self.log("valid_loss", valid_loss, prog_bar=True, on_step=False, on_epoch=True)
#        self.log("valid_low_loss", low_loss, on_step=False, on_epoch=True)
#        self.log("valid_high_loss", high_loss, on_step=False, on_epoch=True)
#        self.log("valid_acc", self.valid_acc(F.softmax(out_high+out_low, dim=1), answer.squeeze(1)), prog_bar=True, on_step=False, on_epoch=True)
#        self.log("valid_low_acc", self.valid_acc(out_low, answer.squeeze(1)), on_step=False, on_epoch=True)
#        self.log("valid_high_acc", self.valid_acc(out_high, answer.squeeze(1)), on_step=False, on_epoch=True)
#        return valid_loss
#
#
#
#
#class Hopfield_3(pl.LightningModule):
#    def __init__(self, args, n_answers, ans2idx):   # Pass ans2idx from relevant dataset object
#        super().__init__()
#        self.args = args
#        # LXMERT Models
#        self.high_lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
#        self.low_lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
#        # Language/Vision LSTM
#        self.lng_lstm = nn.LSTM(768, 1024, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
#        self.vis_lstm = nn.LSTM(768, 1024, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
#        # Hopfield Nets
#        self.high_hopfield = hpf.Hopfield(input_size = 8960, hidden_size = 1024, output_size = 1024, pattern_size = 1, num_heads = 7, scaling = args.hopfield_beta_high, update_steps_max = 3, update_steps_eps = 1e-4, dropout = 0.2)
#        self.low_hopfield = hpf.Hopfield(input_size = 8960, hidden_size = 1024, output_size = 1024, pattern_size = 1, num_heads = 7, scaling = args.hopfield_beta_low, update_steps_max = 3, update_steps_eps = 1e-4, dropout = 0.2)
#        fc_intermediate = ((n_answers-1024)//2)+1024
#         # High-norm / low-norm may mean high abstract/concrete. But generalised for other norms
#        self.low_classifier_fc = nn.Sequential(
#            nn.Dropout(0.2),
#            nn.Linear(1024, fc_intermediate),
#            nn.BatchNorm1d(fc_intermediate),
#            nn.GELU(),
#            nn.Dropout(0.2),
#            nn.Linear(fc_intermediate, n_answers+1)   #GQA has 1842 unique answers, so we pass in 1841
#        )
#        self.high_classifier_fc = nn.Sequential(
#            nn.Dropout(0.2),
#            nn.Linear(1024, fc_intermediate),
#            nn.BatchNorm1d(fc_intermediate),
#            nn.GELU(),
#            nn.Dropout(0.2),
#            nn.Linear(fc_intermediate, n_answers+1)
#        )
#        if args.unfreeze == "all":
#            pass
#        elif args.unfreeze == "heads":
#            for param in self.high_lxmert.base_model.parameters():
#                param.requires_grad = False
#            for param in self.low_lxmert.base_model.parameters():
#                param.requires_grad = False
#        elif args.unfreeze == "none":
#            for param in self.high_lxmert.parameters():
#                param.requires_grad = False
#            for param in self.low_lxmert.parameters():
#                param.requires_grad = False
#        if args.loss == "default":
#            self.criterion = nn.CrossEntropyLoss(reduction='none')
#        elif args.loss == "avsc":
#            self.criterion = nn.BCEWithLogitsLoss(reduction='none')
#        else:
#            raise NotImplementedError(f"Loss {args.loss} not implement for Hopfield_3 net")
#        self.valid_acc = pl.metrics.Accuracy()
#        self.valid_low_acc = pl.metrics.Accuracy()
#        self.valid_high_acc = pl.metrics.Accuracy()
#        self.train_acc = pl.metrics.Accuracy()
#        self.train_low_acc = pl.metrics.Accuracy()
#        self.train_high_acc = pl.metrics.Accuracy()
#
#
#    def forward(self, question, bboxes, features, image):
#        # Process language
#        out_low = self.low_lxmert(question, features, bboxes)       #['language_output', 'vision_output', 'pooled_output']
#        lng_out_low, vis_out_low, x_out_low = out_low['language_output'], out_low['vision_output'], out_low['pooled_output']
#        out_high = self.high_lxmert(question, features, bboxes)     #['language_output', 'vision_output', 'pooled_output']
#        lng_out_high, vis_out_high, x_out_high = out_high['language_output'], out_high['vision_output'], out_high['pooled_output']
#        # x stands for 'cross', see naming scheme in documentation
#        # Language/Vision LSTM processing
#        _, (_, lng_out_low) = self.lng_lstm(lng_out_low)
#        _, (_, lng_out_high) = self.lng_lstm(lng_out_high)
#        _, (_, vis_out_low) = self.vis_lstm(vis_out_low)
#        _, (_, vis_out_high) = self.vis_lstm(vis_out_high)
#        lng_out_low = lng_out_low.permute(1,0,2).contiguous().view(self.args.bsz, -1)
#        lng_out_high = lng_out_high.permute(1,0,2).contiguous().view(self.args.bsz, -1)
#        vis_out_low = vis_out_low.permute(1,0,2).contiguous().view(self.args.bsz, -1)
#        vis_out_high = vis_out_high.permute(1,0,2).contiguous().view(self.args.bsz, -1)
#        # Hopfield
#        out_low = torch.cat((lng_out_low, vis_out_low, x_out_low), dim=1).unsqueeze(1)
#        out_high = torch.cat((lng_out_high, vis_out_high, x_out_high), dim=1).unsqueeze(1)
#        out_low = self.low_hopfield(out_low)
#        out_high = self.high_hopfield(out_high)
#        out_low = out_low.squeeze(1)
#        out_high = out_high.squeeze(1)
#        out_low = self.low_classifier_fc(out_low)
#        out_high = self.high_classifier_fc(out_high)
#        return out_low, out_high
#
#    def configure_optimizers(self):
#        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
#        return optimizer
#
#    def training_step(self, train_batch, batch_idx):
#        # Prepare data
#        question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens = train_batch
#        high_norms = return_norm
#        low_norms = torch.ones(len(high_norms)).to(self.device)
#        low_norms = low_norms - high_norms
#        out_low, out_high = self(question, bboxes, features, image)
#        if self.args.loss == "default":
#            low_loss = self.criterion(out_low, answer.squeeze(1))
#            high_loss = self.criterion(out_high, answer.squeeze(1))
#        elif self.args.loss == "avsc":
#            low_loss = torch.mean(self.criterion(out_low, abs_answer_tens), 1)
#            high_loss = torch.mean(self.criterion(out_high, conc_answer_tens), 1)
#        low_loss = torch.dot(low_norms, low_loss) / len(low_loss)
#        high_loss = torch.dot(high_norms, high_loss) / len(high_loss)
#        train_loss = low_loss + high_loss
#        out_high = F.softmax(out_high, dim=1)
#        out_low = F.softmax(out_low, dim=1)
#        self.log("train_loss", train_loss, prog_bar=True, on_step=False, on_epoch=True)
#        self.log("train_low_loss", low_loss, on_step=False, on_epoch=True)
#        self.log("train_high_loss", high_loss, on_step=False, on_epoch=True)
#        self.log("train_acc", self.train_acc(F.softmax(out_high+out_low, dim=1), answer.squeeze(1)), prog_bar=True, on_step=False, on_epoch=True)
#        self.log("train_low_acc", self.train_acc(out_low, answer.squeeze(1)), on_step=False, on_epoch=True)
#        self.log("train_high_acc", self.train_acc(out_high, answer.squeeze(1)), on_step=False, on_epoch=True)
#        return train_loss
#
#    def validation_step(self, val_batch, batch_idx):
#        raise NotImplementedError("Check that this works")
#        question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens = val_batch
#        high_norms = return_norm
#        low_norms = torch.ones(len(high_norms)).to(self.device)
#        low_norms = low_norms - high_norms
#        out_low, out_high = self(question, bboxes, features, image)
#        if self.args.loss == "default":
#            low_loss = self.criterion(out_low, answer.squeeze(1))
#            high_loss = self.criterion(out_high, answer.squeeze(1))
#        elif self.args.loss == "avsc":
#            low_loss = torch.mean(self.criterion(out_low, abs_answer_tens), 1)
#            high_loss = torch.mean(self.criterion(out_high, conc_answer_tens), 1)
#        low_loss = torch.dot(low_norms, low_loss) / len(low_loss)
#        high_loss = torch.dot(high_norms, high_loss) / len(high_loss)
#        valid_loss = low_loss + high_loss
#        out_high = F.softmax(out_high, dim=1)
#        out_low = F.softmax(out_low, dim=1)
#        self.log("valid_loss", valid_loss, prog_bar=True, on_step=False, on_epoch=True)
#        self.log("valid_low_loss", low_loss, on_step=False, on_epoch=True)
#        self.log("valid_high_loss", high_loss, on_step=False, on_epoch=True)
#        self.log("valid_acc", self.valid_acc(F.softmax(out_high+out_low, dim=1), answer.squeeze(1)), prog_bar=True, on_step=False, on_epoch=True)
#        self.log("valid_low_acc", self.valid_acc(out_low, answer.squeeze(1)), on_step=False, on_epoch=True)
#        self.log("valid_high_acc", self.valid_acc(out_high, answer.squeeze(1)), on_step=False, on_epoch=True)
#        return valid_loss



class Dual_LxLSTM(pl.LightningModule):
    def __init__(self, args, n_answers, ans2idx):   # Pass ans2idx from relevant dataset object
        super().__init__()
        self.args = args
        # LXMERT Models
        self.high_lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.low_lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
        # Language/Vision LSTM
        self.lng_lstm = nn.LSTM(768, 1024, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.vis_lstm = nn.LSTM(768, 1024, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        fc_intermediate = ((n_answers-8960)//2)+8960
        self.low_classifier_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(8960, fc_intermediate),
            nn.BatchNorm1d(fc_intermediate),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fc_intermediate, n_answers+1)   #GQA has 1842 unique answers, so we pass in 1841
        )
        self.high_classifier_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(8960, fc_intermediate),
            nn.BatchNorm1d(fc_intermediate),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fc_intermediate, n_answers+1)
        )
        for name, param in self.high_lxmert.named_parameters():
            param.requires_grad = True
        for name, param in self.low_lxmert.named_parameters():
            param.requires_grad = True
        if args.unfreeze == "all":
            pass
        elif args.unfreeze == "heads":
            for name, param in self.high_lxmert.named_parameters():
                if not("attention" in name):
                    param.requires_grad = False
            for name, param in self.low_lxmert.named_parameters():
                if not("attention" in name):
                    param.requires_grad = False
        elif args.unfreeze == "none":
            for name, param in self.high_lxmert.named_parameters():
                param.requires_grad = False
            for name, param in self.low_lxmert.named_parameters():
                param.requires_grad = False
        if args.loss == "default":
            self.criterion = nn.CrossEntropyLoss(reduction='none')
        elif args.loss == "avsc":
            self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        else:
            raise NotImplementedError(f"Loss {args.loss} not implement for {args.model} net")
        # Logging for metrics
        self.valid_acc = torchmetrics.Accuracy()
        self.valid_low_acc = torchmetrics.Accuracy()
        self.valid_high_acc = torchmetrics.Accuracy()
        self.train_acc = torchmetrics.Accuracy()
        self.train_low_acc = torchmetrics.Accuracy()
        self.train_high_acc = torchmetrics.Accuracy()
        self.best_acc = 0
        self.predictions = {}
        self.high_predictions = {}
        self.high_attentions = {}
        self.low_predictions = {}
        self.low_attentions = {}
        # TODO DEPRECATED# Correct answer ids to work out consistency/plausability etc..
        # self.correct_answers = []
        # self.correct_answers_low = []
        # self.correct_answers_high = []
        # Manual optimisation to allow slower training for previous layers
        self.automatic_optimization = False

        # RUBi things
        if args.rubi == "rubi":
            self.biased_bert = BertModel.from_pretrained("bert-base-uncased")
            for name, param in self.biased_bert.named_parameters():
                #if not("attention" in name):
                param.requires_grad = False
            self.biased_classifier_fc = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(768, fc_intermediate),#nn.Linear(4096, fc_intermediate),
                nn.BatchNorm1d(fc_intermediate),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(fc_intermediate, n_answers+1)
            )
            # Overwrite criterion
            if args.loss == "default":
                self.criterion = myutils.RUBi_Criterion(loss_type="CrossEntropyLoss")
            elif args.loss == "avsc":
                self.criterion = myutils.RUBi_Criterion(loss_type="BCEWithLogitsLoss")
            else:
                raise NotImplementedError(f"Loss {args.loss} not implement for {args.model} net")


    def forward(self, question, bboxes, features, image):
        # Process language
        out_low = self.low_lxmert(question, features, bboxes, output_attentions=True)
        # NOTE According to GitHub issue: https://github.com/airsplay/lxmert/issues/26, bounding boxes are of form (x0,y0,x1,y1) for lxmert
        #['language_output', 'vision_output', 'pooled_output', 'language_attentions', 'vision_attentions', 'cross_attentions']
        lng_out_low, vis_out_low, x_out_low = out_low['language_output'], out_low['vision_output'], out_low['pooled_output']
        vis_attns_low = torch.stack(out_low['vision_attentions']).mean(dim=0).mean(dim=1)
        out_high = self.high_lxmert(question, features, bboxes, output_attentions=True)
        #['language_output', 'vision_output', 'pooled_output', 'language_attentions', 'vision_attentions', 'cross_attentions']
        lng_out_high, vis_out_high, x_out_high = out_high['language_output'], out_high['vision_output'], out_high['pooled_output']
        vis_attns_high = torch.stack(out_high['vision_attentions']).mean(dim=0).mean(dim=1)
        if self.args.rubi == "rubi":
            ## Language only
            #TODO deprecated? lng_out_biased = self.biased_bert(question)[0]
            #_, (_, lng_out_biased) = self.biased_lng_lstm(lng_out_biased)
            #lng_out_biased = lng_out_biased.permute(1,0,2).contiguous().view(self.args.bsz, -1)
            lng_out_biased = self.biased_bert(question)[1].squeeze(1)
            out_biased = self.biased_classifier_fc(lng_out_biased)
        else:
            out_biased = None
        # x stands for 'cross', see naming scheme in documentation
        # Language/Vision LSTM processing
        _, (_, lng_out_low) = self.lng_lstm(lng_out_low)
        _, (_, lng_out_high) = self.lng_lstm(lng_out_high)
        _, (_, vis_out_low) = self.vis_lstm(vis_out_low)
        _, (_, vis_out_high) = self.vis_lstm(vis_out_high)
        bsz = lng_out_low.shape[1]
        lng_out_low = lng_out_low.permute(1,0,2).contiguous().view(bsz, -1)
        lng_out_high = lng_out_high.permute(1,0,2).contiguous().view(bsz, -1)
        vis_out_low = vis_out_low.permute(1,0,2).contiguous().view(bsz, -1)
        vis_out_high = vis_out_high.permute(1,0,2).contiguous().view(bsz, -1)
        out_low = torch.cat((lng_out_low, vis_out_low, x_out_low), dim=1)
        out_high = torch.cat((lng_out_high, vis_out_high, x_out_high), dim=1)
        out_low = self.low_classifier_fc(out_low)
        out_high = self.high_classifier_fc(out_high)
        return out_low, out_high, out_biased, vis_attns_low, vis_attns_high

    def configure_optimizers(self):
        other_optimizer = torch.optim.Adam(nn.ParameterList([p for n,p in self.named_parameters() if "lxmert" not in n]), lr=self.args.lr)
        lxmert_optimizer = torch.optim.Adam(nn.ParameterList([p for n,p in self.named_parameters() if "lxmert" in n]), lr=self.args.lr/5)
        return other_optimizer, lxmert_optimizer

    def training_step(self, train_batch, batch_idx, optimizer_idx):
        # Prepare data
        other_optimizer, lxmert_optimizer = self.optimizers()
        other_optimizer.zero_grad()
        lxmert_optimizer.zero_grad()
        question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens, _, q_id_ret, _ = train_batch
        out_low, out_high, out_biased, vis_attns_low, vis_attns_high = self(question, bboxes, features, image) # out_biased is from potential RUBi outputs
        if self.args.dual_loss_style == "linear":
            high_norms = return_norm
            low_norms = torch.ones(len(high_norms)).to(self.device)
            low_norms = low_norms - high_norms
        elif self.args.dual_loss_style == "quadr":
            high_norms = (return_norm)**2
            low_norms = (return_norm-1)**2
        elif self.args.dual_loss_style == "cubic":
            high_norms = (return_norm)**3
            low_norms = -1*((return_norm-1)**3)
        elif self.args.dual_loss_style == "4th":
            high_norms = (return_norm)**4
            low_norms = (return_norm-1)**4
        else:
            raise NotImplementedError(f"`{self.args.dual_loss_style}` not implemented")
        if self.args.loss == "default":
            if self.args.rubi == "rubi":
                #['combined_loss']
                #['main_loss']
                #['biased_loss']
                low_loss = self.criterion(out_low, out_biased, answer.squeeze(1), biased_loss_weighting=1.0)
                low_biased_loss = low_loss['biased_loss']
                low_loss = low_loss['combined_loss']+low_biased_loss
                high_loss = self.criterion(out_high, out_biased, answer.squeeze(1), biased_loss_weighting=1.0)
                high_biased_loss = high_loss['biased_loss']
                high_loss = high_loss['combined_loss']+high_biased_loss
            else:
                low_loss = self.criterion(out_low, answer.squeeze(1))
                high_loss = self.criterion(out_high, answer.squeeze(1))
        elif self.args.loss == "avsc":
            if self.args.rubi == "rubi":
                low_loss = self.criterion(out_high, out_biased, abs_answer_tens, biased_loss_weighting=1.0)
                low_biased_loss = low_loss['biased_loss']
                low_loss = low_loss['combined_loss']+low_biased_loss
                low_loss = torch.mean(low_loss, 1)
                high_loss = self.criterion(out_high, out_biased, conc_answer_tens, biased_loss_weighting=1.0)
                high_biased_loss = high_loss['biased_loss']
                high_loss = high_loss['combined_loss']+high_biased_loss
                high_loss = torch.mean(high_loss, 1)
            else:
                low_loss = torch.mean(self.criterion(out_low, abs_answer_tens), 1)
                high_loss = torch.mean(self.criterion(out_high, conc_answer_tens), 1)
        low_loss = torch.dot(low_norms, low_loss) / len(low_loss)
        high_loss = torch.dot(high_norms, high_loss) / len(high_loss)
        train_loss = low_loss + high_loss
        out_high = F.softmax(out_high, dim=1)
        out_low = F.softmax(out_low, dim=1)
        self.log("train_loss", train_loss, prog_bar=False, on_step=True)#, on_epoch=True)
        self.log("train_low_loss", low_loss, on_step=True)#, on_epoch=True)
        self.log("train_high_loss", high_loss, on_step=True)#False, on_epoch=True)
        self.log("train_acc", self.train_acc(F.softmax(out_high+out_low, dim=1), answer.squeeze(1)), prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_low_acc", self.train_acc(out_low, answer.squeeze(1)), on_step=False, on_epoch=True)
        self.log("train_high_acc", self.train_acc(out_high, answer.squeeze(1)), on_step=False, on_epoch=True)
        if self.args.rubi == "rubi":
            self.log("train_low_biased_loss", low_biased_loss, on_step=True)#False, on_epoch=True)
            self.log("train_high_biased_loss", high_biased_loss, on_step=True)#False, on_epoch=True)
        self.manual_backward(train_loss)
        other_optimizer.step()
        lxmert_optimizer.step()
        #return train_loss

    def validation_step(self, val_batch, batch_idx):
        question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens, _, q_id_ret, img_dims = val_batch
        out_low, out_high, out_biased, vis_attns_low, vis_attns_high = self(question, bboxes, features, image)
        if self.args.dual_loss_style == "linear":
            high_norms = return_norm
            low_norms = torch.ones(len(high_norms)).to(self.device)
            low_norms = low_norms - high_norms
        elif self.args.dual_loss_style == "quadr":
            high_norms = (return_norm)**2
            low_norms = (return_norm-1)**2
        elif self.args.dual_loss_style == "cubic":
            high_norms = (return_norm)**3
            low_norms = -1*((return_norm-1)**3)
        elif self.args.dual_loss_style == "4th":
            high_norms = (return_norm)**4
            low_norms = (return_norm-1)**4
        else:
            raise NotImplementedError(f"`{self.args.dual_loss_style}` not implemented")
        if self.args.loss == "default":
            if self.args.rubi == "rubi":
                #['combined_loss']
                #['main_loss']
                #['biased_loss']
                low_loss = self.criterion(out_low, out_biased, answer.squeeze(1), biased_loss_weighting=1.0)
                low_biased_loss = low_loss['biased_loss']
                low_loss = low_loss['combined_loss']+low_biased_loss
                high_loss = self.criterion(out_high, out_biased, answer.squeeze(1), biased_loss_weighting=1.0)
                high_biased_loss = high_loss['biased_loss']
                high_loss = high_loss['combined_loss']+high_biased_loss
            else:
                low_loss = self.criterion(out_low, answer.squeeze(1))
                high_loss = self.criterion(out_high, answer.squeeze(1))
        elif self.args.loss == "avsc":
            if self.args.rubi == "rubi":
                low_loss = self.criterion(out_high, out_biased, abs_answer_tens, biased_loss_weighting=1.0)
                low_biased_loss = low_loss['biased_loss']
                low_loss = low_loss['combined_loss']+low_biased_loss
                low_loss = torch.mean(low_loss, 1)
                high_loss = self.criterion(out_high, out_biased, conc_answer_tens, biased_loss_weighting=1.0)
                high_biased_loss = high_loss['biased_loss']
                high_loss = high_loss['combined_loss']+high_biased_loss
                high_loss = torch.mean(high_loss, 1)
            else:
                low_loss = torch.mean(self.criterion(out_low, abs_answer_tens), 1)
                high_loss = torch.mean(self.criterion(out_high, conc_answer_tens), 1)
        low_loss = torch.dot(low_norms, low_loss) / len(low_loss)
        high_loss = torch.dot(high_norms, high_loss) / len(high_loss)
        valid_loss = low_loss + high_loss
        out_high = F.softmax(out_high, dim=1)
        out_low = F.softmax(out_low, dim=1)
        combined = F.softmax(out_high+out_low, dim=1)
        # TODO DEPRECATED?? #### Save image_ids of correct 
        ### Total
        #corrects = F.softmax(out_high+out_low, dim=1).argmax(dim=1) == answer.squeeze(1)
        #ids = (corrects*q_id_ret[:,0]).cpu()
        #id_lens = (corrects*q_id_ret[:,1]).cpu()
        #corrects = [f"{ids[i]}".zfill(id_lens[i]) for i in range(len(id_lens)) if id_lens[i] != 0]
        #self.correct_answers += corrects
        ### High corrects
        #corrects = out_high.argmax(dim=1) == answer.squeeze(1)
        #ids = (corrects*q_id_ret[:,0]).cpu()
        #id_lens = (corrects*q_id_ret[:,1]).cpu()
        #corrects = [f"{ids[i]}".zfill(id_lens[i]) for i in range(len(id_lens)) if id_lens[i] != 0]
        #self.correct_answers_high += corrects
        ### Abstract corrects
        #corrects = out_low.argmax(dim=1) == answer.squeeze(1)
        #ids = (corrects*q_id_ret[:,0]).cpu()
        #id_lens = (corrects*q_id_ret[:,1]).cpu()
        #corrects = [f"{ids[i]}".zfill(id_lens[i]) for i in range(len(id_lens)) if id_lens[i] != 0]
        #self.correct_answers_low += corrects
        self.log("valid_loss", valid_loss, prog_bar=False, on_step=True)#False, on_epoch=True)
        self.log("valid_low_loss", low_loss, on_step=True)#False, on_epoch=True)
        self.log("valid_high_loss", high_loss, on_step=True)#False, on_epoch=True)
        self.log("valid_acc", self.valid_acc(combined, answer.squeeze(1)), prog_bar=False, on_step=False, on_epoch=True)
        self.log("valid_low_acc", self.valid_acc(out_low, answer.squeeze(1)), on_step=False, on_epoch=True)
        self.log("valid_high_acc", self.valid_acc(out_high, answer.squeeze(1)), on_step=False, on_epoch=True)
        combined = combined.argmax(dim=1)
        out_high = out_high.argmax(dim=1)
        out_low = out_low.argmax(dim=1)
        # Move the rescaling to the forward pass
        vis_attns_high = vis_attns_high.mean(dim=1)
        vis_attns_low = vis_attns_low.mean(dim=1)
        #vis_attns_high.cpu()
        #vis_attns_low.cpu()
        #bboxes.cpu()
        for i in range(len(q_id_ret)):
            q_idx = f"{q_id_ret[i][0]}".zfill(q_id_ret[i][1])
            self.predictions[q_idx] = self.val_dataloader.dataloader.dataset.idx2ans[int(combined[i])]
            self.high_predictions[q_idx] = self.val_dataloader.dataloader.dataset.idx2ans[int(out_high[i])]
            self.high_attentions[q_idx] = [((float(bboxes[i][j][0]), float(bboxes[i][j][1]), float(bboxes[i][j][2]), float(bboxes[i][j][3])), float(vis_attns_high[i][j])) for j in range(len(vis_attns_high[i]))]
            self.low_predictions[q_idx] = self.val_dataloader.dataloader.dataset.idx2ans[int(out_low[i])]
            self.low_attentions[q_idx] = [((float(bboxes[i][j][0]), float(bboxes[i][j][1]), float(bboxes[i][j][2]), float(bboxes[i][j][3])), float(vis_attns_low[i][j])) for j in range(len(vis_attns_low[i]))]
        if self.args.rubi == "rubi":
            self.log("valid_low_biased_loss", low_biased_loss, on_step=True)#False, on_epoch=True)
            self.log("valid_high_biased_loss", high_biased_loss, on_step=True)#False, on_epoch=True)
        return valid_loss

    def validation_epoch_end(self, val_step_outputs):
        current_acc = (self.valid_acc.correct/self.valid_acc.total) if self.valid_acc.total != 0 else 0
        if current_acc >= self.best_acc:
            if not self.trainer.running_sanity_check:
                # Save predictions and attentions to .json file to later be handled
                metrics_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "checkpoints", self.args.jobname)
                if os.path.exists(metrics_dir):
                    shutil.rmtree(metrics_dir)
                os.makedirs(metrics_dir)
                myutils.save_json(self.predictions, os.path.join(metrics_dir, "predictions.json"))
                myutils.save_json(self.high_predictions, os.path.join(metrics_dir, "high_predictions.json"))
                myutils.save_json(self.low_predictions, os.path.join(metrics_dir, "low_predictions.json"))
                myutils.save_json(self.high_attentions, os.path.join(metrics_dir, "high_attentions.json"))
                myutils.save_json(self.low_attentions, os.path.join(metrics_dir, "low_attentions.json"))
                # Plot 'predictions.json' without attention
                if self.args.dataset == "GQA":
                    val_questions = "val_balanced_questions.json"
                elif self.args.dataset == "GQA-ABSMIXED":
                    val_questions = "absMixed_val_questions.json"
                os.system(f"python eval.py --tier 'val' --checkpoint_path 'checkpoints/{args.jobname}' --score_file_name 'scores.txt' --scenes 'val_sceneGraphs.json' --questions '{val_questions}' --choices 'val_choices.json' --predictions 'predictions.json' --consistency")
                with open(os.path.join(metrics_dir, "scores.txt")) as f:
                    scores = f.read().replace('\n', '<br />')
                    scores = "<p>"+scores+"</p>"
                    self.log("scores", wandb.Html(scores))
                # Plot 'high_predictions.json' with high attentions
                os.system(f"python eval.py --tier 'val' --checkpoint_path 'checkpoints/{args.jobname}' --score_file_name 'high_scores.txt' --scenes 'val_sceneGraphs.json' --questions '{val_questions}' --choices 'val_choices.json' --predictions 'high_predictions.json' --attentions 'high_attentions.json' --consistency --grounding --objectFeatures")
                with open(os.path.join(metrics_dir, "high_scores.txt")) as f:
                    scores = f.read().replace('\n', '<br />')
                    scores = "<p>"+scores+"</p>"
                    self.log("high_scores", wandb.Html(scores))
                # Plot 'low_predictions.json' with low attentions
                os.system(f"python eval.py --tier 'val' --checkpoint_path 'checkpoints/{args.jobname}' --score_file_name 'low_scores.txt' --scenes 'val_sceneGraphs.json' --questions '{val_questions}' --choices 'val_choices.json' --predictions 'low_predictions.json' --attentions 'low_attentions.json' --consistency --grounding --objectFeatures")
                with open(os.path.join(metrics_dir, "low_scores.txt")) as f:
                    scores = f.read().replace('\n', '<br />')
                    scores = "<p>"+scores+"</p>"
                    self.log("low_scores", wandb.Html(scores))
        # TODO DEPRECATED?
        #correct = 0
        #total = 0
        #val_qids = self.__dict__['val_dataloader'].dataloader.dataset.q_as.keys()
        #for correct_ans in self.correct_answers:
        #    entailed_qs = [entailed_q for entailed_q in self.__dict__['val_dataloader'].dataloader.dataset.q_as[correct_ans]['entailed'] if entailed_q in val_qids ]
        #    total += len(entailed_qs)
        #    correct += sum([1 for entailed_q in entailed_qs if entailed_q in self.correct_answers])
        #consistency = correct/float(total) if total != 0 else 0
        #self.log("valid_consistency", consistency)
        ### High
        #correct = 0
        #total = 0
        #val_qids = self.__dict__['val_dataloader'].dataloader.dataset.q_as.keys()
        #for correct_ans in self.correct_answers_high:
        #    entailed_qs = [entailed_q for entailed_q in self.__dict__['val_dataloader'].dataloader.dataset.q_as[correct_ans]['entailed'] if entailed_q in val_qids ]
        #    total += len(entailed_qs)
        #    correct += sum([1 for entailed_q in entailed_qs if entailed_q in self.correct_answers_high])
        #consistency = correct/float(total) if total != 0 else 0
        #self.log("valid_consistency_high", consistency)
        ### Low
        #correct = 0
        #total = 0
        #val_qids = self.__dict__['val_dataloader'].dataloader.dataset.q_as.keys()
        #for correct_ans in self.correct_answers_low:
        #    entailed_qs = [entailed_q for entailed_q in self.__dict__['val_dataloader'].dataloader.dataset.q_as[correct_ans]['entailed'] if entailed_q in val_qids ]
        #    total += len(entailed_qs)
        #    correct += sum([1 for entailed_q in entailed_qs if entailed_q in self.correct_answers_low])
        #consistency = correct/float(total) if total != 0 else 0
        #self.log("valid_consistency_low", consistency)

class Dummy_Lxmert_Conf():
    # Just to pass hidden_size to LxmertVisualAnswerHead
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

#class Dual_LxForQA(pl.LightningModule):
#    def __init__(self, args, n_answers, ans2idx):   # Pass ans2idx from relevant dataset object
#        super().__init__()
#        self.args = args
#        # LXMERT Models
#        dummy_conf = Dummy_Lxmert_Conf(hidden_size=768)
#        if args.dataset == "GQA":
#            high_ans_head = LxmertVisualAnswerHead(config=dummy_conf, num_labels=len(ans2idx))
#            low_ans_head = LxmertVisualAnswerHead(config=dummy_conf, num_labels=len(ans2idx))
#        elif args.dataset in ["VQACP","VQACP2"]:
#            high_ans_head = LxmertVisualAnswerHead(config=dummy_conf, num_labels=len(ans2idx)+1)
#            low_ans_head = LxmertVisualAnswerHead(config=dummy_conf, num_labels=len(ans2idx)+1)
#        self.high_lxmert = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-base-uncased")
#        self.low_lxmert = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-base-uncased")
#        self.high_lxmert.answer_head = high_ans_head
#        self.low_lxmert.answer_head = low_ans_head
#        for name, param in self.high_lxmert.named_parameters():
#            param.requires_grad = True
#        for name, param in self.low_lxmert.named_parameters():
#            param.requires_grad = True
#        if args.unfreeze == "all":
#            pass
#        elif args.unfreeze == "heads":
#            for name, param in self.high_lxmert.named_parameters():
#                if not("attention" in name):
#                    param.requires_grad = False
#            for name, param in self.low_lxmert.named_parameters():
#                if not("attention" in name):
#                    param.requires_grad = False
#        elif args.unfreeze == "qa_head":
#            for name, param in self.high_lxmert.named_parameters():
#                if not("answer_head" in name):
#                    param.requires_grad = False
#            for name, param in self.low_lxmert.named_parameters():
#                if not("answer_head" in name):
#                    param.requires_grad = False
#        elif args.unfreeze == "none":
#            for name, param in self.high_lxmert.named_parameters():
#                param.requires_grad = False
#            for name, param in self.low_lxmert.named_parameters():
#                param.requires_grad = False
#        if args.loss == "default":
#            self.criterion = nn.CrossEntropyLoss(reduction='none')
#        elif args.loss == "avsc":
#            self.criterion = nn.BCEWithLogitsLoss(reduction='none')
#        else:
#            raise NotImplementedError(f"Loss {args.loss} not implement for Hopfield_3 net")
#        self.valid_acc = pl.metrics.Accuracy()
#        self.valid_low_acc = pl.metrics.Accuracy()
#        self.valid_high_acc = pl.metrics.Accuracy()
#        self.train_acc = pl.metrics.Accuracy()
#        self.train_low_acc = pl.metrics.Accuracy()
#        self.train_high_acc = pl.metrics.Accuracy()
#                
#
#    def forward(self, question, bboxes, features, image):
#        # Process language
#        out_low = self.low_lxmert(question, features, bboxes)['question_answering_score']
#        out_high = self.high_lxmert(question, features, bboxes)['question_answering_score']
#        return out_low, out_high
#
#    def configure_optimizers(self):
#        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
#        return optimizer
#
#    def training_step(self, train_batch, batch_idx):
#        # Prepare data
#        question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens, _ = train_batch
#        high_norms = return_norm
#        low_norms = torch.ones(len(high_norms)).to(self.device)
#        low_norms = low_norms - high_norms
#        out_low, out_high = self(question, bboxes, features, image)
#        if self.args.loss == "default":
#            low_loss = self.criterion(out_low, answer.squeeze(1))
#            high_loss = self.criterion(out_high, answer.squeeze(1))
#        elif self.args.loss == "avsc":
#            low_loss = torch.mean(self.criterion(out_low, abs_answer_tens), 1)
#            high_loss = torch.mean(self.criterion(out_high, conc_answer_tens), 1)
#        low_loss = torch.dot(low_norms, low_loss) / len(low_loss)
#        high_loss = torch.dot(high_norms, high_loss) / len(high_loss)
#        train_loss = low_loss + high_loss
#        out_high = F.softmax(out_high, dim=1)
#        out_low = F.softmax(out_low, dim=1)
#        self.log("train_loss", train_loss, prog_bar=True, on_step=False, on_epoch=True)
#        self.log("train_low_loss", low_loss, on_step=False, on_epoch=True)
#        self.log("train_high_loss", high_loss, on_step=False, on_epoch=True)
#        self.log("train_acc", self.train_acc(F.softmax(out_high+out_low, dim=1), answer.squeeze(1)), prog_bar=True, on_step=False, on_epoch=True)
#        self.log("train_low_acc", self.train_acc(out_low, answer.squeeze(1)), on_step=False, on_epoch=True)
#        self.log("train_high_acc", self.train_acc(out_high, answer.squeeze(1)), on_step=False, on_epoch=True)
#        return train_loss
#
#    def validation_step(self, val_batch, batch_idx):
#        question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens, _ = val_batch
#        high_norms = return_norm
#        low_norms = torch.ones(len(high_norms)).to(self.device)
#        low_norms = low_norms - high_norms
#        out_low, out_high = self(question, bboxes, features, image)
#        if self.args.loss == "default":
#            low_loss = self.criterion(out_low, answer.squeeze(1))
#            high_loss = self.criterion(out_high, answer.squeeze(1))
#        elif self.args.loss == "avsc":
#            low_loss = torch.mean(self.criterion(out_low, abs_answer_tens), 1)
#            high_loss = torch.mean(self.criterion(out_high, conc_answer_tens), 1)
#        low_loss = torch.dot(low_norms, low_loss) / len(low_loss)
#        high_loss = torch.dot(high_norms, high_loss) / len(high_loss)
#        valid_loss = low_loss + high_loss
#        out_high = F.softmax(out_high, dim=1)
#        out_low = F.softmax(out_low, dim=1)
#        self.log("valid_loss", valid_loss, prog_bar=True, on_step=False, on_epoch=True)
#        self.log("valid_low_loss", low_loss, on_step=False, on_epoch=True)
#        self.log("valid_high_loss", high_loss, on_step=False, on_epoch=True)
#        self.log("valid_acc", self.valid_acc(F.softmax(out_high+out_low, dim=1), answer.squeeze(1)), prog_bar=True, on_step=False, on_epoch=True)
#        self.log("valid_low_acc", self.valid_acc(out_low, answer.squeeze(1)), on_step=False, on_epoch=True)
#        self.log("valid_high_acc", self.valid_acc(out_high, answer.squeeze(1)), on_step=False, on_epoch=True)
#        return valid_loss




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument_group("Running Arguments")
    parser.add_argument("--jobname", type=str, required=True, help="Jobname")
    parser.add_argument("--dataset", type=str.upper, required=True, choices=["VQACP","VQACP2","GQA","GQA-ABSMIXED"], help="Choose VQA dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--bsz", type=int, default=16, help="Training batch size")
    parser.add_argument("--val_bsz", type=int, default=100, help="Validation batch size")
    parser.add_argument("--device", type=int, default=-1, help="Which device to run things on. (-1 = CPU)")
    parser.add_argument("--wandb", action="store_true", help="Plot wandb results online")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of pytroch workers. More should increase disk reads, but will increase RAM usage. 0 = main process")
    parser.add_argument("--norm", type=str, default="conc-m", help="The norm to consider in relevant models. (conc-m == mean concreteness)")
    parser.add_argument_group("Model Arguments")
    parser.add_argument("--model", type=str, default="basic", choices=["basic", "induction", "lx-lstm", "bert-lstm", "hpf-0", "hpf-1", "hpf-2", "hpf-3", "dual-lx-lstm", "dual-lxforqa"], help="Which model")
    parser.add_argument("--unfreeze", type=str, required=True, choices=["heads","all","none","qa_head"], help="What parts of LXMERT to unfreeze")
    parser.add_argument_group("VQA-CP arguments")
    parser.add_argument("--hopfield_beta_high", type=float, default=0.7, help="When running a high-low norm network, this is the beta scaling for the high norm hopfield net")
    parser.add_argument("--hopfield_beta_low", type=float, default=0.3, help="When running a high-low norm network, this is the beta scaling for the low norm hopfield net")
    parser.add_argument("--loss", type=str, default="default", choices=["default","avsc"], help="Whether or not to use a special loss")
    parser.add_argument("--rubi", type=str, default=None, choices=["none", "rubi"], help="Using the Reducing Unimodal Bias")
    parser.add_argument("--dual_loss_style", type=str, default="linear", choices=["linear", "quadr", "cubic", "4th"], help="For dual models, e.g: linear=(k/1-k), quadr=**2, cubic=**3 etc...")

    parser.add_argument_group("Dataset arguments")
    parser.add_argument("--norm_gt", default="answer", choices=["answer", "nsubj", "qtype", "qtype-full"], help="Where to derive the norm information of the question. 'answer'=consider the concreteness of the answer, 'nsubj'=use the concreteness of the subject of the input question")
    #### VQA-CP must have one of these 2 set to non-default values
    parser.add_argument("--topk", type=int, default=-1, help="Keep the k-top scoring answers. -1 implies ignore")
    parser.add_argument("--min_ans_occ", type=int, default=-1, help="The minimum occurence threshold for keeping an answers. -1 implies ignore")

    """
    --loss:
        default: Regular softmax loss across answer
        avsc:    Add to the cross entropy other answers scaled by their occurence as word-pair norms in the actual answers 
    """
    ####
    args = parser.parse_args()
    myutils.print_args(args)

    # Prepare dataloaders
    vqa_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/vqa")

    # Checks on if you can run
    if args.dataset in ["VQACP", "VQACP2"]:
        assert (args.topk != -1) or (args.min_ans_occ != -1), f"For VQA-CP v1/2, you must set one of topk or min_ans_occ to not default. This decides which scheme to follow to keep which answers"
        assert not((args.topk != -1) and (args.min_ans_occ != -1)), f"You must leave one of topk, or min_ans_occ at default value"
    #TODO: Automate/instruct how to use the multimodal package to neatly download files
    if (args.dataset in ["VQACP", "VQACP2"]) and (not os.path.exists(os.path.join(vqa_path, "datasets"))):
        dset_utils.download_vqa()

    # Set the correct flags for dataset processing based on which model
    # TODO Consider a more elegant way to handle these flags
    assert args.model in ["basic", "induction", "lx-lstm", "bert-lstm", "hpf-0", "hpf-1", "hpf-2", "hpf-3", "dual-lx-lstm", "dual-lxforqa"], f"Make sure to account the feature flags for any new model: {args.model} needs considering"
    objects_flag = True 
    images_flag = False
    resnet_flag = True if args.model in ["hpf-2"] else False
    return_norm = True if args.model in ["induction","hpf-0","hpf-1","hpf-2","hpf-3","dual-lx-lstm","dual-lxforqa"] else False
    return_avsc = True if args.loss in ["avsc"] else False

    if args.dataset == "VQACP":
        train_dset = VQA(args, version="cp-v1", split="train", objects=objects_flag, images=images_flag, resnet=resnet_flag, return_norm=return_norm, return_avsc=return_avsc)
        valid_dset = VQA(args, version="cp-v1", split="test", objects=objects_flag, images=images_flag, resnet=resnet_flag, return_norm=return_norm, return_avsc=return_avsc)
        # TODO Remove these?
        #train_loader = DataLoader(train_dset, batch_size=args.bsz, num_workers=args.num_workers, drop_last=True)
        #valid_loader = DataLoader(valid_dset, batch_size=args.bsz, num_workers=args.num_workers, drop_last=True)
    elif args.dataset == "VQACP2":
        train_dset = VQA(args, version="cp-v2", split="train", objects=objects_flag, images=images_flag, resnet=resnet_flag, return_norm=return_norm, return_avsc=return_avsc)
        valid_dset = VQA(args, version="cp-v2", split="test", objects=objects_flag, images=images_flag, resnet=resnet_flag, return_norm=return_norm, return_avsc=return_avsc)
        #train_loader = DataLoader(train_dset, batch_size=args.bsz, num_workers=args.num_workers, drop_last=True)
        #valid_loader = DataLoader(valid_dset, batch_size=args.bsz, num_workers=args.num_workers, drop_last=True)
    elif args.dataset == "GQA":
        # TODO Instructions for downloading GQA
        train_dset = GQA(args, split="train", objects=objects_flag, images=images_flag, resnet=resnet_flag, return_norm=return_norm, return_avsc=return_avsc)
        valid_dset = GQA(args, split="valid", objects=objects_flag, images=images_flag, resnet=resnet_flag, return_norm=return_norm, return_avsc=return_avsc)
    elif args.dataset == "GQA-ABSMIXED":
        train_dset = GQA(args, split="train-absMixed", objects=objects_flag, images=images_flag, resnet=resnet_flag, return_norm=return_norm, return_avsc=return_avsc)
        valid_dset = GQA(args, split="valid-absMixed", objects=objects_flag, images=images_flag, resnet=resnet_flag, return_norm=return_norm, return_avsc=return_avsc)
    else:
        raise NotImplementedError(f"{args.dataset} not recognised.")

    if args.model in ["hpf-2"]:
        train_loader = DataLoader(train_dset, batch_size=args.bsz, num_workers=args.num_workers, drop_last=True, collate_fn=pad_question_collate)
        valid_loader = DataLoader(valid_dset, batch_size=args.val_bsz, num_workers=args.num_workers, drop_last=True, collate_fn=pad_question_collate)
    else:
        train_loader = DataLoader(train_dset, batch_size=args.bsz, num_workers=args.num_workers, collate_fn=pad_question_collate)
        valid_loader = DataLoader(valid_dset, batch_size=args.val_bsz, num_workers=args.num_workers, collate_fn=pad_question_collate)
    
    # Prepare model & pytorch_lightning system
    wandb_logger = pl.loggers.WandbLogger(project="a_vs_c", name=args.jobname, offline=not args.wandb)#, resume="allow")
    if args.dataset in ["VQACP","VQACP2"]:
        n_answers = len(train_dset.ans2idx)
    elif args.dataset == "GQA":
        n_answers = len(train_dset.ans2idx)-1   # There are 1842 answers, we pass in 1841 because +1 will be added in model definition (for VQA-CP)
    elif args.dataset == "GQA-ABSMIXED":
        n_answers = len(train_dset.ans2idx)-1   # 1712
    else:
        raise NotImplementedError(f"{args.dataset} not implemented yet")

    ##################################
    ##################################
    ##################################
    ## CONDITIONS FOR RUNNING COMBINATIONS OF PARAMETERS (some things will not be implemented together, hopefully these checks will catch this)
    ##################################
    # TODO ERRONEOUS UNFREEZING MUST BE ADJUSTED
    if args.model not in ["dual-lx-lstm", "dual-lxforqa", "lx-lstm"]:
        raise NotImplementedError(f"So far only dual-lx-lstm model has had the erroneous unfreezing adjusted. FIX THIS")
    # TODO NOT ALL MODELS HAVE BEEN IMPLEMENTED WITH RUBi
    if (args.rubi is not None) and (args.model not in ["dual-lx-lstm"]):
        raise NotImplementedError(f"Model {args.model} has not been updated to accomodate RUBi")
    # TODO NOT ALL METRICS HAVE BEEN UPDATED TO USE TORCHMETRICS 
    if args.model not in ["dual-lx-lstm"]:
        raise NotImplementedError(f"Model {args.model} does not have metrics updated to torchmetrics with ")
    if args.model not in ["dual-lx-lstm"]:
        raise NotImplementedError(f"Model {args.model} has not been upgraded to handle the question_id returning")
    # TODO Metrics plotting isnt working
    if args.rubi != "none":
        raise NotImplementedError("RUBI metrics plotting isnt currently working")
    ##################################
    ##################################
    ##################################

    if args.model == "basic":
        pl_system = Basic(args, n_answers)
    elif args.model == "induction":
        pl_system = Induction(args, n_answers, train_dset.ans2idx)
    elif args.model == "lx-lstm":
        pl_system = LxLSTM(args, n_answers, train_dset.ans2idx)
    elif args.model == "bert-lstm":
        pl_system = BERTLSTM(args, n_answers)
    elif args.model == "hpf-0":
        pl_system = Hopfield_0(args, n_answers, train_dset.ans2idx)     # Pass the ans2idx to get answer norms  
    elif args.model == "hpf-1":
        pl_system = Hopfield_1(args, n_answers, train_dset.ans2idx)     # Pass the ans2idx to get answer norms  
    elif args.model == "hpf-2":
        pl_system = Hopfield_2(args, n_answers, train_dset.ans2idx)     # Pass the ans2idx to get answer norms 
    elif args.model == "hpf-3":
        pl_system = Hopfield_3(args, n_answers, train_dset.ans2idx)     # Pass the ans2idx to get answer norms 
    elif args.model == "dual-lx-lstm":
        pl_system = Dual_LxLSTM(args, n_answers, train_dset.ans2idx)    # Pass the ans2idx to get answer norms 
    elif args.model == "dual-lxforqa":
        pl_system = Dual_LxForQA(args, n_answers, train_dset.ans2idx)    # Pass the ans2idx to get answer norms 
    else:
        raise NotImplementedError("Model: {args.model} not implemented")
    if args.device == -1:
        gpus = None
    else:
        gpus = [args.device]    # TODO Implement multigpu support

    # Checkpointing and running
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='valid_acc',
        dirpath=os.path.join(os.path.dirname(os.path.realpath(__file__)), "checkpoints"),
        filename=f"{args.jobname}"+'-{epoch:02d}-{valid_acc:.2f}',
        save_top_k=1,
        mode='max',
    )
    trainer = pl.Trainer(callbacks=[checkpoint_callback], logger=wandb_logger, gpus=gpus)
    trainer.fit(pl_system, train_loader, valid_loader)
