# Standard imports
import os, sys
import random
import h5py
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from tqdm import tqdm


# Complex imports
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import LxmertTokenizer
import spacy
from multimodal.text import BasicTokenizer

# Local imports
import misc.myutils as myutils
import misc.dset_utils as dset_utils
from misc.multimodal_pip_vqa_utils import process_annotations
from misc.word_norms import word_is_cOrI



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
    #breakpoint()
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
        #if args.dataset in ["VQA","VQA2","VQACP","VQACP2"]:
        #    BCE_assoc_tensor.append(0)
        #    BCE_ctgrcl_tensor.append(0)
        idx2BCE_assoc_tensor[idx] = torch.Tensor(BCE_assoc_tensor)
        idx2BCE_ctgrcl_tensor[idx] = torch.Tensor(BCE_ctgrcl_tensor)
    #TODO DEPRECATED # Final unknown token if needed
    #if args.dataset in ["VQA","VQA2","VQACP","VQACP2"]:
    #    idx2BCE_assoc_tensor[len(answers)] = torch.Tensor([0]*len(answers)+[1])
    #    idx2BCE_ctgrcl_tensor[len(answers)] = torch.Tensor([0]*len(answers)+[1])
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
    if args.dataset in ["VQA","VQA2","VQACP","VQACP2"]:
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
        self.norm_dict = myutils.load_norms_pickle( os.path.join(os.path.dirname(__file__),"misc/all_norms.pickle")) 

        # Answer2Idx
        if version == "cp-v1":
            data_root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/vqa/datasets/vqacp")
        elif version == "cp-v2":
            data_root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/vqa/datasets/vqacp2")
        elif version == "v1":
            data_root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/vqa/datasets/vqa")
        elif version == "v2":
            data_root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/vqa/datasets/vqa2")
        if self.topk_flag:
            anno_prepro_path = os.path.join(data_root_dir, f"{'normAnsOnly_' if args.norm_ans_only else ''}top{args.topk}_answers.json")
        else: # min_ans_occ
            anno_prepro_path = os.path.join(data_root_dir, f"{'normAnsOnly_' if args.norm_ans_only else ''}occ_gt{args.min_ans_occ}_answers.json")
        if os.path.exists(anno_prepro_path):
            self.ans2idx = myutils.load_json(anno_prepro_path)
        else:
            self.create_ans2idx(version)
            self.ans2idx = myutils.load_json(anno_prepro_path)

        if self.min_ans_occ_flag:
            self.ans2idx = {ans:ans_idx for ans_idx, ans in enumerate(self.ans2idx)}
        else:   # topk_flag
            self.ans2idx = {ans[0]:ans_idx for ans_idx, ans in enumerate(self.ans2idx)}
        self.idx2ans = {idx:ans for ans,idx in self.ans2idx.items()}

        if self.args.model == "BUTD":
            if self.args.dataset in ["VQA","VQACP"]:
                #TODO acknowledge this difference raise NotImplementedError("pretrained-vqa tokeniser isnt available currently")
                self.tokeniser = BasicTokenizer.from_pretrained("pretrained-vqa2")
            elif self.args.dataset in ["VQA2","VQACP2"]:
                self.tokeniser = BasicTokenizer.from_pretrained("pretrained-vqa2")
            else:
                raise NotImplementedError("Not done for GQA yet")
        else:
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
        elif version == "v1":
            if split == "train":
                self.qs = myutils.load_json(os.path.join(data_root_dir, "train", "OpenEnded_mscoco_train2014_questions.json"))
                self.ans = myutils.load_json(os.path.join(data_root_dir, "train", "processed_mscoco_train2014_annotations.json"))
            elif split == "valid":
                self.qs = myutils.load_json(os.path.join(data_root_dir, "val", "OpenEnded_mscoco_val2014_questions.json"))
                self.ans = myutils.load_json(os.path.join(data_root_dir, "val", "processed_mscoco_val2014_annotations.json"))
            self.qs = self.qs['questions']
        elif version == "v2":
            if split == "train":
                self.qs = myutils.load_json(os.path.join(data_root_dir, "train", "v2_OpenEnded_mscoco_train2014_questions.json"))
                self.ans = myutils.load_json(os.path.join(data_root_dir, "train", "processed_v2_mscoco_train2014_annotations.json"))
            elif split == "valid":
                self.qs = myutils.load_json(os.path.join(data_root_dir, "val", "v2_OpenEnded_mscoco_val2014_questions.json"))
                self.ans = myutils.load_json(os.path.join(data_root_dir, "val", "processed_v2_mscoco_val2014_annotations.json"))
            self.qs = self.qs['questions']

        # VQA-CP, remove all questions that don't have an answer given the answer scheme
        original_len = len(self.qs)
        original_n_ans = []
        for q_idx in range(len(self.qs)-1, -1, -1): # Using range in reverse means we shift our start and end points by -1 to get the right values
            scores = self.ans[q_idx]["scores"]
            answer = max(scores, key=scores.get)
            original_n_ans.append(answer)
            answer = self.ans2idx.get(answer, -1) # The final key is the designated no answer token 
            if answer == -1: # If this answer iear not in ans2idx
               del self.qs[q_idx]
               del self.ans[q_idx]
        original_n_ans = len(set(original_n_ans))
        # TODO DEPRECATED self.ans2idx = {ans:i for i, ans in enumerate(self.ans2idx)}
        # Print the percentage of questions with valid answer
        print(f"There are {len(self.ans2idx)} answers in this {'topk='+str(args.topk) if self.topk_flag else 'min_ans_occ='+str(args.min_ans_occ)} {'(keeping only questions with psycholinguistic norms)' if args.norm_ans_only else ''} scheme")
        print(f"{100*len(self.qs)/original_len}% of dataset kept. Full Dataset: {original_len}, Kept dataset: {len(self.qs)}")
        print(f"{100*len(self.ans2idx)/original_n_ans}% of unique answers kept. Full Dataset: {original_n_ans}, Kept dataset: {len(self.ans2idx)}")

        # Objects
        if self.objects_flag:
            object_root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/vqa/features/coco-bottom-up/trainval")
            self.object_root_dir = object_root_dir 
            #TODO DEPRECATED h5_path = os.path.join(object_root_dir, "features.h5")
            #self.h5_path = h5_path
            #if not os.path.exists(h5_path):
            #    print(f"No features/bbox files. Generating them at {h5_path}. This'll take a while...")
            #    dset_utils.vqa_tsv_to_h5( os.path.join(object_root_dir, "karpathy_val_resnet101_faster_rcnn_genome.tsv"), h5_path )
            #    dset_utils.vqa_tsv_to_h5( os.path.join(object_root_dir, "karpathy_test_resnet101_faster_rcnn_genome.tsv"), h5_path )
            #    dset_utils.vqa_tsv_to_h5( os.path.join(object_root_dir, "karpathy_train_resnet101_faster_rcnn_genome.tsv.0"), h5_path )
            #    dset_utils.vqa_tsv_to_h5( os.path.join(object_root_dir, "karpathy_train_resnet101_faster_rcnn_genome.tsv.1"), h5_path )
            #    print("Created h5 file! Continuing...")
            #    #self.feats = h5py.File(h5_path, "r", driver=None)                
            #else:
            #    pass
            #    #self.feats = h5py.File(h5_path, "r", driver=None)# MOVED to __getitem__ to avoid num_workers>0 error with h5
        if self.images_flag:
            raise NotImplementedError(f"This is implemented and working, but shouldnt be used right now until needed")
            self.images_root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/vqa/images")
        # Pre-extracted resnet features
        if self.resnet_flag:
            raise NotImplementedError("h5 with multiple workers AND multiple proceses crash. remove this if you dont care")
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

    def __len__(self):
        return len(self.qs)

    def __getitem__(self, idx):
        # TODO DEPRECATED?
        #if self.objects_flag:
        #    if not hasattr(self, 'feats'):
        #        self.feats = h5py.File(self.h5_path, "r")#, driver=None)
        #if self.resnet_flag:
        #    if not hasattr(self, "resnet_h5"):
        #        self.resnet_h5 = h5py.File(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/vqa/resnet", "resnet.h5"), "r", driver="core")   # File is small enough to fit in memory
        #TODO deprecated? question = torch.LongTensor(self.tokeniser(self.qs[idx]['question'], padding="max_length", truncation=True, max_length=self.max_q_len)["input_ids"])
        if self.args.model == "BUTD":
            question = torch.LongTensor(self.tokeniser(self.qs[idx]['question']))
        else:
            question = torch.LongTensor(self.tokeniser(self.qs[idx]['question'])["input_ids"])
        scores = self.ans[idx]["scores"]
        answer = max(scores, key=scores.get)
        answer_text = max(scores, key=scores.get)
        answer = self.ans2idx[answer]                  
        answer = torch.LongTensor([ answer ])         
        img_id = self.qs[idx]['image_id']
        if self.objects_flag:
            obj_data = myutils.load_pickle( os.path.join(self.object_root_dir, f"{img_id}.pickle") )
            bboxes = torch.from_numpy(obj_data['bboxes'][:self.n_objs]).round()
            features = torch.from_numpy(obj_data['features'][:self.n_objs])
            # TODO DEPRECATED?
            #bboxes = torch.from_numpy(self.feats[str(img_id)]['bboxes'][:self.n_objs]).round()
            #features = torch.from_numpy(self.feats[str(img_id)]['features'][:self.n_objs])
        else:   # Create dummy inputs
            bboxes = torch.zeros(self.n_objs, 4)
            features = torch.zeros(self.n_objs, 2048)
        if self.images_flag:
            # TODO finish implementing VQACP images input
            split = self.qs[idx]["coco_split"]
            padded_img_id = f"{self.qs[idx]['image_id']:012}"
            image = cv2.imread(f"{self.images_root_dir}/{split}/COCO_{split}_{padded_img_id}.jpg")
            image = torch.from_numpy(image).permute(2,0,1) # (channels, height, width)
            img_dims = torch.tensor(image.shape[1:])
        else:
            image = torch.zeros(3,244,244)
            img_dims = torch.tensor(image.shape[1:])
            
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
        if self.args.dataset in ["VQACP","VQACP2"]:
            if self.qs[idx]["coco_split"] == "train2014":
                ret_img_id= torch.Tensor([0, img_id]).long()
            elif self.qs[idx]["coco_split"] == "val2014":
                ret_img_id= torch.Tensor([1, img_id]).long()
            else:
                raise ValueError("You got the split wrong Tom")# TODO remove this after works???
        else:
            ret_img_id = self.qs[idx]['image_id']
            #split = 0 if self.split == "train" else 1
            #ret_img_id = torch.Tensor([split, ret_img_id]).long()
            ret_img_id = torch.Tensor([ret_img_id]).long()
        q_id_ret = torch.tensor([self.qs[idx]['question_id']])
        return question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens, ret_img_id, q_id_ret, img_dims
        #      question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens, ret_img_id, q_id_ret, img_dims


    # UTILITY FUNCTIONS
    def create_ans2idx(self, version):
        #TODO This is an untidy update of previous code versions and should be streamlined later
        # Note that these are just an ordered list of answers, not a dictionary of them. You can derive ans2idx by simply enumerating the list
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
        elif version == "v1":
            data_root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/vqa/datasets/vqa")
            train_path = os.path.join(data_root_dir, "train", "mscoco_train2014_annotations.json")
            valid_path = os.path.join(data_root_dir, "val", "mscoco_val2014_annotations.json")
            train_annotations = myutils.load_json(train_path)
            valid_annotations = myutils.load_json(valid_path)
            train_annotations = train_annotations["annotations"]
            valid_annotations = valid_annotations["annotations"]
            train_path = os.path.join(data_root_dir, "train", "processed_mscoco_train2014_annotations.json")
            valid_path = os.path.join(data_root_dir, "val", "processed_mscoco_val2014_annotations.json")
        elif version == "v2":
            data_root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/vqa/datasets/vqa2")
            train_path = os.path.join(data_root_dir, "train", "v2_mscoco_train2014_annotations.json")
            valid_path = os.path.join(data_root_dir, "val", "v2_mscoco_val2014_annotations.json")
            train_annotations = myutils.load_json(train_path)
            valid_annotations = myutils.load_json(valid_path)
            train_annotations = train_annotations["annotations"]
            valid_annotations = valid_annotations["annotations"]
            train_path = os.path.join(data_root_dir, "train", "processed_v2_mscoco_train2014_annotations.json")
            valid_path = os.path.join(data_root_dir, "val", "processed_v2_mscoco_val2014_annotations.json")
        answers_path = os.path.join(data_root_dir)
        # Process annotations
        process_annotations(
            train_annotations, 
            valid_annotations, 
            train_path,
            valid_path,
            answers_path,
            self.args,
            self.norm_dict
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
        self.norm_dict = myutils.load_norms_pickle( os.path.join(os.path.dirname(__file__),"misc/all_norms.pickle")) 

        # Tokeniser
        if self.args.model == "BUTD":
            self.tokeniser = BasicTokenizer.from_pretrained("pretrained-vqa2")
        else:
            self.tokeniser = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        # Questions and Answers 
        if split == "train":
            self.q_as = myutils.load_json(os.path.join(data_root_dir, "train_balanced_questions.json"))
            ans2idxFile = f"{'normAnsOnly_' if args.norm_ans_only else ''}ans2idx.pickle"
        elif split == "valid":
            self.q_as = myutils.load_json(os.path.join(data_root_dir, "val_balanced_questions.json"))
            ans2idxFile = f"{'normAnsOnly_' if args.norm_ans_only else ''}ans2idx.pickle"
        elif split == "train-absMixed":
            self.q_as = myutils.load_json(os.path.join(data_root_dir, "absMixed_train_questions.json"))
            ans2idxFile = f"{'normAnsOnly_' if args.norm_ans_only else ''}ans2idx-absMixed.pickle"
        elif split == "valid-absMixed":
            self.q_as = myutils.load_json(os.path.join(data_root_dir, "absMixed_val_questions.json"))
            ans2idxFile = f"{'normAnsOnly_' if args.norm_ans_only else ''}ans2idx-absMixed.pickle"
        # Ans2Idx
        #if split == "train-absMixed":
        #    print("ALLOW FULL GQA")
        #    sub_keys = list(self.q_as.keys())[:100]
        #    self.q_as = {key:self.q_as[key] for key in sub_keys}
        
        if self.args.norm_ans_only:
            self.q_as = {key:value for key,value in self.q_as.items() if word_is_cOrI(self.norm_dict, value['answer'])}

        ans2idx_path = os.path.join(data_root_dir, ans2idxFile)
        if os.path.exists(ans2idx_path):
            self.ans2idx = myutils.load_pickle(ans2idx_path)
            self.idx2ans = {value:key for key,value in self.ans2idx.items()}
        else:
            print(f"{ans2idxFile} for this dataset split not found. generating...")
            if ans2idxFile == "ans2idx-absMixed.pickle":
                # GQA-ABSMIXED
                train_path = os.path.join(data_root_dir, "absMixed_train_questions.json")
                valid_path = os.path.join(data_root_dir, "absMixed_val_questions.json")
            else:
                # GQA
                train_path = os.path.join(data_root_dir, "train_balanced_questions.json")
                valid_path = os.path.join(data_root_dir, "val_balanced_questions.json")
            self.create_ans2idx(train_path=train_path, valid_path=valid_path, save_path=ans2idx_path)
            print(f"{ans2idxFile} created! Continuing...")
            self.ans2idx = myutils.load_pickle(ans2idx_path)
            self.idx2ans = {value:key for key,value in self.ans2idx.items()}

        self.idx_2_q = {q_idx:key for q_idx, key in enumerate(self.q_as.keys())}
        # Objects
        if self.objects_flag:
            self.objects_json = myutils.load_json(os.path.join(self.data_root_dir, "objects", "gqa_objects_info.json"))
            # This will be handled in __getitem__ because of h5py parallelism problem
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

    #def load_obj_h5(self):
    # TODO DEPRECATED
    #    data_root_dir = self.data_root_dir
    #    self.objects_json = myutils.load_json(os.path.join(data_root_dir, "objects", "gqa_objects_info.json"))
    #    self.objects_h5s = {
    #        0:h5py.File(os.path.join(data_root_dir, "objects", "gqa_objects_0.h5"), "r"),#, driver=None),
    #        1:h5py.File(os.path.join(data_root_dir, "objects", "gqa_objects_1.h5"), "r"),#, driver=None),
    #        2:h5py.File(os.path.join(data_root_dir, "objects", "gqa_objects_2.h5"), "r"),#, driver=None),
    #        3:h5py.File(os.path.join(data_root_dir, "objects", "gqa_objects_3.h5"), "r"),#, driver=None),
    #        4:h5py.File(os.path.join(data_root_dir, "objects", "gqa_objects_4.h5"), "r"),#, driver=None),
    #        5:h5py.File(os.path.join(data_root_dir, "objects", "gqa_objects_5.h5"), "r"),#, driver=None),
    #        6:h5py.File(os.path.join(data_root_dir, "objects", "gqa_objects_6.h5"), "r"),#, driver=None),
    #        7:h5py.File(os.path.join(data_root_dir, "objects", "gqa_objects_7.h5"), "r"),#, driver=None),
    #        8:h5py.File(os.path.join(data_root_dir, "objects", "gqa_objects_8.h5"), "r"),#, driver=None),
    #        9:h5py.File(os.path.join(data_root_dir, "objects", "gqa_objects_9.h5"), "r"),#, driver=None),
    #        10:h5py.File(os.path.join(data_root_dir, "objects", "gqa_objects_10.h5"), "r"),#, driver=None),
    #        11:h5py.File(os.path.join(data_root_dir, "objects", "gqa_objects_11.h5"), "r"),#, driver=None),
    #        12:h5py.File(os.path.join(data_root_dir, "objects", "gqa_objects_12.h5"), "r"),#, driver=None),
    #        13:h5py.File(os.path.join(data_root_dir, "objects", "gqa_objects_13.h5"), "r"),#, driver=None),
    #        14:h5py.File(os.path.join(data_root_dir, "objects", "gqa_objects_14.h5"), "r"),#, driver=None),
    #        15:h5py.File(os.path.join(data_root_dir, "objects", "gqa_objects_15.h5"), "r"),#, driver=None)
    #    }

    def __len__(self):
        return len(self.q_as)

    def __getitem__(self, idx):
        #if self.objects_flag:
        #    if not hasattr(self, 'objects_h5s'):
        #        self.load_obj_h5()
        #if self.resnet_flag:
        #    if not hasattr(self, "resnet_h5"):
        #        self.resnet_h5 = h5py.File(os.path.join(self.data_root_dir, "resnet", "resnet.h5"), "r", driver="core") # small enough
        # Question
        q_idx = self.idx_2_q[idx]
        if self.args.model == "BUTD":
            question = torch.LongTensor(self.tokeniser(self.q_as[q_idx]['question']))
        else:
            question = torch.LongTensor(self.tokeniser(self.q_as[q_idx]['question'])["input_ids"])
        # Answer
        answer = torch.LongTensor([ self.ans2idx[self.q_as[q_idx]['answer']] ])
        img_id = self.q_as[q_idx]['imageId']
        # Objects
        img_dims = torch.tensor([self.objects_json[img_id]['width'],self.objects_json[img_id]['height']]).long()
        if self.objects_flag:
            #ih5_file, ih5_idx = self.objects_json[img_id]['file'], self.objects_json[img_id]['idx']
            objs_data = myutils.load_pickle(os.path.join(self.data_root_dir, "objects", f"{img_id}.pickle"))
            bboxes = torch.from_numpy(objs_data['bboxes'][:self.n_objs]).round()
            #bboxes = torch.from_numpy(self.objects_h5s[ih5_file]['bboxes'][ih5_idx][:self.n_objs]).round()
            bboxes[:,0]/=img_dims[0]
            bboxes[:,1]/=img_dims[1]
            bboxes[:,2]/=img_dims[0]
            bboxes[:,3]/=img_dims[1]
            features = torch.from_numpy(objs_data['features'][:self.n_objs])
            #features = torch.from_numpy(self.objects_h5s[ih5_file]['features'][ih5_idx][:self.n_objs])
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
        if self.args.norm_ans_only:
            answers = [ans for ans in answers if word_is_cOrI(self.norm_dict, ans)]
        ans2idx = {answer:a_idx for a_idx, answer in enumerate(answers)}
        myutils.save_pickle(ans2idx, save_path)
