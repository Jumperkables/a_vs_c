# Standard imports
import os, sys
import random
import h5py
import ssl
import pandas as pd
ssl._create_default_https_context = ssl._create_unverified_context
from tqdm import tqdm
from collections import Counter
import json

# Complex imports
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import LxmertTokenizer
import spacy
#from multimodal.text import BasicTokenizer

# Local imports
import misc.myutils as myutils
import misc.dset_utils as dset_utils
from misc.glossary import normalize_word
from misc.word_norms import word_is_assoc_or_simlex, wordlist_is_expanded_norm, avg_list
from misc.compare_USF_ASSOC import simlex_assoc_compare_for_words_list



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
                if args.norm_ans_only == "simlex":
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

                elif args.norm_ans_only in ["expanded", "None"]:
                    nd = norm_dict.word_pairs.get(f"{ans}|{answer}", None)
                    assoc_score = 0
                    simlex_score = 0
                    if nd != None:
                        try:
                            assoc = nd["assoc"]['sources']['USF']['scaled']
                        except KeyError:
                            assoc = None
                        try:
                            simlex = nd["simlex999-m"]["sources"]["SimLex999"]["scaled"]
                        except KeyError:
                            simlex = None
                        try:
                            sim = nd["sim"]["avg"]
                        except KeyError:
                            sim = None
                        try:
                            usf_str = nd["str"]["avg"]
                        except KeyError:
                            usf_str = None
                        #print(f"assoc: {assoc} | simlex: {simlex} | sim: {sim} | str: {usf_str}")
                        if assoc != None or sim != None or usf_str != None or simlex != None:
                            if assoc == None:
                                assoc = 0
                            if simlex == None:
                                simlex = 0
                            if sim == None:
                                sim = 0
                            if usf_str == None:
                                usf_str = 0
        
                            assoc_norms = [i for i in [assoc, usf_str] if i != 0]
                            simlex_norms = [i for i in [simlex, sim] if i != 0]
        
                            if avg_list(assoc_norms) >= args.norm_clipping:
                                assoc_score = avg_list(assoc_norms)
                            if avg_list(simlex_norms) >= args.norm_clipping:
                                simlex_score = avg_list(simlex_norms)
                    BCE_assoc_tensor.append(assoc_score)
                    BCE_ctgrcl_tensor.append(simlex_score)

        idx2BCE_assoc_tensor[idx] = torch.Tensor(BCE_assoc_tensor)
        idx2BCE_ctgrcl_tensor[idx] = torch.Tensor(BCE_ctgrcl_tensor)
    #TODO DEPRECATED # Final unknown token if needed
    #if args.dataset in ["VQA","VQA2","VQACP","VQACP2"]:
    #    idx2BCE_assoc_tensor[len(answers)] = torch.Tensor([0]*len(answers)+[1])
    #    idx2BCE_ctgrcl_tensor[len(answers)] = torch.Tensor([0]*len(answers)+[1])
    simlex_assoc_compare_for_words_list(list(ans2idx.keys()))

    # Assoc tensor
    total_avg_sum = []
    normonly_avg_sum = []
    total_avg_count = []
    normonly_avg_count = []
    for idx, tens in idx2BCE_assoc_tensor.items():
        if float(tens.sum()) != 1.:
            normonly_avg_sum.append(float(tens.sum()))
            normonly_avg_count.append(int(tens.count_nonzero()))
        total_avg_sum.append(float(tens.sum()))
        total_avg_count.append(int(tens.count_nonzero()))
    total_avg_sum = sum(total_avg_sum)/len(total_avg_sum)
    normonly_avg_sum = sum(normonly_avg_sum)/len(normonly_avg_sum)
    total_avg_count = sum(total_avg_count)/len(total_avg_count)
    normonly_avg_count = sum(normonly_avg_count)/len(normonly_avg_count)
    print(f"Unique answers with assoc score:")
    print(f"\tTotal Average Sum: {total_avg_sum:.3f}")
    print(f"\tNormonly Average Sum: {normonly_avg_sum:.3f}")
    print(f"\tTotal Average Count: {total_avg_count:.3f}")
    print(f"\tNormonly Average Count: {normonly_avg_count:.3f}")

    # Ctgrcl tensor
    total_avg_sum = []
    normonly_avg_sum = []
    total_avg_count = []
    normonly_avg_count = []
    for idx, tens in idx2BCE_ctgrcl_tensor.items():
        if float(tens.sum()) != 1.:
            normonly_avg_sum.append(float(tens.sum()))
            normonly_avg_count.append(int(tens.count_nonzero()))
        total_avg_sum.append(float(tens.sum()))
        total_avg_count.append(int(tens.count_nonzero()))
    total_avg_sum = sum(total_avg_sum)/len(total_avg_sum)
    normonly_avg_sum = sum(normonly_avg_sum)/len(normonly_avg_sum)
    total_avg_count = sum(total_avg_count)/len(total_avg_count)
    normonly_avg_count = sum(normonly_avg_count)/len(normonly_avg_count)
    print(f"Unique answers with ctgrcl score:")
    print(f"\tTotal Average Sum: {total_avg_sum:.3f}")
    print(f"\tNormonly Average Sum: {normonly_avg_sum:.3f}")
    print(f"\tTotal Average Count: {total_avg_count:.3f}")
    print(f"\tNormonly Average Count: {normonly_avg_count:.3f}")
    return idx2BCE_assoc_tensor, idx2BCE_ctgrcl_tensor, norm_dict

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
            data_root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/vqa/datasets/vqacp")
            if split == "train":
                self.qs = myutils.load_json(os.path.join(data_root_dir, "vqacp_v1_train_questions.json"))
                self.ans = myutils.load_json(os.path.join(data_root_dir, "vqacp_v1_train_annotations.json"))
            elif split == "test":
                self.qs = myutils.load_json(os.path.join(data_root_dir, "vqacp_v1_test_questions.json"))
                self.ans = myutils.load_json(os.path.join(data_root_dir, "vqacp_v1_test_annotations.json"))
        elif version == "cp-v2":
            data_root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/vqa/datasets/vqacp2")
            if split == "train":
                self.qs = myutils.load_json(os.path.join(data_root_dir, "vqacp_v2_train_questions.json"))
                self.ans = myutils.load_json(os.path.join(data_root_dir, "vqacp_v2_train_annotations.json"))
            elif split == "test":
                self.qs = myutils.load_json(os.path.join(data_root_dir, "vqacp_v2_test_questions.json"))
                self.ans = myutils.load_json(os.path.join(data_root_dir, "vqacp_v2_test_annotations.json"))
        elif version == "v1":
            data_root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/vqa/datasets/vqa")
            if split == "train":
                self.qs = myutils.load_json(os.path.join(data_root_dir, "OpenEnded_mscoco_train2014_questions.json"))
                self.ans = myutils.load_json(os.path.join(data_root_dir, "mscoco_train2014_annotations.json"))
            elif split == "valid":
                self.qs = myutils.load_json(os.path.join(data_root_dir, "OpenEnded_mscoco_val2014_questions.json"))
                self.ans = myutils.load_json(os.path.join(data_root_dir, "mscoco_val2014_annotations.json"))
            self.qs = self.qs['questions']
            self.ans = self.ans['annotations']
        elif version == "v2":
            data_root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/vqa/datasets/vqa2")
            if split == "train":
                self.qs = myutils.load_json(os.path.join(data_root_dir, "v2_OpenEnded_mscoco_train2014_questions.json"))
                self.ans = myutils.load_json(os.path.join(data_root_dir, "v2_mscoco_train2014_annotations.json"))
            elif split == "valid":
                self.qs = myutils.load_json(os.path.join(data_root_dir, "v2_OpenEnded_mscoco_val2014_questions.json"))
                self.ans = myutils.load_json(os.path.join(data_root_dir, "v2_mscoco_val2014_annotations.json"))
            self.qs = self.qs['questions']
            self.ans = self.ans['annotations']

        # Answer2Idx
        if args.norm_ans_only == "simlex":
            normonly_prefix = "AssocSimlexAnsOnly_"
        elif args.norm_ans_only == "expanded":
            normonly_prefix = f"Expanded-nc-gt{args.norm_clipping}_"
        else:
            normonly_prefix = f"full-nc-gt{args.norm_clipping}_"
        if self.topk_flag:
            ans_prepro_path = os.path.join(data_root_dir, f"{normonly_prefix}top{args.topk}_answers.json")
        else: # min_ans_occ
            ans_prepro_path = os.path.join(data_root_dir, f"{normonly_prefix}occ_gte{args.min_ans_occ}_answers.json")
        if os.path.exists(ans_prepro_path):
            self.ans2idx = myutils.load_json(ans_prepro_path)
        else:
            self.create_ansfile(version)
            self.ans2idx = myutils.load_json(ans_prepro_path)
        self.ans2idx = {ans:ans_idx for ans_idx, ans in enumerate(self.ans2idx)}
        self.idx2ans = {idx:ans for ans,idx in self.ans2idx.items()}

        # remove all questions that don't have an answer given the answer scheme
        original_len = len(self.qs)
        original_n_ans = []
        for q_idx in range(len(self.qs)-1, -1, -1): # Using range in reverse means we shift our start and end points by -1 to get the right values
            answer = self.ans[q_idx]['multiple_choice_answer']
            original_n_ans.append(answer)
            answer = self.ans2idx.get(answer, -1) # The final key is the designated no answer token 
            if answer == -1: # If this answer iear not in ans2idx
               del self.qs[q_idx]
               del self.ans[q_idx]
        original_n_ans = len(set(original_n_ans))
        if args.norm_ans_only == "expanded":
            print(f"There are {len(self.ans2idx)} answers in this {'topk='+str(args.topk) if self.topk_flag else 'min_ans_occ='+str(args.min_ans_occ)} expanded norm scheme")
        else:
            print(f"There are {len(self.ans2idx)} answers in this {'topk='+str(args.topk) if self.topk_flag else 'min_ans_occ='+str(args.min_ans_occ)} {'(keeping only questions with assoc or simlex norms)' if args.norm_ans_only else ''} scheme")
        print(f"{100*len(self.qs)/original_len}% of dataset kept. Full Dataset: {original_len}, Kept dataset: {len(self.qs)}")
        print(f"{100*len(self.ans2idx)/original_n_ans}% of unique answers kept. Full Dataset: {original_n_ans}, Kept dataset: {len(self.ans2idx)}")

        # Objects
        if self.objects_flag:
            object_root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/vqa/features/coco-bottom-up/trainval")
            self.object_root_dir = object_root_dir 
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
            self.idx2BCE_assoc_tensor, self.idx2BCE_ctgrcl_tensor, _ = set_avsc_loss_tensor(args, self.ans2idx) # loads norm_dict
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
        answer_text = normalize_word(self.ans[idx]["multiple_choice_answer"])
        answer = self.ans2idx[answer_text]                  
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


    def create_ansfile(self, version):
        #TODO This is an untidy update of previous code versions and should be streamlined later
        # Note that these are just an ordered list of answers, not a dictionary of them. You can derive ans2idx by simply enumerating the list
        answers = []
        if version == "cp-v1":
            data_root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/vqa/datasets/vqacp")
            train_path = os.path.join(data_root_dir, "vqacp_v1_train_annotations.json")
            valid_path = os.path.join(data_root_dir, "vqacp_v1_test_annotations.json")
            train_annotations = myutils.load_json(train_path)
            valid_annotations = myutils.load_json(valid_path) 
            train_path = os.path.join(data_root_dir, "vqacp_v1_train_annotations.json")
            valid_path = os.path.join(data_root_dir, "vqacp_v1_test_annotations.json")
        elif version == "cp-v2":
            data_root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/vqa/datasets/vqacp2")
            train_path = os.path.join(data_root_dir, "vqacp_v2_train_annotations.json")
            valid_path = os.path.join(data_root_dir, "vqacp_v2_test_annotations.json")
            train_annotations = myutils.load_json(train_path)
            valid_annotations = myutils.load_json(valid_path)
            train_path = os.path.join(data_root_dir, "vqacp_v2_train_annotations.json")
            valid_path = os.path.join(data_root_dir, "vqacp_v2_test_annotations.json")
        elif version == "v1":
            data_root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/vqa/datasets/vqa")
            train_path = os.path.join(data_root_dir, "mscoco_train2014_annotations.json")
            valid_path = os.path.join(data_root_dir, "mscoco_val2014_annotations.json")
            train_annotations = myutils.load_json(train_path)
            valid_annotations = myutils.load_json(valid_path)
            train_annotations = train_annotations["annotations"]
            valid_annotations = valid_annotations["annotations"]
            train_path = os.path.join(data_root_dir, "mscoco_train2014_annotations.json")
            valid_path = os.path.join(data_root_dir, "mscoco_val2014_annotations.json")
        elif version == "v2":
            data_root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/vqa/datasets/vqa2")
            train_path = os.path.join(data_root_dir, "v2_mscoco_train2014_annotations.json")
            valid_path = os.path.join(data_root_dir, "v2_mscoco_val2014_annotations.json")
            train_annotations = myutils.load_json(train_path)
            valid_annotations = myutils.load_json(valid_path)
            train_annotations = train_annotations["annotations"]
            valid_annotations = valid_annotations["annotations"]
            train_path = os.path.join(data_root_dir, "v2_mscoco_train2014_annotations.json")
            valid_path = os.path.join(data_root_dir, "v2_mscoco_val2014_annotations.json")
        answers_path = os.path.join(data_root_dir)

        # Process annotations
        top_k_flag = (self.args.topk != -1)
        min_ans_occ_flag = not top_k_flag    
        all_annotations = train_annotations + valid_annotations
        all_major_answers = []
        for annot in tqdm(all_annotations):
            nw = normalize_word(annot["multiple_choice_answer"])
            if nw != "":
                all_major_answers.append(nw)
        if min_ans_occ_flag:
            # NOTE THE DEFAULT IS self.args.min_ans_occ >= 9
            kept_answers = {k:v for k,v in Counter(all_major_answers).items() if v >= self.args.min_ans_occ}
            kept_answers = list(kept_answers.keys())
        else:
            kept_answers = Counter(all_major_answers)
            kept_answers = kept_answers.most_common(self.args.topk)
            kept_answers = [k for k,v in kept_answers]
        if self.args.norm_ans_only == "simlex":
            # Ignore all questions with answers that are not themselves a psycholinguistic conc/imag norm
            #kept_answers = [ ans for ans in kept_answers if word_is_assoc_or_simlex(ans)]
            kept_answers = wordlist_is_assoc_or_simlex(kept_answers)
        elif self.args.norm_ans_only == "expanded":
            kept_answers = wordlist_is_expanded_norm(kept_answers, self.args.norm_clipping)
        else:
            kept_answers = list(set(kept_answers))
        print(f"Number of Unique Answers: {len(kept_answers)}")
        print(f"Removing uncommon answers")

        if self.args.norm_ans_only == "simlex":
            normonly_prefix = "AssocSimlexAnsOnly_"
        elif self.args.norm_ans_only == "expanded":
            normonly_prefix = f"Expanded-nc-gt{self.args.norm_clipping}_"
        else:
            normonly_prefix = f"full-nc-gt{self.args.norm_clipping}_"
        threshold_answers_path = f"{answers_path}/{normonly_prefix}occ_gte{self.args.min_ans_occ}_answers.json"
        topk_answers_path = f"{answers_path}/{normonly_prefix}top{self.args.topk}_answers.json"
        print(f"Saving answers at {answers_path}")
        print(f"Top {self.args.topk} answers: {topk_answers_path}. Threshold > {self.args.min_ans_occ} answers:{threshold_answers_path}")
        if min_ans_occ_flag:
            with open(threshold_answers_path, "w") as f:
                json.dump(kept_answers, f)
        else:
            with open(topk_answers_path, "w") as f:
                json.dump(kept_answers, f)





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
        if args.norm_ans_only == "simlex":
            normonly_prefix = "AssocSimlexAnsOnly_"
        elif args.norm_ans_only == "expanded":
            normonly_prefix = f"Expanded-nc-gt{args.norm_clipping}_"
        else:
            normonly_prefix = f"full-nc-gt{args.norm_clipping}_"
        if split == "train":
            self.q_as = myutils.load_json(os.path.join(data_root_dir, "train_balanced_questions.json"))
            ans2idxFile = f"{normonly_prefix}ans2idx.pickle"
        elif split == "valid":
            self.q_as = myutils.load_json(os.path.join(data_root_dir, "val_balanced_questions.json"))
            ans2idxFile = f"{normonly_prefix}ans2idx.pickle"
        
        if self.args.norm_ans_only == "simlex":
            # Ignore all questions with answers that are not themselves a psycholinguistic conc/imag norm
            all_answers = [ value['answer'] for value in self.q_as.values() ]
            all_answers = list(set(all_answers))
            kept_answers = wordlist_is_assoc_or_simlex(all_answers)
            self.q_as = {key:value for key,value in self.q_as.items() if value['answer'] in kept_answers}
        elif self.args.norm_ans_only in ["None", "expanded"]:
            all_answers = [ value['answer'] for value in self.q_as.values() ]
            all_answers = list(set(all_answers))
            if self.args.norm_ans_only == "expanded":
                kept_answers = wordlist_is_expanded_norm(all_answers)
            else:
                kept_answers = all_answers
            self.q_as = {key:value for key,value in self.q_as.items() if value['answer'] in kept_answers}

        ans2idx_path = os.path.join(data_root_dir, ans2idxFile)
        if os.path.exists(ans2idx_path):
            self.ans2idx = myutils.load_pickle(ans2idx_path)
            self.idx2ans = {value:key for key,value in self.ans2idx.items()}
        else:
            print(f"{ans2idxFile} for this dataset split not found. generating...")
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
            self.idx2BCE_assoc_tensor, self.idx2BCE_ctgrcl_tensor, _ = set_avsc_loss_tensor(args, self.ans2idx) # loads norm_dict

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
        if self.args.norm_ans_only == "simlex":
            #answers = [ans for ans in answers if word_is_assoc_or_simlex(ans)]
            answers = wordlist_is_assoc_or_simlex(answers)
        elif self.args.norm_ans_only == "expanded":
            answers = wordlist_is_expanded_norm(answers)
        ans2idx = {answer:a_idx for a_idx, answer in enumerate(answers)}
        myutils.save_pickle(ans2idx, save_path)
