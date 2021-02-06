# Standard imports
import os, sys
import argparse
import ipdb
import h5py
from tqdm import tqdm


# Complex imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#from multimodal.datasets import VQA, VQA2, VQACP, VQACP2, VQACPDataModule, VQACP2DataModule
from transformers import LxmertConfig, LxmertForQuestionAnswering, LxmertModel, LxmertTokenizer, BertTokenizer
import pytorch_lightning as pl

# Local imports
import myutils, dset_utils
from misc.multimodal_pip_vqa_utils import process_annotations
#from models.assoc_vs_ctgrcl import VQA_AvsC

class vqa_dummy_args():
    """
    Dummy class to contain vqa args
    """
    def __init__(self, topk=-1, mao=-1):
        self.topk = topk
        self.mao = mao


# Dataset classes
class VQA(Dataset):
    """
    The VQA Changing Priors Dataset
    """
    def __init__(self, args, version="cp-v1", split="train", images=False, spatial=False, objects=False, obj_names=False, n_objs=10, max_q_len=30):
        # Feature flags
        self.images_flag = images
        self.spatial_flag = spatial
        self.objects_flag = objects
        self.obj_names_flag = obj_names
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
            if not os.path.exists(h5_path):
                print(f"No features/bbox files. Generating them at {h5_path}. This'll take a while...")
                dset_utils.vqa_tsv_to_h5( os.path.join(object_root_dir, "karpathy_val_resnet101_faster_rcnn_genome.tsv"), h5_path )
                dset_utils.vqa_tsv_to_h5( os.path.join(object_root_dir, "karpathy_test_resnet101_faster_rcnn_genome.tsv"), h5_path )
                dset_utils.vqa_tsv_to_h5( os.path.join(object_root_dir, "karpathy_train_resnet101_faster_rcnn_genome.tsv.0"), h5_path )
                dset_utils.vqa_tsv_to_h5( os.path.join(object_root_dir, "karpathy_train_resnet101_faster_rcnn_genome.tsv.1"), h5_path )
                print("Created h5 file! Continuing...")
                self.feats = h5py.File(h5_path, "r", driver=None)                
            else:
                self.feats = h5py.File(h5_path, "r", driver=None)
        self.features = []
        self.features += ['images' if images else '']
        self.features += ['spatial' if spatial else '']
        self.features += ['objects' if objects else '']
        self.features += ['obj_names' if obj_names else '']
        nl = "\n"
        print(f"{split}{nl}Features:{nl}{nl.join(self.features)}")

    def __len__(self):
        return len(self.qs)

    def __getitem__(self, idx):
        question = torch.LongTensor(self.tokeniser(self.qs[idx]['question'], padding="max_length", truncation=True, max_length=self.max_q_len)["input_ids"])
        scores = self.ans[idx]["scores"]
        answer = max(scores, key=scores.get)
        answer = self.ans2idx.get(answer, len(self.ans2idx)) # The final key is the designated no answer token 
        answer = torch.LongTensor([ answer ])            # i.e. len(ans2idx) == 3000 => 0-2999 answer ids and 3000 is the unknown token
        img_id = self.qs[idx]['image_id']
        if self.objects_flag:
            bboxes = torch.from_numpy(self.feats[str(img_id)]['bboxes'][:self.n_objs]).round()
            features = torch.from_numpy(self.feats[str(img_id)]['features'][:self.n_objs])
        else:   # Create dummy inputs
            bboxes = torch.zeros(self.n_objs, 4)
            features = torch.zeros(self.n_objs, 2048)
        #print(question.shape, answer.shape, bboxes.shape, features.shape)
        return question, answer, bboxes, features 


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
    def __init__(self, args, split="train", images=False, spatial=False, objects=False, obj_names=False, n_objs=10, max_q_len=30):
        # Feature flags
        self.images_flag = images
        self.spatial_flag = spatial
        self.objects_flag = objects
        self.obj_names_flag = obj_names
        self.n_objs = n_objs
        self.max_q_len = max_q_len
        # Answer2Idx
        data_root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/gqa")
        if (not os.path.exists(os.path.join(data_root_dir, "processed_train_q_as.pickle"))) or (not os.path.exists(os.path.join(data_root_dir, "processed_valid_q_as.pickle"))):
            print(f"Processed questions/answers for valid or train split doesn't exist. Generating...")
            self.process_q_as() 
        if os.path.exists(os.path.join(data_root_dir, "ans2idx.pickle")):
            self.ans2idx = myutils.load_pickle(os.path.join(data_root_dir, "ans2idx.pickle"))
        else:
            raise FileNotFoundError(f"This should not have happened. process_q_as() call should have generated the ans2idx file.")
        # Tokeniser
        self.tokeniser = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        # Questions and Answers
        if split == "train":
            #self.q_as = myutils.load_json(os.path.join(data_root_dir, "train_balanced_questions.json"))
            self.q_as = myutils.load_pickle(os.path.join(data_root_dir, "processed_train_q_as.pickle"))
        if split == "valid":
            #self.q_as = myutils.load_json(os.path.join(data_root_dir, "val_balanced_questions.json"))
            self.q_as = myutils.load_pickle(os.path.join(data_root_dir, "processed_valid_q_as.pickle"))

        # Objects
        if self.objects_flag:
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
        self.features = []
        self.features += ['images' if images else '']
        self.features += ['spatial' if spatial else '']
        self.features += ['objects' if objects else '']
        self.features += ['obj_names' if obj_names else '']
        nl = "\n"
        print(f"{split}{nl}Features:{nl}{nl.join(self.features)}")

    def __len__(self):
        return len(self.q_as)

    def __getitem__(self, idx):
        question = torch.LongTensor(self.tokeniser(self.q_as[idx]['question'], padding="max_length", truncation=True, max_length=self.max_q_len)["input_ids"])
        answer = torch.LongTensor([ self.ans2idx[self.q_as[idx]['answer']] ])
        img_id = self.q_as[idx]['imageId']
        if self.objects_flag:
            ih5_file, ih5_idx = self.objects_json[img_id]['file'], self.objects_json[img_id]['idx']
            bboxes = torch.from_numpy(self.objects_h5s[ih5_file]['bboxes'][ih5_idx][:self.n_objs]).round()
            features = torch.from_numpy(self.objects_h5s[ih5_file]['features'][ih5_idx][:self.n_objs])
        else:   # Create dummy inputs
            bboxes = torch.zeros(self.n_objs, 4)
            features = torch.zeros(self.n_objs, 2048)
        #print(question.shape, answer.shape, bboxes.shape, features.shape)
        return question, answer, bboxes, features 


    # UTILITY FUNCTIONS
    def create_ans2idx(questions):
        data_root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/gqa")
        save_path = os.path.join(data_root_dir, "ans2idx.pickle")
        answers = []
        train_questions = myutils.load_json(os.path.join(data_root_dir, "train_balanced_questions.json"))
        valid_questions = myutils.load_json(os.path.join(data_root_dir, "val_balanced_questions.json")) 
        for idx, key in tqdm(enumerate(train_questions.keys()), total=len(train_questions)):
            answers.append(train_questions[key]['answer'])
        for idx, key in tqdm(enumerate(valid_questions.keys()), total=len(valid_questions)):
            answers.append(valid_questions[key]['answer'])
        answers = list(set(answers))
        ans2idx = {answer:a_idx for a_idx, answer in enumerate(answers)}
        myutils.save_pickle(ans2idx, save_path)

    # Needed to index questions with integers for dataloader
    def process_q_as(self):
        # Answer2Idx
        data_root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/gqa")
        if os.path.exists(os.path.join(data_root_dir, "ans2idx.pickle")):
            ans2idx = myutils.load_pickle(os.path.join(data_root_dir, "ans2idx.pickle"))
        else:
            print("ans2idx file doesn't exist. Creating...")
            self.create_ans2idx()
            ans2idx = myutils.load_pickle(os.path.join(data_root_dir, "ans2idx.pickle"))
            print(f"ans2idx created at {os.path.join(data_root_dir, 'ans2idx.pickle')}") 
        # Tokeniser
        tokeniser = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        # Questions and Answers
        # Train
        questions = myutils.load_json(os.path.join(data_root_dir, "train_balanced_questions.json"))
        q_as = {}
        for idx, key in tqdm(enumerate(questions.keys()), total=len(questions)):
            q_as[idx] = questions[key]
            # TODO DEPRECATED?
            #q_as[idx] = {}
            #q_as[idx]['idx'] = idx
            #q_as[idx]['question'] = questions[key]['question']
            #q_as[idx]['answer'] = questions[key]['answer']
            #q_as[idx]['annotations'] = questions[key]['annotations']
            #q_as[idx]['imageId'] = questions[key]['imageId']
        myutils.save_pickle(q_as, os.path.join(data_root_dir, "processed_train_q_as.pickle")) 
        # Validation
        questions = myutils.load_json(os.path.join(data_root_dir, "val_balanced_questions.json"))
        q_as = {}
        for idx, key in tqdm(enumerate(questions.keys()), total=len(questions)):
            q_as[idx] = questions[key]
            # TODO DEPRECATED?
            #q_as[idx] = {}
            #q_as[idx]['idx'] = idx
            #q_as[idx]['question'] = questions[key]['question']
            #q_as[idx]['answer'] = questions[key]['answer']
            #q_as[idx]['annotations'] = questions[key]['annotations']
            #q_as[idx]['imageId'] = questions[key]['imageId']
        myutils.save_pickle(q_as, os.path.join(data_root_dir, "processed_valid_q_as.pickle")) 













"""
Pytorch_Lightning Model handling system
"""
# Pytorch_Lightning is a package that cleanly handles training and testing pytorch models. Check their website https://www.pytorchlightning.ai/
class VQACP_AvsC(pl.LightningModule):
    def __init__(self, args, n_answers):
        super().__init__()
        self.args = args
        lxmert_cfg = LxmertConfig()
        self.lxmert = LxmertModel(lxmert_cfg)
        fc_intermediate = ((n_answers-768)//2)+768
        self.classifier_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768, fc_intermediate),
            nn.BatchNorm1d(fc_intermediate),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fc_intermediate, n_answers+1)   # n+1 (includes unknown answer token)
        )
        if args.unfreeze == "all":
            pass
        elif args.unfreeze == "heads":
            for param in self.lxmert.base_model.parameters():
                param.requires_grad = False
        elif args.unfreeze == "none":
            for param in self.lxmert.parameters():
                param.requires_grad = False
        self.criterion = nn.CrossEntropyLoss()
        self.valid_acc = pl.metrics.Accuracy()

    def forward(self, question, bboxes, features):
        out = self.lxmert(question, features, bboxes)[2]    #['language_output', 'vision_output', 'pooled_output']
        out = self.classifier_fc(out)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        # Prepare data
        question, answer, bboxes, features = train_batch
        out = self(question, bboxes, features)
        train_loss = self.criterion(out, answer.squeeze(1))
        self.log("train_loss", train_loss, prog_bar=True, on_step=False, on_epoch=True)
        return train_loss

    def validation_step(self, val_batch, batch_idx):
        question, answer, bboxes, features = val_batch
        out = self(question, bboxes, features)
        valid_loss = self.criterion(out, answer.squeeze(1))
        self.log("valid_loss", valid_loss, on_step=False, on_epoch=True)
        self.log("valid_acc", self.valid_acc(out, answer.squeeze(1)), prog_bar=True, on_step=False, on_epoch=True)
        return valid_loss


# GQA LightningModule
class GQA_AvsC(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        lxmert_cfg = LxmertConfig()
        self.lxmert = LxmertModel(lxmert_cfg)
        self.classifier_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768, 1200),
            nn.BatchNorm1d(1200),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1200, 1842)   # 1842 unique answers
        )
        if args.unfreeze == "all":
            pass
        elif args.unfreeze == "heads":
            for param in self.lxmert.base_model.parameters():
                param.requires_grad = False
        elif args.unfreeze == "none":
            for param in self.lxmert.parameters():
                param.requires_grad = False
        self.criterion = nn.CrossEntropyLoss()
        self.valid_acc = pl.metrics.Accuracy()

    def forward(self, question, bboxes, features):
        out = self.lxmert(question, features, bboxes)[2]    #['language_output', 'vision_output', 'pooled_output']
        out = self.classifier_fc(out)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        # Prepare data
        question, answer, bboxes, features = train_batch
        out = self(question, bboxes, features)
        train_loss = self.criterion(out, answer.squeeze(1))
        self.log("train_loss", train_loss, prog_bar=True, on_step=False, on_epoch=True)
        return train_loss

    def validation_step(self, val_batch, batch_idx):
        question, answer, bboxes, features = val_batch
        out = self(question, bboxes, features)
        valid_loss = self.criterion(out, answer.squeeze(1))
        self.log("valid_loss", valid_loss, on_step=False, on_epoch=True)
        self.log("valid_acc", self.valid_acc(out, answer.squeeze(1)), prog_bar=True, on_step=False, on_epoch=True)
        return valid_loss










if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument_group("Running Arguments")
    parser.add_argument("--jobname", type=str, required=True, help="Jobname")
    parser.add_argument("--dataset", type=str, required=True, choices=["VQACP","VQACP2","GQA"], help="Choose VQA dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--bsz", type=int, default=16, help="Training batch size")
    parser.add_argument("--val_bsz", type=int, default=100, help="Validation batch size")
    parser.add_argument("--device", type=int, default=-1, help="Which device to run things on. (-1 = CPU)")
    parser.add_argument("--wandb", action="store_true", help="Plot wandb results online")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of pytroch workers. More should increase disk reads, but will increase RAM usage. 0 = main process")
    parser.add_argument_group("Model Arguments")
    parser.add_argument("--unfreeze", type=str, required=True, choices=["heads","all","none"], help="What parts of LXMERT to unfreeze")
    parser.add_argument_group("VQA-CP arguments")
    #### VQA-CP must have one of these 2 set to non-default values
    parser.add_argument("--topk", type=int, default=-1, help="Keep the k-top scoring answers. -1 implies ignore")
    parser.add_argument("--min_ans_occ", type=int, default=-1, help="The minimum occurence threshold for keeping an answers. -1 implies ignore")
    ####
    args = parser.parse_args()
    myutils.print_args(args)

    # Prepare dataloaders
    vqa_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/vqa")

    if args.dataset in ["VQACP", "VQACP2"]:
        assert (args.topk != -1) or (args.min_ans_occ != -1), f"For VQA-CP v1/2, you must set one of topk or min_ans_occ to not default. This decides which scheme to follow to keep which answers"
        assert not((args.topk != -1) and (args.min_ans_occ != -1)), f"You must leave one of topk, or min_ans_occ at default value"

    #TODO: Automate/instruct how to use the multimodal package to neatly download files
    if (args.dataset in ["VQACP", "VQACP2"]) and (not os.path.exists(os.path.join(vqa_path, "datasets"))):
        dset_utils.download_vqa()


    if args.dataset == "VQACP":
        train_dset = VQA(args, version="cp-v1", split="train", objects=True)
        valid_dset = VQA(args, version="cp-v1", split="test", objects=True)
        train_loader = DataLoader(train_dset, batch_size=args.bsz, num_workers=args.num_workers)
        valid_loader = DataLoader(valid_dset, batch_size=args.bsz, num_workers=args.num_workers)
        # TODO DEPRECATED??
        #vqacp_dm = VQACPDataModule(
        #    dir_data=vqa_path,
        #    min_ans_occ=1,
        #    features=None,
        #    label="best",#"multilabel"
        #    batch_size=args.bsz
        #)
        #train_loader = vqacp_dm.train_dataloader()
        #valid_loader = vqacp_dm.test_dataloader()
        #del vqacp_dm
        #train_dset = VQACP(split="train", features="coco-bottomup", dir_data=vqa_path)
        #valid_dset = VQACP(split="test", features="coco-bottomup", dir_data=vqa_path)
        #train_loader = DataLoader(train_dset, batch_size=args.bsz, collate_fn=VQACP.collate_fn)
        #valid_loader = DataLoader(valid_dset, batch_size=args.val_bsz, collate_fn=VQACP.collate_fn)
    elif args.dataset == "VQACP2":
        train_dset = VQA(args, version="cp-v2", split="train", objects=True)
        valid_dset = VQA(args, version="cp-v2", split="test", objects=True)
        train_loader = DataLoader(train_dset, batch_size=args.bsz, num_workers=args.num_workers)
        valid_loader = DataLoader(valid_dset, batch_size=args.bsz, num_workers=args.num_workers)
    elif args.dataset == "GQA":
        # TODO Instructions for downloading GQA
        train_dset = GQA(args, split="train", objects=True)
        valid_dset = GQA(args, split="valid", objects=True)
        train_loader = DataLoader(train_dset, batch_size=args.bsz, num_workers=args.num_workers)
        valid_loader = DataLoader(valid_dset, batch_size=args.bsz, num_workers=args.num_workers)
    
    # Prepare model & pytorch_lightning system
    wandb_logger = pl.loggers.WandbLogger(project="a_vs_c", name=args.jobname, offline=not args.wandb)#, resume="allow")
    if args.dataset in ["VQACP","VQACP2"]:
        n_answers = len(train_dset.ans2idx)
        pl_system = VQACP_AvsC(args, n_answers)
    elif args.dataset == "GQA":
        pl_system = GQA_AvsC(args)
    else:
        raise NotImplementedError(f"{args.dataset} not implemented yet")
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
