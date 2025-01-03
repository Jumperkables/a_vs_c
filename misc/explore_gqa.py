import os, sys
import argparse
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import myutils
from VQA_dsets import GQA, pad_question_collate





if __name__ == "__main__":
    # Load in all train and val files, keep only global-query and other and re-save
    data_dir = os.path.join(os.path.dirname(__file__), "data/gqa")
    ########################################################################################################################
    ########################################################################################################################
    ## MIXED THE ABSTRACT AND BALANCED QUESTIONS TOGETHER
    ########################################################################################################################
    ########################################################################################################################
    #abstract_val_questions = myutils.load_json(os.path.join(data_dir, "abstract_val_questions.json"))
    #abstract_val_keys = abstract_val_questions.keys()
    #keep_val_questions = {}
    #val_questions = myutils.load_json(os.path.join(data_dir, "val_balanced_questions.json"))
    #while len(keep_val_questions) != len(abstract_val_questions):
    #    key, value = val_questions.popitem()
    #    if key not in abstract_val_keys:
    #        assert value['types']['semantic'] != "global"
    #        keep_val_questions[key] = value
    #    print(f"{len(keep_val_questions)}/{len(abstract_val_questions)}")
    #keep_val_questions = {**keep_val_questions, **abstract_val_questions}
    #print(len(keep_val_questions))
    #myutils.save_json(keep_val_questions, os.path.join(data_dir, "absMixed_val_questions.json"))    
    #abstract_train_questions = myutils.load_json(os.path.join(data_dir, "abstract_train_questions.json"))
    #abstract_train_keys = abstract_train_questions.keys()
    #keep_train_questions = {}
    #train_questions = myutils.load_json(os.path.join(data_dir, "train_balanced_questions.json"))
    #while len(keep_train_questions) != len(abstract_train_questions):
    #    key, value = train_questions.popitem()
    #    if key not in abstract_train_keys:
    #        assert value['types']['semantic'] != "global"
    #        keep_train_questions[key] = value
    #    print(f"{len(keep_train_questions)}/{len(abstract_train_questions)}")
    #keep_train_questions = {**keep_train_questions, **abstract_train_questions}
    #print(len(keep_train_questions))
    #myutils.save_json(keep_train_questions, os.path.join(data_dir, "absMixed_train_questions.json"))       

    ########################################################################################################################
    ########################################################################################################################
    ## GENERATE VALIDATION AND TRAIN 'ABSTRACT' QUESTIONS
    ########################################################################################################################
    ########################################################################################################################
    #abstract_train_questions = {}
    #abstract_val_questions = {}
    #questions = myutils.load_json(os.path.join(data_dir, "val_all_questions.json"))
    #for key, value in questions.items():
    #    if value['types']['semantic'] == "global":
    #        abstract_val_questions[key]=value
    #del questions
    #print(f"val loaded, {len(abstract_val_questions)}")
    #questions = myutils.load_json(os.path.join(data_dir, "train_all_questions", f"train_all_questions_0.json"))
    #for key, value in questions.items():
    #    if value['types']['semantic'] == "global":
    #        abstract_train_questions[key]=value
    #del questions
    #print(f"train 0 loaded, {len(abstract_train_questions)}")
    #questions = myutils.load_json(os.path.join(data_dir, "train_all_questions", f"train_all_questions_1.json"))
    #for key, value in questions.items():
    #    if value['types']['semantic'] == "global":
    #        abstract_train_questions[key]=value
    #del questions
    #print(f"train 1 loaded, {len(abstract_train_questions)}")
    #questions = myutils.load_json(os.path.join(data_dir, "train_all_questions", f"train_all_questions_2.json"))
    #for key, value in questions.items():
    #    if value['types']['semantic'] == "global":
    #        abstract_train_questions[key]=value
    #del questions
    #print(f"train 2 loaded, {len(abstract_train_questions)}")
    #questions = myutils.load_json(os.path.join(data_dir, "train_all_questions", f"train_all_questions_3.json"))
    #for key, value in questions.items():
    #    if value['types']['semantic'] == "global":
    #        abstract_train_questions[key]=value
    #del questions
    #print("train 3 loaded")
    #questions = myutils.load_json(os.path.join(data_dir, "train_all_questions", f"train_all_questions_4.json"))
    #for key, value in questions.items():
    #    if value['types']['semantic'] == "global":
    #        abstract_train_questions[key]=value
    #del questions
    #print("train 4 loaded")
    #questions = myutils.load_json(os.path.join(data_dir, "train_all_questions", f"train_all_questions_5.json"))
    #for key, value in questions.items():
    #    if value['types']['semantic'] == "global":
    #        abstract_train_questions[key]=value
    #del questions
    #print("train 5 loaded")
    #questions = myutils.load_json(os.path.join(data_dir, "train_all_questions", f"train_all_questions_6.json"))
    #for key, value in questions.items():
    #    if value['types']['semantic'] == "global":
    #        abstract_train_questions[key]=value
    #del questions
    #print("train 6 loaded")
    #questions = myutils.load_json(os.path.join(data_dir, "train_all_questions", f"train_all_questions_7.json"))
    #for key, value in questions.items():
    #    if value['types']['semantic'] == "global":
    #        abstract_train_questions[key]=value
    #del questions
    #print("train 7 loaded")
    #questions = myutils.load_json(os.path.join(data_dir, "train_all_questions", f"train_all_questions_8.json"))
    #for key, value in questions.items():
    #    if value['types']['semantic'] == "global":
    #        abstract_train_questions[key]=value
    #del questions
    #print("train 8 loaded")
    #questions = myutils.load_json(os.path.join(data_dir, "train_all_questions", f"train_all_questions_9.json"))
    #for key, value in questions.items():
    #    if value['types']['semantic'] == "global":
    #        abstract_train_questions[key]=value
    #del questions
    #print("train 9 loaded")
    #print("creating and saving abstract dictionary")
    #myutils.save_json(abstract_train_questions, os.path.join(data_dir, "abstract_train_questions.json"))
    #print(f"total of {len(abstract_train_questions)} questions!")
    #myutils.save_json(abstract_val_questions, os.path.join(data_dir, "abstract_val_questions.json"))
    #print(f"total of {len(abstract_val_questions)} questions!")
    sys.exit()
    parser = argparse.ArgumentParser()
    parser.add_argument_group("Running Arguments")
    parser.add_argument("--dataset", type=str.upper, required=True, choices=["VQACP","VQACP2","GQA"], help="Choose VQA dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--bsz", type=int, default=16, help="Training batch size")
    parser.add_argument("--val_bsz", type=int, default=100, help="Validation batch size")
    parser.add_argument("--device", type=int, default=-1, help="Which device to run things on. (-1 = CPU)")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of pytroch workers. More should increase disk reads, but will increase RAM usage. 0 = main process")
    parser.add_argument("--norm", type=str, default="conc-m", help="The norm to consider in relevant models. (conc-m == mean concreteness)")
    parser.add_argument_group("Model Arguments")

    parser.add_argument_group("Dataset arguments")
    parser.add_argument("--norm_gt", default="nsubj", choices=["answer", "nsubj"], help="Where to derive the norm information of the question. 'answer'=consider the concreteness of the answer, 'nsubj'=use the concreteness of the subject of the input question")
    parser.add_argument("--loss", type=str, default="default", choices=["default","avsc"], help="Whether or not to use a special loss")
    #### VQA-CP must have one of these 2 set to non-default values
    parser.add_argument("--topk", type=int, default=-1, help="Keep the k-top scoring answers. -1 implies ignore")
    parser.add_argument("--min_ans_occ", type=int, default=-1, help="The minimum occurence threshold for keeping an answers. -1 implies ignore")

    args = parser.parse_args()
    myutils.print_args(args)
 
    gqa_data_dir = os.path.join(os.path.dirname(__file__), "data/gqa")
   
    ################################################################################
    if False:
        import h5py
        spatial_h5_path = os.path.join(gqa_data_dir, "spatial", "gqa_spatial_0.h5")
        spatial_h5 = h5py.File(os.path.join(spatial_h5_path), "r", driver=None)
        breakpoint()
        sys.exit()
    ################################################################################

    if False:
        val_scene_graphs_path = os.path.join(gqa_data_dir, "val_sceneGraphs.json")
        val_questions_path = os.path.join(gqa_data_dir, "val_balanced_questions.json") 
    
        val_scene_graphs = myutils.load_json(val_scene_graphs_path)
        val_questions = myutils.load_json(val_questions_path)

    if args.dataset == "GQA":
        objects_flag = True 
        images_flag = False
        resnet_flag = False#True if args.model in ["hpf-2"] else False
        return_norm = True #if args.model in ["induction","hpf-0","hpf-1","hpf-2","hpf-3","dual-lx-lstm","dual-lxforqa"] else False
        return_avsc = True if args.loss in ["avsc"] else False
        #train_dset = GQA(args, split="train", objects=objects_flag, images=images_flag, resnet=resnet_flag, return_norm=return_norm, return_avsc=return_avsc)
        valid_dset = GQA(args, split="valid", objects=objects_flag, images=images_flag, resnet=resnet_flag, return_norm=return_norm, return_avsc=return_avsc)
        valid_loader = DataLoader(valid_dset, batch_size=args.bsz, num_workers=args.num_workers, drop_last=True)
        for batch_idx, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens, _ = batch
            assert answer.shape == (16,1)
            assert bboxes.shape == (16,10,4)
            assert features.shape == (16,10,2048)
            assert image.shape == (16,2048)
            assert return_norm.shape == torch.Size([16])
            assert abs_answer_tens.shape == (16,1842)
            assert conc_answer_tens.shape == (16,1842)
    breakpoint()
    print("Exiting..")
