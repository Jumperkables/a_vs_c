#TODO Sort this script out at some point
import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#from multimodal.datasets import VQA, VQA2, VQACP, VQACP2, VQACPDataModule, VQACP2DataModule
from transformers import LxmertConfig, LxmertForQuestionAnswering, LxmertModel, LxmertTokenizer, BertTokenizer, BertModel, BertConfig

# Local imports
from misc.BERT_analysis import get_model_tokenizer, analyse_sequences, get_dset_at_norm_threshold
import myutils


if __name__ == "__main__":
    device = 1
    norm_dict = myutils.load_norms_pickle( os.path.join(os.path.dirname(__file__),"misc/all_norms.pickle"))
    model, tokeniser = get_model_tokenizer(model="lxmert", device=device, model_path=None)
    conc_sequences = get_dset_at_norm_threshold(dataset="GQA", norm="conc-m", norm_threshold=0.98, greater_than=True, norm_dict=norm_dict, vqa_condition=None)
    # vqa_condition = "topk" or "mao" only relevant for VQACP
    analyse_sequences(
        model_name="lxmert", 
        model=model, 
        sequences=conc_sequences, 
        max_seq_len=470, 
        tokenizer=tokeniser, 
        plot_title="Standard LXMERT Concrete", 
        save_path=os.path.dirname(__file__), 
        threshold=0.9, 
        mode="mean", 
        device=device
    )
