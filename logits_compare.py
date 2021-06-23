import os, sys
import argparse

import torch
import torch.nn as nn
import numpy

# Datasets
from VQA_dsets import VQA, GQA

# Models
from VQA_dsets import LxLSTM
from multimodal.models import UpDownModel
from multimodal.text import BasicTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["vqa-v1","vqa-v2","vqacp-v1","vqacp-v2","gqa"])
    parser.add_argument("--model", type=str, required=True, choices=["lx-lstm","updown"])
    ssl._create_default_https_context = ssl._create_unverified_context
    tokenizer = BasicTokenizer.from_pretrained("pretrained-vqa2")
    breakpoint()
    updown = UpDownModel(num_ans=1842, tokens=tokenizer.tokens)
    breakpoint()
    print("Clevr collected")
