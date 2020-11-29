__author__ = "Jumperkables"
import os, sys

import torch
from torch import nn
import torch.nn.functional as F

import hopfield_layers.modules as hpf
from transformers import BertModel, BertConfig, BertTokenizer, AlbertModel, AlbertConfig, AlbertTokenizer

class Assoc_vs_Ctgrcl(nn.Module):
    def __init__(self):
        # Associative Stream
        pass
        # Categorical Stream

    def forward(x):
        # Associative Stream

        # Categorical Stream

        return x

if __name__ == "__main__":
    hh = hpf.HopfieldLayer(50)
