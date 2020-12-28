__author__ = "Jumperkables"
# Standard
import os, sys

# Deep learning
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--purpose", type=str, default="test_hopfield", choices=["test_hopfield"])
    args = parser.parse_args()
    if args.purpose == "test_hopfield":
        test = hpf.Hopfield(
            input_size=450,
            hidden_size=768,
            output_size=300,
            pattern_size=50,
            num_heads=9,
            scaling=1.0,
        )
        inp = torch.ones(8,20,450)
        out = test(inp)
        print(out.shape)
        print(f"Test successful")
