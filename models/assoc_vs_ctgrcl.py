__author__ = "Jumperkables"
# Standard
import os, sys

# Deep learning
import torch
from torch import nn
import torch.nn.functional as F
import hopfield_layers.modules as hpf
from transformers import BertModel, BertForMultipleChoice, BertForQuestionAnswering, LxmertModel, LxmertForQuestionAnswering, LxmertConfig


# To be sorted
class Assoc_vs_Ctgrcl(nn.Module):
    def __init__(self):
        # Associative Stream
        pass
        # Categorical Stream

    def forward(self, x):
        # Associative Stream

        # Categorical Stream

        return x


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--purpose", type=str, default="test_hopfield", choices=["test_hopfield"])
    args = parser.parse_args()
    if args.purpose == "test_hopfield":
        hopfield = Hopfield(
            scaling=1.0,
        
            # do not project layer input
            state_pattern_as_static=True,
            stored_pattern_as_static=True,
            pattern_projection_as_static=True,
        
            # do not pre-process layer input
            normalize_stored_pattern=False,
            normalize_stored_pattern_affine=False,
            normalize_state_pattern=False,
            normalize_state_pattern_affine=False,
            normalize_pattern_projection=False,
            normalize_pattern_projection_affine=False,
        
            # do not post-process layer output
            disable_out_projection=True
        )


