__author__ = "Jumperkables"

import utils
import pandas as pd
import argparse, ipdb
from pathlib import Path

def resolve_path():
    """
    Resolve the relative path of this main.py 
    """
    return(Path(__file__).parent)

def compile_wordlist(args):
    mt40k = pd.read_csv(args.mt40k_path, sep="\t") 
    print(f"MT40k \n {mt40k} \n\n")
    return(True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mt40k_data_path", type=str, default="~/mt40k.txt", help="Path to MT40k")
    parser.add_argument("--CSLB", type=str, default="~/")

    args = parser.parse_args()
    args.mt40k_path = Path(__file__).parent.resolve() / args.mt40k_path
