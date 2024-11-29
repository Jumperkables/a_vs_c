# standard imports
import os, sys

# 3rd party imports
import pandas as pd

# local imports
from myutils import load_pickle
import word_norms
from word_norms import Word2Norm, clean_word

norm_dict_path = os.path.join( os.path.dirname(__file__) , "all_norms.pickle")
norm_dict = load_pickle(norm_dict_path)

df = pd.read_csv('data/SimLex-999/SimLex-999.txt', sep='\t')
most_conc_words = set(df.sort_values(['conc(w1)'], ascending=False)[:310]['word1'].tolist())
breakpoint()
print("Figure out a way to make a toy dataset out of this")
