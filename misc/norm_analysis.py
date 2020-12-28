__author__ = "Jumperkables"
"""
This file contains wrapper functions for considering the clusters and neighbourhoods of certainconcepts in the norm dictionary I've collected
"""

# Standard imports
import sys, os, argparse
import math
import pandas as pd
from tqdm import tqdm

import networkx as nx

import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap,LinearSegmentedColormap

# My Imports
myimportpath = os.path.abspath(f"{os.path.abspath(__file__)}/../..")
sys.path.insert(0, myimportpath)
import word_norms
from word_norms import Word2Norm, clean_word
import myutils




"""
PLOTTING FUNCTIONS
"""
def G_2_nx(G, draw_style="default"):
    edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
    if draw_style == "spectral":
        nx.draw_spectral(G, edgelist=edges, edge_color=weights, edge_cmap=plt.cm.cool, node_size=1, font_size=7 ,with_labels=True)
    if draw_style == "default":
        nx.draw(G, edgelist=edges, edge_color=weights, edge_cmap=plt.cm.cool, node_size=1, font_size=7 ,with_labels=True)
    plt.show()

def df_2_pxtreemap(df):
    fig = px.treemap(df, labels=df.columns)
    fig.show() 


"""
NORM DICT FUNCTIONS
"""
def normdict_2_G(norm_dict, norm="assoc", norm_threshold=0.5):
    G = nx.Graph()
    word_pairs = norm_dict.word_pairs
    print(f"Process wordpairs")
    #colours = cm.get_cmap("cool", 101)
    for wpkey, wpdct in tqdm(word_pairs.items(), total=len(word_pairs)):
        w0, w1, = wpkey.split("|") # Get both words
        if True:
            nrm = wpdct.get(norm, None)
            if nrm != None:
                nrm = nrm["avg"]
                if nrm != None and (nrm>norm_threshold):
                    #print(math.floor(100*nrm))
                    G.add_edge(w0, w1, weight=nrm*10)#, color=colours(math.floor(100*nrm)))
    return(G)


def normdict_2_df(norm_dict, norm="assoc", norm_threshold=0.5):
    temp_dict = {}
    word_pairs = norm_dict.word_pairs
    print(f"Process wordpairs")
    for wpkey, wpdct in tqdm(word_pairs.items(), total=len(word_pairs)):
        w0, w1, = wpkey.split("|") # Get both words
        if True:
            nrm = wpdct.get(norm, None)
            if nrm != None:
                nrm = nrm["avg"]
                if nrm != None and (nrm>norm_threshold):
                    #print(math.floor(100*nrm))i
                    if temp_dict.get(w0, None) == None:
                        temp_dict[w0]={}
                    #import ipdb; ipdb.set_trace()
                    temp_dict[w0][w1]=nrm
    df = pd.DataFrame(temp_dict)
    label_union = df.index.union(df.columns)
    #df = df.reindex(index=label_union, columns=label_union)
    df = df.fillna(0)
    return(df)








if __name__ == "__main__":
    norm_dict_path = os.path.join( "/home/jumperkables/kable_management/projects/a_vs_c" , "misc", "all_norms.pickle")
    norm_dict = myutils.load_pickle(norm_dict_path)

    parser = argparse.ArgumentParser()
    parser.add_argument("--purpose", default="clustering", choices=["clustering"], help="How to run this file")
    args = parser.parse_args()

    if args.purpose == "clustering":
        G = normdict_2_G(norm_dict, "assoc", norm_threshold=0.4)
        G_2_nx(G, draw_style="default")
        #####
        #df = normdict_2_df(norm_dict, "assoc")
        #df_2_pxtreemap(df)
