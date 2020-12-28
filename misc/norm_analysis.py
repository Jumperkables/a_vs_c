__author__ = "Jumperkables"
"""
This file contains wrapper functions for considering the clusters and neighbourhoods of certainconcepts in the norm dictionary I've collected
"""

# Standard imports
import sys, os 
import argparse
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
def G_2_nx(G, draw_style="default", title="Default Title", save=False, save_path="/CHANGEME"):
    if save == True:
        assert save_path != "/CHANGEME", f"If you're going to save the plot, change the save_path"   
    edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
    if draw_style == "spectral":
        nx.draw_spectral(G, edgelist=edges, edge_color=weights, edge_cmap=plt.cm.cool, node_size=1, font_size=7 ,with_labels=True)
    if draw_style == "default":
        nx.draw(G, edgelist=edges, edge_color=weights, edge_cmap=plt.cm.cool, node_size=1, font_size=7 ,with_labels=True)
    plt.title(title)
    if save == True:
        plt.savefig(save_path)
        print(f"Saved plot: {title} to {save_path}")
    else:
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
    #parser.add_argument_group("Norm processing args")
    parser.add_argument("--norm", type=str, default="assoc", help="Which norm to consider across (word pairs)")
    parser.add_argument("--norm_threshold", type=float, default=0.5, help="Ignore all norms underneath this threshold")

    #parser.add_argument_group("Drawing args")
    parser.add_argument("--purpose", type=str, default="G_2_nx", choices=["G_2_nx", "df_2_pxtreemap"], help="How to run this file")
    parser.add_argument("--draw_style", type=str, default="default", help="The drawing style for given plots")
    parser.add_argument("--save", action="store_true", help="if to save the figure generated")
    parser.add_argument("--title", type=str, default="Default Title", help="The title for a plot")
    parser.add_argument("--save_path", type=str, default="/CHANGEME", help="save path for the plot")
    args = parser.parse_args()
    myutils.print_args(args)

    if args.purpose == "G_2_nx":
        G = normdict_2_G(norm_dict, args.norm, norm_threshold=args.norm_threshold)
        G_2_nx(G, draw_style=args.draw_style, title=args.title, save=args.save, save_path=args.save_path)
    elif args.purpose == "df_2_pxtreemap":
        raise NotImplementedError("Update this to use the parser args") 
        df = normdict_2_df(norm_dict, "assoc")
        df_2_pxtreemap(df)
    else:
        print("No running purpose executed, please look at the argparser")
