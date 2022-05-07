import os, sys

import pandas as pd
import numpy as np
import scipy.special
import scipy.spatial
import scipy.stats
import torch
import torch.nn.functional as F
from math import log2

def cosine_sim(a, b):
    return scipy.spatial.distance.cosine(a, b)
    #return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def wasserstein_distance(a, b):
    a = a/np.linalg.norm(a)
    b = b/np.linalg.norm(b)
    return scipy.stats.wasserstein_distance(a, b)

#def kl_div(p, q):
#    p = p/np.linalg.norm(p)
#    q = q/np.linalg.norm(q)
#    listy = [ p[i] * log2(p[i]/q[i]) for i in range(len(p)) if (p[i]!=0 and q[i]!=0) ]
#    listy2 = [ p[i] * log2(p[i]/q[i]) if (p[i]!=0 and q[i]!=0) else 0 for i in range(len(p)) ]
#    breakpoint()
#    return sum(listy)
#    #breakpoint()
#    #return F.kl_div(torch.Tensor(a), torch.Tensor(b))
#    #return scipy.special.kl_div(a, b)


def kolmogorov_smirnov(a, b):
    # Unsure wether or not to normalise both
    #a = a/np.linalg.norm(a)
    #b = b/np.linalg.norm(b)
    return scipy.stats.ks_2samp(a, b)[0]

def spearman_corr(a, b):
    return scipy.stats.spearmanr(a, b)[0]

def simlex_assoc_compare_for_words_list(words=None):
    """
    Prints a set of comparisons for USF and SimLex words that may appear in a list of given words
    words: A list of strings from which to compare
        if words=None, simply report stats for the whole simlex dataset
    """
    assert type(words) == list or words == None, "Words should be a list, or None"
    simlex999 = pd.read_csv("/home/jumperkables/kable_management/data/a_vs_c/SimLex-999/SimLex-999.txt", delimiter="\t")
    seen = []
    to_drop = []
    # Remove the duplicate entries
    for idx, row in enumerate(simlex999.iterrows()):
        w1 = row[1]['word1']
        w2 = row[1]['word2']
        if (f"{w1}|{w2}" in seen) or (f"{w2}|{w1}" in seen):
            print("DUPLICATE FOUND!")
            #breakpoint()
            to_drop.append(row[0])
        else:
            seen.append(f"{w1}|{w2}")
    simlex999 = simlex999.drop(to_drop)
    to_drop = []
    if words != None:
        # Remove any answer that doesn't appear in the supplied words list
        for idx, row in enumerate(simlex999.iterrows()):
            w1 = row[1]['word1'].lower()
            w2 = row[1]['word2'].lower()
            if not( (w1 in words) and (w2 in words) ):
                to_drop.append(row[0])
        to_drop = list(set(to_drop))
        simlex999 = simlex999.drop(to_drop)
        unique_answers = []
        unique_answers_assoc = []
        for row in simlex999.iterrows():
            unique_answers.append(row[1]['word1'].lower())
            unique_answers.append(row[1]['word2'].lower())
        unique_answers = list(set(unique_answers))
        print(f"\nNumber of answers with Assoc or Simlex Scores: {len(unique_answers)}/{len(words)}")
        for row in simlex999[simlex999['Assoc(USF)']>0].iterrows():
            unique_answers_assoc.append(row[1]['word1'].lower())
            unique_answers_assoc.append(row[1]['word2'].lower())
        unique_answers_assoc = list(set(unique_answers_assoc))
        print(f"Number of answers with Assoc Scores: {len(unique_answers_assoc)}/{len(words)}")
        print(f"Number of answers with Simlex Scores: {len(unique_answers)}/{len(words)}") # All answers have simlex scores
        print(f"Percentage of answers with Assoc or Simlex Scores: {100*len(unique_answers)/len(words):.2f}%")
    simlex = simlex999['SimLex999'].values/10
    assoc = simlex999['Assoc(USF)'].values/10 
   
    # Cosine Similarity
    print("Cosine Similarity")
    print(f"Full Dataset: {cosine_sim(assoc, simlex):.3f}")
    
    # KL-Divergence
    #print("KL-Divergence")
    #print(f"Full Dataset: {kl_div(assoc, simlex):.5f}")
    
    # Kolmogorov-Smirnov
    print("\n\nKolmogorov-Smirnov statistic")
    print(f"Full Dataset: {kolmogorov_smirnov(assoc, simlex):.3f}")
    
    # Wasserstein Distance
    print("\n\nWasserstein Distance")
    print(f"Full Dataset: {wasserstein_distance(assoc, simlex):.3f}")
    
    # Spearman Correlation
    print("\n\nSpearman Correlation")
    print(f"Full Dataset: {spearman_corr(assoc, simlex):.3f}")


if __name__ == "__main__":
    simlex_assoc_compare_for_words_list()
