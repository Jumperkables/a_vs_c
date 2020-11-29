__author__ = "Jumperkables"
import os, sys
import math
import matplotlib.pyplot as plt
import collections
import seaborn as sns
import plotly.express as px
import statistics
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F

from transformers import BertModel, BertConfig, BertTokenizer, AlbertModel, AlbertConfig, AlbertTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
import word_norms
from word_norms import Word2Norm, clean_word
import myutils


def analyse_sequences(model, sequences, max_seq_len, tokenizer, plot_title, save_path, threshold=0.9, mode="mean", device=0):
    """
    model:          A BERT model input (huggingface)
    sequence:       Sequences in strings
    max_seq_len:    Longer sequence lengths create exponentially higher runtime (attention layers are dense)
    tokenizer:      A BERT tokenizer (huggingface)
    plot_title:     Title to plot in matplotlib
    save_path:      Path to save
    device:         CUDA device ID
    """
    ################
    # Prepare sequences adequately
    assert max_seq_len >= 2, f"max_seq_len less than 2 causes a bug, not applicable"
    tokenized_sequences = [ tokenizer.tokenize(sequence) for sequence in sequences ]
    indexed_sequences = [ tokenizer.encode(tokenized_sequence, return_tensors="pt") for tokenized_sequence in tokenized_sequences ]
    sos_token, eos_token = torch.tensor([101]), torch.tensor([102])
    n_cut = 0
    for sequence in indexed_sequences:
        if len(sequence)<=max_seq_len: 
            sequence = sequence.unsqueeze(0)
        else:
            sequence = torch.cat( ( sos_token, sequence[1:max_seq_len], eos_token ) ).unsqueeze(0)
            n_cut+=1
    print(f"{n_cut}/{len(indexed_sequences)} sequences were cropped")
    test_seq = torch.tensor([[101, 102]]).to(device)
    _, _, _, attentions = model(test_seq)
    nlayers = len(attentions)
    nheads = attentions[0].shape[1]

    ################
    # Process sequences into attentions
    layer_head_k = [ [ [] for n in range(nheads) ] for k in range(nlayers) ]
    for seq in tqdm(indexed_sequences):
        with torch.no_grad():
            seq=seq.to(device)
            last_hidden_state, pooler_output, hidden_states, attentions = model(seq)
        attentions = attentions#.cpu()
        for layer in range(0,nlayers):
            for attn_head in range(0,nheads):
                self_attentions = attentions[layer][0][attn_head]
                for idx,attn in enumerate(self_attentions):
                    layer_head_k[layer][attn_head].append(myutils.n_softmax_threshold(attn, threshold=threshold))

    ################
    # Plotting time
    my_dpi = 91
    plt.figure(figsize=(900/my_dpi, 950/my_dpi), dpi=my_dpi)
    for layer in tqdm(range(0,nlayers)):
        for attn_head in range(0,nheads):
            plt.subplot(nlayers,nheads,1+((nlayers*(layer))+(attn_head)))
            layer = nlayers-layer-1
            myutils.colour_violin(layer_head_k[layer][attn_head], max_x=max_seq_len, mode=mode)
            plt.tick_params(
                axis='y',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False,
                labeltop=False
                )
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False,
                labeltop=False
            )
            if layer == (nlayers-1):
                plt.xlabel(f"Head {attn_head+1}")

            if layer == 0:
                plt.tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=True,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=True,
                    labeltop=False,
                    labelsize=6
                )

            if attn_head == 0:
                plt.ylabel(f"Layer {layer+1}")

    plt.suptitle(plot_title)
    plt.savefig(save_path)
    return f"{plot_title} saved at {save_path}"


def topkbottomk_mt40k(k=3000):
    # Create the norm dictionary
    norm_dict_path = os.path.join( os.path.dirname(os.path.dirname(__file__)) , "misc", "all_norms.pickle")
    norm_dict = myutils.load_pickle(norm_dict_path)
    word_2_concm = { word:ndict["conc-m"] for word, ndict in norm_dict.words.items() if "conc-m" in ndict.keys()}
    word_2_mt40k_conc = { word:concm["sources"]["MT40k"]["scaled"] for word,concm in word_2_concm.items() if "MT40k" in concm["sources"].keys() } 
    #wordpair_2_assoc = { wordpair:ndict["assoc"] for wordpair, ndict in norm_dict.word_pairs.items() if "assoc" in ndict.keys() }
    """
    word_2_concm:
        - 184667 words with concretenss (conc-m (mean))
    wordpair_2_assoc:
        - 1996 word pairs
        - (might be reversed)

    """


    # TOP AND BOTTOM CONCRETNESS WORDS FOR MT40K
    coll = collections.Counter(word_2_mt40k_conc)
    top = coll.most_common(k)
    bot = coll.most_common()[-k:]
    top_seq = [tup[0] for tup in top]
    bot_seq = [str(tup[0]) for tup in bot]
    #top_string = " ".join(top_seq)
    #bot_string = " ".join(bot_seq)


    # PREPARE THE BERT MODEL AND TOKENINSER
    #import ipdb; ipdb.set_trace()
    ## Full BERT
    if False:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=True)
        with torch.no_grad():
            model = BertModel.from_pretrained('bert-base-uncased', config=config).to(0)
    else:
        # Smaller, Albert
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v1')
        config = AlbertConfig.from_pretrained('albert-base-v1', output_hidden_states=True, output_attentions=True)
        with torch.no_grad():
            model = AlbertModel.from_pretrained('albert-base-v1', config=config).to(0)

    # Run
    #top_seq = ["hello me", "never forget"] TESTING
    #bot_seq = ["the horror", "of lake karachay"]
    analyse_sequences(model, sequences=top_seq, max_seq_len=5, tokenizer=tokenizer, plot_title="BERT: Distribution of Minimal Softmax Logits to 0.9 (Highly Concrete Tokens)", save_path=os.path.join( os.path.dirname(os.path.dirname(__file__)) , "plots_n_stats/BERT",  f"test{k}.png" ), threshold=0.9, mode="mean", device=0)
    analyse_sequences(model, sequences=bot_seq, max_seq_len=5, tokenizer=tokenizer, plot_title="BERT: Distribution of Minimal Softmax Logits to 0.9 (Highly Abstract Tokens)", save_path=os.path.join( os.path.dirname(os.path.dirname(__file__)) , "plots_n_stats/BERT", f"testlow{k}.png" ), threshold=0.9, mode="mean", device=0)




if __name__ == "__main__":
    # Run the top and bottom 3000 concreteness words in MT40k
    topkbottomk_mt40k(3000)


