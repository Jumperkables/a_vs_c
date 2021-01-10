__author__ = "Jumperkables"
"""
This file is for analysis of huggingface transformer models, the experiments not unlike those seen in the 'Hopfield Networks is all you need' paper
"""
import os, sys
import math
import matplotlib.pyplot as plt
import collections
import seaborn as sns
import plotly.express as px
import statistics
from tqdm import tqdm
import argparse
import h5py

import torch
from torch import nn
import torch.nn.functional as F

from transformers import BertModel, BertConfig, BertTokenizer, AlbertModel, AlbertConfig, AlbertTokenizer, LxmertTokenizer, LxmertConfig, LxmertModel, LxmertForQuestionAnswering

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
import word_norms
from word_norms import Word2Norm, clean_word
import myutils, dset_utils



########## Flexible functions
def analyse_sequences(args, model, sequences, max_seq_len, tokenizer, plot_title, save_path, threshold=0.9, mode="mean", device=0):
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
    if args.model.split("-")[0] == "lxmert":
        sequences, vid_feats = sequences
    if args.model.split("-")[-1] == "qa":
        #sequences = [ f"{seq.split('@@')[0]} {seq.split('@@')[1]}" for seq in sequences ]
        sequences = [ [f"{seq.split('@@')[0]}", f"{seq.split('@@')[1]}"] for seq in sequences ]
        #import ipdb; ipdb.set_trace()
        indexed_sequences = [ tokenizer(seq[0], seq[1], return_tensors="pt") for seq in sequences ]
        maxxx = 0

        #import ipdb; ipdb.set_trace()
        n_cut = 0
        for sequence in indexed_sequences:
            if len(sequence["token_type_ids"][0]) > max_seq_len:
                import ipdb; ipdb.set_trace()
                n_cut += 1
        assert n_cut == 0, f"{n_cut} sequences were cut, please handle this"
        for seq in indexed_sequences:
            if len(seq["token_type_ids"][0]) > maxxx:
                maxxx = len(seq["token_type_ids"][0])
    else:
        #import ipdb; ipdb.set_trace()
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
        maxxx = 0
        for seq in indexed_sequences:
            if len(seq[0]) > maxxx:
                maxxx = len(seq[0])
    print(maxxx)
    print("This is odd behaviour please change")
    max_seq_len = maxxx
    print(f"{n_cut}/{len(indexed_sequences)} sequences were cropped")
    if args.model.split("-")[0] == "lxmert":
        test_seq = torch.tensor([[101, 102]]).to(device)
        test_feat = torch.ones(1,20,2048).to(device)
        test_pos = torch.ones(1,20,4).to(device)
        if args.model == "lxmert":
            _, _, _, language_attentions, vision_attentions, cross_attentions = model(test_seq, test_feat, test_pos)
            attentions = [cross_attentions, language_attentions, vision_attentions]
        elif args.model == "lxmert-qa":
            # **inputs
            #import ipdb; ipdb.set_trace()
            _, language_attentions, vision_attentions, cross_attentions = model(test_seq, test_feat, test_pos)
            attentions = [cross_attentions, language_attentions, vision_attentions]
    elif args.model in ["default", "albert"]:
        #import ipdb; ipdb.set_trace()
        test_seq = torch.tensor([[101, 102]]).to(device)
        _, _, attentions = model(test_seq)#, test_feat, test_pos)
        attentions = [attentions]
    else:
        raise NotImplementedError(f"Model {args.model} not implemented")
    
    ################
    # Process sequences into attentions
    if args.model.split("-")[0] == "lxmert":
        all_attentions = ["Cross Attentions", "Language Attentions", "Vision Attentions"]
    elif args.model in ["default", "albert"]:
        all_attentions = ["Attentions"]
    else: 
        raise NotImplementedError(f"Support for model {args.model} not implemented")

    layer_head_dims = [ [len(attn), attn[0].shape[1]] for attn in attentions ] # nlayers, nheads
    layer_head_ks = [ [ [ [] for n in range(dims[1]) ] for k in range(dims[0]) ] for dims in layer_head_dims ]

    for idx, seq in tqdm(enumerate(indexed_sequences), desc="Progress", total=len(indexed_sequences)):
        with torch.no_grad():
            seq=seq.to(device)
            if args.model.split("-")[0] == "lxmert":
                dummy_pos = torch.tensor([0,0,639,359]).unsqueeze(0).repeat(min(max_seq_len, len(vid_feats[idx])),1).float().to(device).unsqueeze(0) # Create dummy bounding box the size of TVQA images
                if args.model == "lxmert":
                    _, _, _, language_attentions, vision_attentions, cross_attentions = model(seq, torch.from_numpy(vid_feats[idx][:max_seq_len]).to(device).unsqueeze(0), dummy_pos) # seq, feats, pos
                    language_attentions = [att.cpu() for att in language_attentions]
                    vision_attentions = [att.cpu() for att in vision_attentions]
                    cross_attentions = [att.cpu() for att in cross_attentions]
                    attentions = [cross_attentions, language_attentions, vision_attentions]
                elif args.model == "lxmert-qa":
                    #import ipdb; ipdb.set_trace()
                    inputs = seq
                    #inputs["input_ids"] = inputs["input_ids"][:30]
                    #inputs["token_type_ids"] = inputs["token_type_ids"][:30]
                    #inputs["attention_mask"] = inputs["attention_mask"][:30]
                    inputs["visual_feats"] = torch.from_numpy(vid_feats[idx][:max_seq_len]).to(device).unsqueeze(0)
                    inputs["visual_pos"] = dummy_pos 
                    _, language_attentions, vision_attentions, cross_attentions = model(**inputs) # seq, feats, pos
                    language_attentions = [att.cpu() for att in language_attentions]
                    vision_attentions = [att.cpu() for att in vision_attentions]
                    cross_attentions = [att.cpu() for att in cross_attentions]
                    attentions = [cross_attentions, language_attentions, vision_attentions]
            elif args.model in ["default", "albert"]:
                #import ipdb; ipdb.set_trace()
                _, _, self_attentions = model(seq)
                self_attentions = [att.cpu() for att in self_attentions]
                attentions = [self_attentions]
            else:
                raise NotImplementedError(f"Model {args.model} not implemented")

        for at_idx, attention in enumerate(attentions):
            for layer in range(layer_head_dims[at_idx][0]): # nlayers
                for attn_head in range(layer_head_dims[at_idx][1]): # nheads
                    atten = attention[layer][0][attn_head]
                    #print(f"At_idx:{at_idx}, Layer:{layer}, Head:{attn_head}")
                    for attn in atten:
                        layer_head_ks[at_idx][layer][attn_head].append(myutils.n_softmax_threshold(attn, threshold=threshold))
            #import ipdb; ipdb.set_trace()

    ################
    # Plotting time
    for at_idx, layer_head_k in enumerate(layer_head_ks):
        nlayers, nheads = layer_head_dims[at_idx]
        my_dpi = 91
        plt.figure(figsize=(900/my_dpi, 950/my_dpi), dpi=my_dpi)
        for layer in tqdm(range(0,nlayers)):
            for attn_head in range(0,nheads):
                plt.subplot(nlayers,nheads,1+attn_head+(nheads*(layer)))
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

        plt.suptitle(f"{all_attentions[at_idx]} {plot_title}")
        attn_type = all_attentions[at_idx].split()[0]
        temp_path = save_path.split("/")
        temp_path[-1] = f"{attn_type}_{temp_path[-1]}"
        temp_path = "/".join(temp_path)
        plt.savefig(temp_path)
        plt.clf()
    return f"{plot_title} saved at {save_path}"



def bertqa_logits(sequences, model="lxmert-qa", device=0):
    """
    args: model, purpose, device
    """
    if model not in ["lxmert-qa", "bert-qa"]:
        raise ValueError(f"{model} is not a valid model for running {purpose}")
    
    # Get tokeniser and model
    if model == "lxmert-qa":
        # Multimodal BERT
        tokenizer = LxmertTokenizer.from_pretrained('unc-nlp/lxmert-base-uncased')
        config = LxmertConfig.from_pretrained('unc-nlp/lxmert-base-uncased')
        with torch.no_grad():
            model = LxmertForQuestionAnswering.from_pretrained('albert-base-v1', config=config).to(device)
    elif model == "bert-qa":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        config = BertConfig.from_pretrained('bert-base-uncased')
        with torch.no_grad():
            model = BertModel.from_pretrained('bert-base-uncased', config=config).to(device)
    else:
        raise NotImplementedError(f"Support for model {model} not implemented")#

    # Get Conc & Abs Sequences
    import ipdb; ipdb.set_trace()
    print("ahyuk")
    print("gunther")













##############################
######### INFLEXIBLE FUNCTIONS
##############################
def tvqaconcqs(args):
    # Get TVQA datasets
    train_tvqa_dat = myutils.load_json(os.path.abspath(f"{os.path.abspath(__file__)}/../../tvqa/tvqa_modality_bias/data/tvqa_train_processed.json"))
    val_tvqa_dat = myutils.load_json(os.path.abspath(f"{os.path.abspath(__file__)}/../../tvqa/tvqa_modality_bias/data/tvqa_val_processed.json"))
    if args.model.split("-")[0] == "lxmert":
        vid_h5 = h5py.File(os.path.abspath(f"{os.path.abspath(__file__)}/../../tvqa/tvqa_modality_bias/data/imagenet_features/tvqa_imagenet_pool5_hq.h5"), "r", driver=None)
    q_n_correcta = [{"q":qdict["q"], "cans":qdict[f"a{qdict['answer_idx']}".lower()], "vid_name":qdict["vid_name"], "located_frame":qdict["located_frame"]} for qdict in train_tvqa_dat+val_tvqa_dat]
    # Get norm dictionary
    norm_dict_path =   os.path.abspath(f"{os.path.abspath(__file__)}/../../misc/all_norms.pickle")
    norm_dict = myutils.load_pickle(norm_dict_path)

    # Filter questions with answers that are of certain concreteness
    conc_ans = []
    abs_ans = []
    qs_w_conc_ans = [] 
    qs_w_abs_ans = []
    print(f"Collecting TVQA Qs and As of certain concretness")
    for qa in tqdm(q_n_correcta, total=len(q_n_correcta) ):
        q,a  = qa["q"], qa["cans"]  # cans (correct answer)
        try:    # Speedily developing this code, comeback later to replace with .get
            ans_conc = norm_dict.words[a]["conc-m"]["sources"]["MT40k"]["scaled"]
            if ans_conc < 0.3:
                abs_ans.append(a)
                qs_w_abs_ans.append(qa)
            elif ans_conc > 0.95:
                conc_ans.append(a)
                qs_w_conc_ans.append(qa)
            else:
                pass
        except KeyError:
            pass
    print(f"Abstract answers:{len(qs_w_abs_ans)}, Concrete Answers:{len(qs_w_conc_ans)}")
    #import ipdb; ipdb.set_trace()
    unique_conc = list(set(conc_ans))
    unique_abs = list(set(abs_ans))
    # 150 of each
    #unique_conc = {ans:conc_ans.count(ans) for ans in conc_ans}
    #unique_abs = {ans:abs_ans.count(ans) for ans in abs_ans}
    #answers = unique_conc+unique_abs
    if args.comp_pool == "abstract":
        answers = unique_abs
    if args.comp_pool == "concrete":
        answers = unique_conc
    #import ipdb; ipdb.set_trace()
    answers = " ".join(answers)
    # Just process abstract and concrete questions with their answers appended
    if args.model.split("-")[-1] == "qa": # Process the question answer sequence differently
        #abs_seqs = ["@@".join([qa["cans"], qa["q"]]) for qa in qs_w_abs_ans]
        #conc_seqs = ["@@".join([qa["cans"], qa["q"]]) for qa in qs_w_conc_ans]
        abs_seqs = ["@@".join( [qa["q"], answers]) for qa in qs_w_abs_ans]
        conc_seqs = ["@@".join( [qa["q"], answers]) for qa in qs_w_conc_ans]
    else:
        #import ipdb; ipdb.set_trace()
        abs_seqs = [" ".join([qa["cans"], answers]) for qa in qs_w_abs_ans]
        conc_seqs = [" ".join([qa["cans"], answers]) for qa in qs_w_conc_ans]
    if args.model.split("-")[0] == "lxmert":    # Load visual features
        abs_imgnt = [ vid_h5[qa["vid_name"]][qa["located_frame"][0]:qa["located_frame"][1]] for qa in qs_w_abs_ans ]
        conc_imgnt = [ vid_h5[qa["vid_name"]][qa["located_frame"][0]:qa["located_frame"][1]] for qa in qs_w_conc_ans ]
        conc_seqs = (conc_seqs, conc_imgnt)
        abs_seqs = (abs_seqs, abs_imgnt)

    # Get tokenisers and model
    if args.model == "default":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        config = BertConfig.from_pretrained('bert-base-uncased', output_attentions=True)#, add_cross_attention=True)  # For some reason here you specify cross attentions specifically and not for the other models o.O
        with torch.no_grad():
            model = BertModel.from_pretrained('bert-base-uncased', config=config).to(args.device)
    elif args.model == "albert":
        # Smaller, Albert
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v1')
        config = AlbertConfig.from_pretrained('albert-base-v1', output_hidden_states=True, output_attentions=True)#, add_cross_attention=True)  # For some reason here you specify cross attentions specifically and not for the other models o.O
        with torch.no_grad():
            model = AlbertModel.from_pretrained('albert-base-v1', config=config).to(args.device)
    elif args.model.split("-")[0] == "lxmert":
        # Multimodal BERT
        tokenizer = LxmertTokenizer.from_pretrained('unc-nlp/lxmert-base-uncased')
        config = LxmertConfig.from_pretrained('unc-nlp/lxmert-base-uncased', output_attentions=True)
        if args.model == "lxmert":
            with torch.no_grad():
                model = LxmertModel.from_pretrained('albert-base-v1', config=config).to(args.device)
        if args.model == "lxmert-qa":
            with torch.no_grad():
                model = LxmertForQuestionAnswering.from_pretrained('albert-base-v1', config=config).to(args.device)
    else:
        raise NotImplementedError(f"Support for model {args.model} not implemented")

    # Analyse the sequences

    analyse_sequences(args, model, conc_seqs, args.max_seq_len, tokenizer, args.plot_title, f"{args.plot_save_path.split('.png')[0]}concseqs.png", threshold=args.threshold, mode=args.threshold_mode, device=args.device)
    analyse_sequences(args, model, abs_seqs, args.max_seq_len, tokenizer, args.plot_title, f"{args.plot_save_path.split('.png')[0]}absseqs.png", threshold=args.threshold, mode=args.threshold_mode, device=args.device)




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
    raise NotImplementedError(f"Tidy this up with args.model to distinguish which BERT model to use")
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
    parser = argparse.ArgumentParser()
    # Which datasets
    parser.add_argument_group("Main running arguments")
    parser.add_argument("--purpose", type=str, default=None, choices=["tvqaconcqs", "bottopmt40k", "tvqa_smarter_concqs", "bertqa_logits"], help="What functionality to demand from this script")
    parser.add_argument("--model", type=str, default=None, choices=["default", "albert", "lxmert", "lxmert-qa"], help="What functionality to demand from this script")
    parser.add_argument("--device", type=int, default=-1, help="run on GPU or CPU")

    parser.add_argument_group("tvqaconcqs arguments")
    parser.add_argument("--comp_pool", type=str, default="abstract", help="Code for the 'comparison pool' of words used with respect to the softmax response BERT study")
    parser.add_argument("--max_seq_len", type=int, help="the max sequence length of input text tolerated")
    parser.add_argument("--plot_title", type=str, help="Title for matplotlib figure")
    parser.add_argument("--plot_save_path", type=str, help="The save destination of said plot")
    parser.add_argument("--threshold", type=float, default=0.9, help="The max threshold for the softmax study")
    parser.add_argument("--threshold_mode", type=str, default="mean", help="Which statistic, mean, mode or median, to display on the plot")

    parser.add_argument_group("Flexible Arguments")
    parser.add_argument("--dataset", type=str, choices=["TVQA", "PVSE", "AVSD"], help="Which dataset to load from")

    args = parser.parse_args()
    myutils.print_args(args)

    if args.device == -1:
        args.device = "cpu"
    if args.purpose == None:
        raise NotImplementedError(f"Purpose of script: {args.purpose} isnt accounted for")
    elif args.purpose == "bottopmt40k":
        topkbottomk_mt40k(3000)
    elif args.purpose == "tvqaconcqs":
        tvqaconcqs(args)
    elif args.purpose == "bertqa_logits":
        if args.dataset == "TVQA":
            sequences = dset_utils.load_tvqa_at_norm_threshold(norm="conc-m", norm_threshold=0.95, greater_than=True, include_vid=True)
        bertqa_logits(sequences, model="lxmert-qa", device=0)
