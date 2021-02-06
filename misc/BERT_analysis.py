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

from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import torch.nn.functional as F

from transformers import BertModel, BertConfig, BertTokenizer, AlbertModel, AlbertConfig, AlbertTokenizer, LxmertTokenizer, LxmertConfig, LxmertModel, LxmertForQuestionAnswering

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from word_norms import Word2Norm, clean_word

import myutils
import dset_utils

from pytorch_lightning.callbacks import ModelCheckpoint

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
    elif args.model == "lxmert+classifier":
        sequences, vid_feats, bboxes = sequences    # Pack the bboxes in too
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
                n_cut += 1
        assert n_cut == 0, f"{n_cut} sequences were cut, please handle this"
        for seq in indexed_sequences:
            if len(seq["token_type_ids"][0]) > maxxx:
                maxxx = len(seq["token_type_ids"][0])
    else:
        #import ipdb; ipdb.set_trace()
        print("Processing sequences")
        for sequence in tqdm(sequences, total=len(sequences)):
            sequence = tokenizer(sequence)["input_ids"]
        indexed_sequences = sequences
        n_cut = 0
        sos_token, eos_token = torch.tensor([101]), torch.tensor([102])
        for idx, sequence in enumerate(indexed_sequences):
            sequence = torch.tensor(tokenizer(sequence)["input_ids"])
            if len(sequence)<=max_seq_len: 
                sequence = sequence.unsqueeze(0)
            else:
                sequence = torch.cat( ( sos_token, sequence[1:max_seq_len], eos_token ) ).unsqueeze(0)
                n_cut+=1
            indexed_sequences[idx] = sequence
        maxxx = 0
        for seq in indexed_sequences:
            if len(seq[0]) > maxxx:
                maxxx = len(seq[0])
    print(maxxx)
    print("This is odd behaviour please change")
    max_seq_len = maxxx
    print(f"{n_cut}/{len(indexed_sequences)} sequences were cropped")
    if args.model[:6] == "lxmert":
        test_seq = torch.tensor([[101, 102]]).to(device)
        test_feat = torch.ones(1,20,2048).to(device)
        test_pos = torch.ones(1,20,4).to(device)
        if args.model in ["lxmert", "lxmert+classifier"]:
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
    if args.model[:6] in ["lxmert","lxmert+classifier"]:
        all_attentions = ["Cross Attentions", "Language Attentions", "Vision Attentions"]
    elif args.model in ["default", "albert"]:
        all_attentions = ["Attentions"]
    else: 
        raise NotImplementedError(f"Support for model {args.model} not implemented")

    layer_head_dims = [ [len(attn), attn[0].shape[1]] for attn in attentions ] # nlayers, nheads
    layer_head_ks = [ [ [ [] for n in range(dims[1]) ] for k in range(dims[0]) ] for dims in layer_head_dims ]

    # Cut the sequence length
    # TODO GQA is very slow, just considering the first 500 questions of each
    indexed_sequences = indexed_sequences[:500]
    ### 
    for idx, seq in tqdm(enumerate(indexed_sequences), desc="Getting attentions", total=len(indexed_sequences)):
        with torch.no_grad():
            seq=seq.to(device)
            if args.model[:6] == "lxmert":
                if args.model == "lxmert+classifier": # GQA Dataset i actually have bboxes for
                    # TODO Generalise this line
                    dummy_pos = bboxes[idx].to(device) 
                else:
                    dummy_pos = torch.tensor([0,0,639,359]).unsqueeze(0).repeat(min(max_seq_len, len(vid_feats[idx])),1).float().to(device).unsqueeze(0) # Create dummy bounding box the size of TVQA images
                if args.model in ["lxmert", "lxmert+classifier"]:
                    _, _, _, language_attentions, vision_attentions, cross_attentions = model(seq, myutils.assert_torch(vid_feats[idx][:max_seq_len]).to(device).unsqueeze(0), dummy_pos) # seq, feats, pos
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
                    inputs["visual_feats"] = myutils.assert_torch(vid_feats[idx][:max_seq_len]).to(device).unsqueeze(0)
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



def bertqa_logits(sequences, plot_title, plot_save_path, softmax_threshold=0.9, model_name="lxmert-qa", device=0):
    """
    args: model, purpose, device
    """
    model, tokenizer = get_model_tokenizer(model_name, device, model_path=None)
    softy = nn.Softmax()
    # Get Conc & Abs Sequences
    if model_name == "lxmert-qa":
        sequences, vid_sequences = sequences
    
    # Iterate through sequences and get logits
    logits = []
    for idx in tqdm(range(len(sequences)), total=len(sequences)):
        if model_name == "lxmert-qa":
            text, vid = sequences[idx], myutils.assert_torch(vid_sequences[idx]).to(device)
            q,a = text.split("@@")
            q = torch.Tensor(tokenizer.encode(tokenizer.tokenize(q))).long().to(device)
            dummy_pos = torch.tensor([0,0,639,359]).unsqueeze(0).repeat(len(vid), 1).float().to(device).unsqueeze(0) # Create dummy bounding box the size of TVQA images
            out = model(q.unsqueeze(0),vid,dummy_pos)[0]
            out = out.cpu().detach()[0]
            out = softy(out).numpy()
            a_count = len(out)
            sftmx_thrshld = myutils.n_softmax_threshold(out, threshold=softmax_threshold)
            logits.append(sftmx_thrshld)
        elif model_name == "bert-qa":
            text = sequences[idx]
            q,a = text.split("@@")
            q = torch.Tensor(tokenizer.encode(tokenizer.tokenize(q))).long().to(device)
            out = model(q.unsqueeze(0))[0]
            out = out.cpu().detach()[0]
            out = softy(out).numpy()
            a_count = len(out)
            sftmx_thrshld = myutils.n_softmax_threshold(out, threshold=softmax_threshold)
            logits.append(sftmx_thrshld)

        pass
    # Violin Plot
    violin = myutils.colour_violin(logits, mode="median", max_x=a_count)    
    plt.suptitle(plot_title)
    plt.savefig(plot_save_path)
    plt.clf()
    print(f"{plot_title} saved at {plot_save_path}")



def get_model_tokenizer(model, device, model_path):
    # Get tokenisers and model
    if model == "default":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        config = BertConfig.from_pretrained('bert-base-uncased', output_attentions=True)#, add_cross_attention=True)  # For some reason here you specify cross attentions specifically and not for the other models o.O
        with torch.no_grad():
            model = BertModel.from_pretrained('bert-base-uncased', config=config).to(device)
    elif model == "albert":
        # Smaller, Albert
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v1')
        config = AlbertConfig.from_pretrained('albert-base-v1', output_hidden_states=True, output_attentions=True)#, add_cross_attention=True)  # For some reason here you specify cross attentions specifically and not for the other models o.O
        with torch.no_grad():
            model = AlbertModel.from_pretrained('albert-base-v1', config=config).to(device)
    elif model.split("-")[0] == "lxmert":
        # Multimodal BERT
        tokenizer = LxmertTokenizer.from_pretrained('unc-nlp/lxmert-base-uncased')
        config = LxmertConfig.from_pretrained('unc-nlp/lxmert-base-uncased', output_attentions=True)
        if model == "lxmert":
            with torch.no_grad():
                model = LxmertModel.from_pretrained('albert-base-v1', config=config).to(device)
        if model == "lxmert-qa":
            with torch.no_grad():
                model = LxmertForQuestionAnswering.from_pretrained('albert-base-v1', config=config).to(device)
    elif model == "lxmert+classifier":
        # My lxmert model
        tokenizer = LxmertTokenizer.from_pretrained('unc-nlp/lxmert-base-uncased')
        config = LxmertConfig.from_pretrained('unc-nlp/lxmert-base-uncased', output_attentions=True)
        lx = LxmertModel.from_pretrained('albert-base-v1', config=config)#.to(device)
        lx_dict = lx.state_dict()
        chkpt = torch.load(model_path)['state_dict']
        new_dict = {k:v for k,v in chkpt.items() if k in lx_dict}
        lx_dict.update(new_dict)
        lx.load_state_dict(lx_dict) 
        model = lx
        model.to(device)
        model.eval()
    else:
        raise NotImplementedError(f"Support for model {model} not implemented")
    return model, tokenizer


def get_dset_at_norm_threshold(dataset, norm, norm_threshold, greater_than, norm_dict, vqa_condition=None):
    from VQA_dsets import GQA, VQA, vqa_dummy_args # TODO Circular dependency
    if dataset == "TVQA":
        sequences = dset_utils.load_tvqa_at_norm_threshold(norm=norm, norm_threshold=norm_threshold, greater_than=greater_than, include_vid=True, unique_ans=False)
    elif dataset in ["VQACP","VQACP2","GQA"]:
        if dataset == "GQA":
            #train_dset = GQA(args=None, split="train", images=False, spatial=False, objects=True, obj_names=False, n_objs=10, max_q_len=30)
            valid_dset = GQA(args=None, split="valid", objects=True, n_objs=10, max_q_len=30)
            ans2idx = myutils.load_pickle(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "data/gqa", "ans2idx.pickle"))
        elif dataset == "VQACP":
            # VQA answer scheme is inferred from the name of the model checkpoint
            assert vqa_condition != None, f"VQA answer condition must be specified. e.g. topk-500, or mao-8. See VQA dataset object args"
            topk,mao = -1,-1
            vqa_condition, condition_k = vqa_condition.split("-")
            condition_k = int(condition_k)
            if vqa_condition == "topk":
                topk = condition_k
            else:
                mao = condition_k
            dummy_args = vqa_dummy_args(topk, mao)
            train_dset = VQA(args=dummy_args, version="cp-v1", split="train", objects=True, n_objs=10, max_q_len=30)
            valid_dset = VQA(args=dummy_args, version="cp-v1", split="test", objects=True, n_objs=10, max_q_len=30)
            ans2idx = valid_dset.ans2idx
        elif dataset == "VQACP2":
            topk,mao = -1,-1
            vqa_condition, condition_k = vqa_condition.split("-")
            condition_k = int(condition_k)
            if vqa_condition == "topk":
                topk = condition_k
            else:
                mao = condition_k
            dummy_args = vqa_dummy_args(topk, mao)
            train_dset = VQA(dummy_args, version="cp-v2", split="train", objects=True, n_objs=10, max_q_len=30)
            valid_dset = VQA(dummy_args, version="cp-v2", split="test", objects=True, n_objs=10, max_q_len=30)
            ans2idx = valid_dset.ans2idx
        else:
            raise(NotImplementedError(f"Dataset: {dataset} not implemented")) 
            
        #train_dset = DataLoader(train_dset, batch_size=1)
        valid_dset = DataLoader(valid_dset, batch_size=1)        
        idx2ans = { v:k for k,v in ans2idx.items()}
        if dataset in ["VQACP","VQACP2"]:
            # We have currently initiated the unknown token for VQACP
            idx2ans[len(ans2idx)] = "None"
        tokeniser = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        unique_answers = []
        questions = []
        lobjects = []
        lbboxes = []
        for idx, batch in tqdm(enumerate(valid_dset), total=len(valid_dset)):
            question, answer, bboxes, features = batch
            bboxes = bboxes[0]
            features = features[0]
            answer = idx2ans[int(answer[0])]
            question = tokeniser.decode(question[0])
            question = question.split()
            try:
                question.remove(["[CLS]"])
            except ValueError:
                pass
            try:
                question.remove(["[SEP]"])
            except ValueError:
                pass
            try:
                question.remove(["[PAD]"])
            except ValueError:
                pass
            question = " ".join(question)
            try:    # Speedily developing this code, comeback later to replace with .get
                if norm == "conc-m":
                    ans_norm = norm_dict.words[answer]["conc-m"]["sources"]["MT40k"]["scaled"]
                    if greater_than:
                        if ans_norm > norm_threshold:
                            unique_answers.append(answer)
                            questions.append(question)
                            lobjects.append(features)
                            lbboxes.append(bboxes)
                    else:
                        if ans_norm < norm_threshold:
                            unique_answers.append(answer)
                            questions.append(question)
                            lobjects.append(features)
                            lbboxes.append(bboxes)
            except KeyError:
                pass
        n_uanswers = len(list(set(unique_answers)))
        print(f"Number of unique answers: {n_uanswers}.")
        if n_uanswers > 512:
            raise(f"You should make sure that the number of unique answers does not let any sequence length go above 512. This is the proper upper limit to huggingface transformers")
        elif n_uanswers > 490:
            print("Careful. Getting close to the 512 upper limit. Sequences may be cut")
        unique_answers = " ".join(list(set(unique_answers)))
        questions = [f"{q} {unique_answers}" for q in questions]
        norm_seqs = (questions, lobjects, lbboxes)
        return norm_seqs






def normqs(args):
    # Create the norm dictionary
    norm_dict_path = os.path.join( os.path.dirname(os.path.dirname(__file__)) , "misc", "all_norms.pickle")
    norm_dict = myutils.load_pickle(norm_dict_path)

    from VQA_dsets import GQA_AvsC # TODO Circular dependency

    # Get dataset
    ## VQA-CP dataset has ambiguous answer scheme. 
    ## We deduce how the given model used answers from the checkpoint model path
    vqa_condition = None
    if args.dataset in ["VQACP","VQACP2"]:
        condition = args.model_path.split("/")[-1]
        if "topk-" in condition: # looks like: "vqacp.._topk-500_...ckpt"
            condition = condition.split("topk-")[1]
            condition = condition.split("_")[0] # Get the number k
            condition = f"topk-{condition}"
        elif  "mao-" in condition: # looks like: "vqacp.._mao-3_...ckpt"
            condition = condition.split("mao-")[1]
            condition = condition.split("_")[0] # Get the number for minimum answer occurence (mao)
            condition = f"mao-{condition}"
        else:
            raise ValueError(f"VQA-CP answer scheme cannot be inferred from that model checkpoint name {condition}")

    if args.high_threshold >= 0:   # Allow to ignore
        high_seqs = get_dset_at_norm_threshold(args.dataset, args.norm, args.high_threshold, greater_than=True, norm_dict=norm_dict, vqa_condition=condition)
    if args.low_threshold >= 0:    # Allow to ignore
        low_seqs = get_dset_at_norm_threshold(args.dataset, args.norm, args.low_threshold, greater_than=False, norm_dict=norm_dict, vqa_condition=condition)

    model, tokenizer = get_model_tokenizer(args.model, args.device, args.model_path)

    if args.high_threshold >= 0:   # Allow to ignore
        analyse_sequences(args, model, high_seqs, args.max_seq_len, tokenizer, args.plot_title, f"{args.plot_save_path.split('.png')[0]}high{args.norm}.png", threshold=args.softmax_threshold, mode=args.threshold_mode, device=args.device)
    if args.low_threshold >= 0:   # Allow to ignore
        analyse_sequences(args, model, low_seqs, args.max_seq_len, tokenizer, args.plot_title, f"{args.plot_save_path.split('.png')[0]}low{args.norm}.png", threshold=args.softmax_threshold, mode=args.threshold_mode, device=args.device)






if __name__ == "__main__":

    # Run the top and bottom 3000 concreteness words in MT40k
    parser = argparse.ArgumentParser()
    # Which datasets
    parser.add_argument_group("Main running arguments")
    parser.add_argument("--purpose", type=str, default=None, choices=["normqs", "bottopmt40k", "bertqa_logits"], help="What functionality to demand from this script")
    parser.add_argument("--model", type=str, default=None, choices=["default", "albert", "lxmert", "lxmert-qa", "bert-qa", "lxmert+classifier"], help="What functionality to demand from this script")
    parser.add_argument("--model_path", type=str, default=None, help="If you want to load a pretrained model")
    parser.add_argument("--device", type=int, default=-1, help="run on GPU or CPU")

    parser.add_argument_group("normqs arguments")
    parser.add_argument("--norm", type=str, default="conc-m", help="Which norm to read from the norm dictionary")
    parser.add_argument("--comp_pool", type=str, default="abstract", help="Code for the 'comparison pool' of words used with respect to the softmax response BERT study")
    parser.add_argument("--max_seq_len", type=int, help="the max sequence length of input text tolerated")
    parser.add_argument("--plot_title", type=str, help="Title for matplotlib figure")
    parser.add_argument("--plot_save_path", type=str, help="The save destination of said plot")
    parser.add_argument("--high_threshold", type=float, default=-1, help="The max threshold for the norm")
    parser.add_argument("--low_threshold", type=float, default=-1, help="The min threshold for the norm")
    parser.add_argument("--softmax_threshold", type=float, default=0.9, help="The threshold at which the softmax cuts off")
    parser.add_argument("--threshold_mode", type=str, default="mean", help="Which statistic, mean, mode or median, to display on the plot")

    parser.add_argument_group("Flexible Arguments")
    parser.add_argument("--dataset", type=str, choices=["TVQA", "PVSE", "AVSD","GQA","VQACP","VQACP2"], help="Which dataset to load from")
    args = parser.parse_args()
    myutils.print_args(args)

    if args.device == -1:
        args.device = "cpu"
    if args.purpose == None:
        raise NotImplementedError(f"Purpose of script: {args.purpose} isnt accounted for")
    elif args.purpose == "bottopmt40k":
        topkbottomk_mt40k(3000)
    elif args.purpose == "normqs":
        normqs(args)
    elif args.purpose == "bertqa_logits":
        if args.dataset == "TVQA":
            sequences = dset_utils.load_tvqa_at_norm_threshold(norm="conc-m", norm_threshold=0.95, greater_than=True, include_vid=True, unique_ans=False)
        plot_title = args.plot_title.replace("@@",f"concgt0pt95_softmax{args.threshold}")
        plot_save_path = args.plot_save_path.replace("@@",f"concgt0pt95_softmax{args.threshold}")
        bertqa_logits(sequences, softmax_threshold=args.threshold, plot_title=plot_title, plot_save_path=plot_save_path, model_name="lxmert-qa", device=0)

        if args.dataset == "TVQA":
            sequences = dset_utils.load_tvqa_at_norm_threshold(norm="conc-m", norm_threshold=0.3, greater_than=False, include_vid=True, unique_ans=False)
        plot_title = args.plot_title.replace("@@",f"conclt0pt3_softmax{args.threshold}")
        plot_save_path = args.plot_save_path.replace("@@",f"conclt0pt3_softmax{args.threshold}")
        bertqa_logits(sequences, softmax_threshold=args.threshold, plot_title=plot_title, plot_save_path=plot_save_path, model_name="lxmert-qa", device=0)



#def topkbottomk_mt40k(k=3000):
#    # Create the norm dictionary
#    norm_dict_path = os.path.join( os.path.dirname(os.path.dirname(__file__)) , "misc", "all_norms.pickle")
#    norm_dict = myutils.load_pickle(norm_dict_path)
#    word_2_concm = { word:ndict["conc-m"] for word, ndict in norm_dict.words.items() if "conc-m" in ndict.keys()}
#    word_2_mt40k_conc = { word:concm["sources"]["MT40k"]["scaled"] for word,concm in word_2_concm.items() if "MT40k" in concm["sources"].keys() } 
#    #wordpair_2_assoc = { wordpair:ndict["assoc"] for wordpair, ndict in norm_dict.word_pairs.items() if "assoc" in ndict.keys() }
#    """
#    word_2_concm:
#        - 184667 words with concretenss (conc-m (mean))
#    wordpair_2_assoc:
#        - 1996 word pairs
#        - (might be reversed)
#
#    """
#
#    # TOP AND BOTTOM CONCRETNESS WORDS FOR MT40K
#    coll = collections.Counter(word_2_mt40k_conc)
#    top = coll.most_common(k)
#    bot = coll.most_common()[-k:]
#    top_seq = [tup[0] for tup in top]
#    bot_seq = [str(tup[0]) for tup in bot]
#    #top_string = " ".join(top_seq)
#    #bot_string = " ".join(bot_seq)
#
#
#    # PREPARE THE BERT MODEL AND TOKENINSER
#    #import ipdb; ipdb.set_trace()
#    ## Full BERT
#    raise NotImplementedError(f"Tidy this up with args.model to distinguish which BERT model to use")
#    if False:
#        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#        config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=True)
#        with torch.no_grad():
#            model = BertModel.from_pretrained('bert-base-uncased', config=config).to(0)
#    else:
#        # Smaller, Albert
#        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v1')
#        config = AlbertConfig.from_pretrained('albert-base-v1', output_hidden_states=True, output_attentions=True)
#        with torch.no_grad():
#            model = AlbertModel.from_pretrained('albert-base-v1', config=config).to(0)
#
#    # Run
#    #top_seq = ["hello me", "never forget"] TESTING
#    #bot_seq = ["the horror", "of lake karachay"]
#    analyse_sequences(model, sequences=top_seq, max_seq_len=5, tokenizer=tokenizer, plot_title="BERT: Distribution of Minimal Softmax Logits to 0.9 (Highly Concrete Tokens)", save_path=os.path.join( os.path.dirname(os.path.dirname(__file__)) , "plots_n_stats/BERT",  f"test{k}.png" ), threshold=0.9, mode="mean", device=0)
#    analyse_sequences(model, sequences=bot_seq, max_seq_len=5, tokenizer=tokenizer, plot_title="BERT: Distribution of Minimal Softmax Logits to 0.9 (Highly Abstract Tokens)", save_path=os.path.join( os.path.dirname(os.path.dirname(__file__)) , "plots_n_stats/BERT", f"testlow{k}.png" ), threshold=0.9, mode="mean", device=0)




