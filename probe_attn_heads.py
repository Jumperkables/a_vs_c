#TODO Sort this script out at some point
import os, sys, argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import LxmertConfig, LxmertForQuestionAnswering, LxmertModel, LxmertTokenizer, BertTokenizer, BertModel, BertConfig
import pytorch_lightning as pl
# Plotting libraries
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
# Local imports
from misc.BERT_analysis import get_model_tokenizer, analyse_sequences, get_dset_at_norm_threshold
import myutils
import VQA_dsets

# UTILS
def get_transformer_from_model(model_name, checkpoint_path):
    if model_name in ["dual-lx-lstm"]:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        state_dict = checkpoint["state_dict"]
        breakpoint()
        low_lxmert = LxmertModel(LxmertConfig())
        high_lxmert = LxmertModel(LxmertConfig())
        low_lxmert_dict = {".".join(key.split(".")[1:]):weights for key, weights in state_dict.items() if "low_lxmert" in key} 
        high_lxmert_dict = {".".join(key.split(".")[1:]):weights for key, weights in state_dict.items() if "high_lxmert" in key}
        low_lxmert.load_state_dict(state_dict=low_lxmert_dict)
        high_lxmert.load_state_dict(state_dict=high_lxmert_dict)
        low_lxmert.eval()
        high_lxmert.eval()
        transformers = {"low_lxmert":low_lxmert, "high_lxmert":high_lxmert}
        tokeniser = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
    return transformers, tokeniser


# Attentiosn sketching functions
def draw_seq_2_seq(ax, left, right, bottom, top, sequences, sequence_attentions, idx, title):
    '''
    Draw a neural network cartoon using matplotilb.
    Borrowed from https://gist.github.com/craffel/2d727968c3aaebd10359 
    :usage:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])
    
    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - sequences : list of strings
            List of text sequences
    '''
    ax.cla()
    ax.axis('off')
    ax.set_title(title)
    cmap = plt.cm.cool 
    assert len(sequences) == 2, f"Between 2 trying to make an animated gif"
    layer_sizes = [len(sequence) for sequence in sequences]
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line_width = 4*sequence_attentions[idx][m][o] if sequence_attentions[idx][m][o] > 0.2 else 0
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                        [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], color=cmap(float(sequence_attentions[idx][m][o])), linewidth=line_width)
                ax.add_line(line)
    # Nodes
    for n, sequence in enumerate(sequences):
        layer_top = v_spacing*(len(sequence) - 1)/2. + (top + bottom)/2.
        ax.text(n*h_spacing + left, layer_top+v_spacing, f"Layer {idx+n+1}", color="r", fontweight="bold")
        for m, word in enumerate(sequence): #was xrange
            #circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,color='w', ec='k', zorder=4, label=word)
            ax.text(n*h_spacing + left, layer_top - m*v_spacing, word, fontweight="bold")

def lang_update(i, *fargs):
    #(ax_abs_vis, attentions, decoded_question)
    draw_seq_2_seq(fargs[0], .1, .9, .1, .9, [fargs[2]]*2, fargs[1], i, fargs[3])

def draw_heatmap(ax, xlabels, ylabels, attentions, idx, title):
    ax.cla()
    #ax.axis('off')
    array = attentions[idx].detach().numpy()
    im = ax.imshow(array)
    #ax.set_title(title)
    # TODO CHECK THIS ORDER ISCORRECT
    ax.set_xlabel(f"Layer {idx+2}")
    ax.set_ylabel(f"Layer {idx+1}")
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    for i in range(len(xlabels)):
        for j in range(len(ylabels)):
            pass
            #text = ax.text(j, i, array[i, j],
            #               ha="center", va="center", color="w")

def heatmap_update(i, *fargs):
    #draw_heatmap(ax, xlabels, ylabels, attentions, idx, title)
    # TODO sort the ordering of these arguments, I did this at midnight
    draw_heatmap(fargs[0], fargs[1], fargs[2], fargs[3], i, fargs[4])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument_group("Running Arguments")
    parser.add_argument("--jobname", type=str, default="default", help="Jobname")
    parser.add_argument("--dataset", type=str, default="VQACP", choices=["VQACP","VQACP2","GQA"], help="Choose VQA dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--bsz", type=int, default=1, help="Training batch size")
    parser.add_argument("--device", type=int, default=-1, help="Which device to run things on. (-1 = CPU)")
    parser.add_argument("--wandb", action="store_true", help="Plot wandb results online")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of pytroch workers. More should increase disk reads, but will increase RAM usage. 0 = main process")
    parser.add_argument("--norm", type=str, default="conc-m", help="The norm to consider in relevant models. (conc-m == mean concreteness)")
    parser.add_argument_group("Model Arguments")
    parser.add_argument("--model", type=str, default="basic", choices=["basic", "induction", "lx-lstm", "bert-lstm", "hpf-0", "hpf-1", "hpf-2", "hpf-3", "dual-lx-lstm"], help="Which model")
    parser.add_argument("--unfreeze", type=str, default="heads", choices=["heads","all","none"], help="What parts of LXMERT to unfreeze")
    parser.add_argument_group("VQA-CP arguments")
    parser.add_argument("--hopfield_beta_high", type=float, default=0.7, help="When running a high-low norm network, this is the beta scaling for the high norm hopfield net")
    parser.add_argument("--hopfield_beta_low", type=float, default=0.3, help="When running a high-low norm network, this is the beta scaling for the low norm hopfield net")
    parser.add_argument("--loss", type=str, default="default", choices=["default","avsc"], help="Whether or not to use a special loss")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument_group("Dataset arguments")
    parser.add_argument("--norm_gt", default="answer", choices=["answer", "nsubj"], help="Where to derive the norm information of the question. 'answer'=consider the concreteness of the answer, 'nsubj'=use the concreteness of the subject of the input question")
    #### VQA-CP must have one of these 2 set to non-default values
    parser.add_argument("--topk", type=int, default=-1, help="Keep the k-top scoring answers. -1 implies ignore")
    parser.add_argument("--min_ans_occ", type=int, default=-1, help="The minimum occurence threshold for keeping an answers. -1 implies ignore")
    args = parser.parse_args()
    myutils.print_args(args)
    # TODO Reuse? model, tokeniser = get_model_tokenizer(model="lxmert", device=device, model_path=None)
    # TODO old running code for analyse_sequences analyse_sequences(model_name="lxmert", model=model, sequences=conc_sequences, max_seq_len=470, tokenizer=tokeniser, plot_title="Standard LXMERT Concrete", save_path=os.path.dirname(__file__), threshold=0.9, mode="mean", device=device)
    #norm_dict = myutils.load_norms_pickle( os.path.join(os.path.dirname(__file__),"misc/all_norms.pickle"))
    #conc_sequences = get_dset_at_norm_threshold(dataset="VQACP", norm="conc-m", norm_threshold=0.98, greater_than=True, norm_dict=norm_dict, vqa_condition=None)

    # Running checks
    assert args.bsz == 1, f"One at a time please"
    vqa_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/vqa")
    if args.dataset in ["VQACP", "VQACP2"]:
        assert (args.topk != -1) or (args.min_ans_occ != -1), f"For VQA-CP v1/2, you must set one of topk or min_ans_occ to not default. This decides which scheme to follow to keep which answers"
        assert not((args.topk != -1) and (args.min_ans_occ != -1)), f"You must leave one of topk, or min_ans_occ at default value"
    # TODO Consider a more elegant way to handle these flags
    assert args.model in ["basic", "induction", "lx-lstm", "bert-lstm", "hpf-0", "hpf-1", "hpf-2", "hpf-3", "dual-lx-lstm"], f"Make sure to account the feature flags for any new model: {args.model} needs considering"

    # Get dataset
    objects_flag = True 
    obj_names_flag = False
    images_flag = False
    resnet_flag = True if args.model in ["hpf-2"] else False
    return_norm = True if args.model in ["induction","hpf-0","hpf-1","hpf-2","hpf-3","dual-lx-lstm"] else False
    return_avsc = True if args.loss in ["avsc"] else False
    if args.dataset == "VQACP":
        train_dset = VQA_dsets.VQA(args, version="cp-v1", split="train", objects=objects_flag, obj_names=obj_names_flag, images=images_flag, resnet=resnet_flag, return_norm=return_norm, return_avsc=return_avsc)
        valid_dset = VQA_dsets.VQA(args, version="cp-v1", split="test", objects=objects_flag, obj_names=obj_names_flag, images=images_flag, resnet=resnet_flag, return_norm=return_norm, return_avsc=return_avsc)
    elif args.dataset == "VQACP2":
        train_dset = VQA_dsets.VQA(args, version="cp-v2", split="train", objects=objects_flag, obj_names=obj_names_flag, images=images_flag, resnet=resnet_flag, return_norm=return_norm, return_avsc=return_avsc)
        valid_dset = VQA_dsets.VQA(args, version="cp-v2", split="test", objects=objects_flag, obj_names=obj_names_flag, images=images_flag, resnet=resnet_flag, return_norm=return_norm, return_avsc=return_avsc)
    elif args.dataset == "GQA":
        train_dset = VQA_dsets.GQA(args, split="train", objects=objects_flag, obj_names=obj_names_flag, images=images_flag, resnet=resnet_flag, return_norm=return_norm, return_avsc=return_avsc)
        valid_dset = VQA_dsets.GQA(args, split="valid", objects=objects_flag, obj_names=obj_names_flag, images=images_flag, resnet=resnet_flag, return_norm=return_norm, return_avsc=return_avsc)
    if args.dataset in ["VQACP","VQACP2"]:
        n_answers = len(train_dset.ans2idx)
    elif args.dataset == "GQA":
        n_answers = 1841    # There are 1842 answers, we pass in 1841 because +1 will be added in model definition (for VQA-CP)
    else:
        raise NotImplementedError(f"{args.dataset} not implemented yet")
    if args.model in ["hpf-2"]:
        train_loader = DataLoader(train_dset, batch_size=args.bsz, num_workers=args.num_workers, drop_last=True, collate_fn=VQA_dsets.pad_question_collate)
        valid_loader = DataLoader(valid_dset, batch_size=args.bsz, num_workers=args.num_workers, drop_last=True, collate_fn=VQA_dsets.pad_question_collate)
    else:
        train_loader = DataLoader(train_dset, batch_size=args.bsz, num_workers=args.num_workers, drop_last=True, collate_fn=VQA_dsets.pad_question_collate)
        valid_loader = DataLoader(valid_dset, batch_size=args.bsz, num_workers=args.num_workers, drop_last=True, collate_fn=VQA_dsets.pad_question_collate)

    # Set device
    if args.device == -1:
        gpus = None
    else:
        gpus = [args.device]    # TODO Implement multigpu support

    # Get LXMERT model and tokeniser
    transformers, tokeniser = get_transformer_from_model(args.model, args.checkpoint_path)
    colours = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:brown","tab:pink","tab:gray","tab:olive","tab:cyan"]
    if args.model == "dual-lx-lstm":
        for idx, batch in enumerate(valid_loader):
            # Batch
            question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens, img_id = batch
            decoded_question = tokeniser.convert_ids_to_tokens(question[0])
            #TODO RESTORE CLS AND SEP FUNCTIONALITY??
            decoded_question = decoded_question[1:-1]
            question = question[:,1:-1]
            
            # Make Plot
            fig = plt.figure()
            ax_conc_lang = plt.subplot2grid((3,10), (0,0), colspan=5) # topleft:conc-language
            ax_abs_lang  = plt.subplot2grid((3,10), (0,5), colspan=5) # topright:abs-language
            ax_conc_vis  = plt.subplot2grid((3,10), (1,0), colspan=3) # topleft:conc-vision
            ax_image     = plt.subplot2grid((3,10), (1,3), colspan=4, rowspan=2) # Image and bboxes
            ax_abs_vis   = plt.subplot2grid((3,10), (1,7), colspan=3) # topright:abs-vision
            ax_conc_x    = plt.subplot2grid((3,10), (2,0), colspan=3) # topleft:conc-cross
            ax_abs_x     = plt.subplot2grid((3,10), (2,7), colspan=3) # topright:abs-cross
            ax_conc_lang.set_axis_off(), ax_abs_lang.set_axis_off()
            ax_conc_vis.set_axis_off(), ax_abs_vis.set_axis_off()
            ax_conc_x.set_axis_off(), ax_abs_x.set_axis_off()
            ax_image.set_axis_off()
            fig.tight_layout()
            # Transformer
            ## CONCRETE
            h_outputs = transformers['high_lxmert'](question, features, bboxes, output_attentions=True)
            _, conc_language_attentions = h_outputs['language_output'], h_outputs['language_attentions']
            vision_output, conc_vision_attentions = h_outputs['vision_output'], h_outputs['vision_attentions']
            _, conc_x_attentions = h_outputs['pooled_output'], h_outputs['cross_encoder_attentions']
            ## ABSTRACT
            l_outputs = transformers['low_lxmert'](question, features, bboxes, output_attentions=True)
            _, abs_language_attentions = l_outputs['language_output'], l_outputs['language_attentions']
            _, abs_vision_attentions = l_outputs['vision_output'], l_outputs['vision_attentions']
            _, abs_x_attentions = l_outputs['pooled_output'], l_outputs['cross_encoder_attentions']
            ## Cross Attentions
            attentions = [torch.mean(attn[0], dim=0) for attn in abs_x_attentions]
            ani_abs_x = matplotlib.animation.FuncAnimation(fig, heatmap_update, fargs=(ax_abs_x, [str(i) for i in range(len(bboxes[0]))], decoded_question, attentions, "Abstract Cross Attns"), frames=np.arange(0,len(attentions)), interval=5000, repeat=True)
            attentions = [torch.mean(attn[0], dim=0) for attn in conc_x_attentions]
            ani_conc_x = matplotlib.animation.FuncAnimation(fig, heatmap_update, fargs=(ax_conc_x, [str(i) for i in range(len(bboxes[0]))], decoded_question, attentions, "Concrete Cross Attns"), frames=np.arange(0,len(attentions)), interval=5000, repeat=True)
            ## Vision attentions
            attentions = [torch.mean(attn[0], dim=0) for attn in abs_vision_attentions]
            ani_abs_vision = matplotlib.animation.FuncAnimation(fig, heatmap_update, fargs=(ax_abs_vis, [str(i) for i in range(len(bboxes[0]))], [str(i) for i in range(len(bboxes[0]))], attentions, "Abstract Vision Attns"), frames=np.arange(0,len(attentions)), interval=5000, repeat=True) 
            attentions = [torch.mean(attn[0], dim=0) for attn in conc_vision_attentions]
            ani_conc_vision = matplotlib.animation.FuncAnimation(fig, heatmap_update, fargs=(ax_conc_vis, [str(i) for i in range(len(bboxes[0]))], [str(i) for i in range(len(bboxes[0]))], attentions, "Concrete Vision Attns"), frames=np.arange(0,len(attentions)), interval=5000, repeat=True)
            ## Lanaguage attentions
            attentions = [torch.mean(attn[0], dim=0) for attn in abs_language_attentions]
            ani_abs_lang = matplotlib.animation.FuncAnimation(fig, lang_update, fargs=(ax_abs_lang, attentions, decoded_question, "Abstract Language Attns"), frames=np.arange(0,len(attentions)), interval=5000, repeat=True)
            attentions = [torch.mean(attn[0], dim=0) for attn in conc_language_attentions]
            ani_conc_lang = matplotlib.animation.FuncAnimation(fig, lang_update, fargs=(ax_conc_lang, attentions, decoded_question, "Concrete Language Attns"), frames=np.arange(0,len(attentions)), interval=5000, repeat=True)

        
            # Plot image & bboxes
            if args.dataset in ["VQACP", "VQACP2"]:
                split = 'train2014' if img_id[0][0] == 0 else 'val2014'
                img_path = os.path.join(os.path.dirname(__file__), "data/vqa/images", f"{split}", f"COCO_{split}_{img_id[0][1]:012}.jpg")
            elif args.dataset in ["GQA"]:
                img_path = os.path.join(os.path.dirname(__file__), "data/gqa/images", f"{'n' if img_id[0][0] == 0 else ''}{img_id[0][1]}.jpg" )
            else:
                raise NotImplementedError(f"Dataset {args.dataset} not implemented yet")
            im = Image.open(img_path)
            width, height = im.size # TODO check if correct
            ax_image.imshow(im)
            # BBOXES
            for idx, bbox in enumerate(bboxes[0]):
                #TODO CHECK IF TRUE bbox[0] = top-left x, bbox[1] = top-left y, bbox[2] = bottom-right x, bbox[3]=bottom-right y
                # patches.Rectangle((bottom-left x, bottom-left-y), width, height)
                #TODO CHECK IF TRUE Conversion is needed
                rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor=colours[idx], facecolor='none', label=f"{idx}") 
                # Add the patch to the Axes
                ax_image.add_patch(rect)
                ax_image.text(bbox[0], bbox[1], str(idx), color=colours[idx], fontweight="bold")
            plt.show()
