import os, sys
import argparse
from tqdm import tqdm as tqdm
from nltk.corpus import stopwords
import spacy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import h5py

import word_norms
from word_norms import Word2Norm, clean_word
import myutils

################
# Global Vars
################
norm_dict_path = os.path.join( os.path.dirname(__file__) , "misc", "all_norms.pickle")
norm_dict = myutils.load_pickle(norm_dict_path)
print(f"Norm dict loaded!\nDatasets: {sorted(list(norm_dict.DSETS.keys()))}\nSingle word norms: {len(norm_dict.words)}\nWord Pairs: {len(norm_dict.word_pairs)}")
english_stopwords = list(stopwords.words("english"))
nlp = spacy.load('en')

################
# Util Functions
################
def word_to_stats(word, norm):
    if word in english_stopwords:
        stat = "STOPWORD"
    else:
        stat = norm_dict.words.get(word, {norm:{"avg":"NO_NORM"}}).get(norm,{"avg":"NO_NORM"})["avg"]  # Neatly handle words that have no norm
    return stat

def line_to_stats(line, norm, eos=False, downcase=True):
    eos_word = "<eos>"
    words = line.lower().split() if downcase else line.split()
    # !!!! remove comma here, since they are too many of them
    words = [w for w in words if w != ","]
    words = words + [eos_word] if eos else words
    stats = [word_to_stats(word, norm) for word in words]
    assert(len(words) == len(stats))
    return words, stats

def norm_stats(sentence, norm):
    stats = {
        "norm":norm,
        "has_norm"      :len([ word for word in sentence if type(word) in [float, int]]),
        "hasnt_norm"    :sentence.count("NO_NORM"),
        "length"        :len(sentence),
        "n_stopwords"   :sentence.count("STOPWORD")
    }
    sentence = [word for word in sentence if word not in ['STOPWORD', "NO_NORM"]]
    sentence_no0 = [word for word in sentence if word not in ["STOPWORD", "NO_NORM", 0]]
    stats["norm==0"]        = sentence.count(0)
    stats["0<norm<=0.2"]    = len([score for score in sentence if (0.2>=score) and (score>0)])
    stats["0.2<norm<=0.4"]  = len([score for score in sentence if (0.4>=score) and (score>0.2)])
    stats["0.4<norm<=0.6"]  = len([score for score in sentence if (0.6>=score) and (score>0.4)])
    stats["0.6<norm<=0.8"]  = len([score for score in sentence if (0.8>=score) and (score>0.6)])
    stats["0.8<norm<=1"]  = len([score for score in sentence if (1>=score) and (score>0.8)])
    stats["1<norm"]  = len([score for score in sentence if score>1])
    if len(sentence) == 0:
        stats["norm_density"] = None
    else:
        stats["norm_density"] = sum(sentence)/float(len(sentence))

    return stats

def dset_stats(stat_dicts, norm):
    nones = list(filter(lambda sdict: sdict["norm_density"]==None, stat_dicts))
    stat_dicts = [sdict for sdict in stat_dicts if sdict not in nones]
    stats={
        "norm":norm,
        "length":len(stat_dicts),
        "n_stopwords":  0,    # IGNORE THIS ITS HERE FOR CONVENIENCE WITH OTHER CALLS
        "hasnt_norm":   len(nones),
        "norm==0":      len(list(filter(lambda sdict: sdict["norm_density"]==0 , stat_dicts))),
        "0<norm<=0.2":  len(list(filter(lambda sdict: 0<sdict["norm_density"] and sdict["norm_density"]<=0.2 , stat_dicts))),
        "0.2<norm<=0.4":len(list(filter(lambda sdict: 0.2<sdict["norm_density"]<=0.4 , stat_dicts))),
        "0.4<norm<=0.6":len(list(filter(lambda sdict: 0.4<sdict["norm_density"]<=0.6 , stat_dicts))),
        "0.6<norm<=0.8":len(list(filter(lambda sdict: 0.6<sdict["norm_density"]<=0.8 , stat_dicts))),
        "0.8<norm<=1":  len(list(filter(lambda sdict: 0.8<sdict["norm_density"]<=1 , stat_dicts))),
        "1<norm":       len(list(filter(lambda sdict: 1<sdict["norm_density"] , stat_dicts)))
    }
    return stats


def print_markdown_table(norm_name, norm_stats):
    """
    Print the norm statistics in the syntax of markdown tables for EASY copy-pasting
    """
    length = norm_stats["length"]
    print(
    f"{norm_name} Range | % of CHANGEME\n\
:-- | --:\n\
None    | {(100*(norm_stats['n_stopwords']+norm_stats['hasnt_norm'])/length):.2f}%\n\
0       | {(100*norm_stats['norm==0']/length):.2f}%\n\
0-0.2   | {(100*norm_stats['0<norm<=0.2']/length):.2f}%\n\
0.2-0.4 | {(100*norm_stats['0.2<norm<=0.4']/length):.2f}%\n\
0.4-0.6 | {(100*norm_stats['0.4<norm<=0.6']/length):.2f}%\n\
0.6-0.8 | {(100*norm_stats['0.6<norm<=0.8']/length):.2f}%\n\
0.8-1 | {(100*norm_stats['0.8<norm<=1']/length):.2f}%\n\
1+ | {(100*norm_stats['1<norm']/length):.2f}%")


def normdict2plot(norm_dicts, dict_labels, title="DEFAULT TITLE", xlab="DEFAULT XLAB", ylab="%", save_path="DEFAULT"):
    n_groups = 8
    
    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 1.0/n_groups

    opacity = 0.8

    # Data
    data = [ (
        (100*ndict["hasnt_norm"]+ndict["n_stopwords"])/ndict["length"],
        100*ndict["norm==0"]/ndict["length"],
        100*ndict["0<norm<=0.2"]/ndict["length"],
        100*ndict["0.2<norm<=0.4"]/ndict["length"],
        100*ndict["0.4<norm<=0.6"]/ndict["length"],
        100*ndict["0.6<norm<=0.8"]/ndict["length"],
        100*ndict["0.8<norm<=1"]/ndict["length"],
        100*ndict["1<norm"]/ndict["length"],
    ) for ndict in norm_dicts ]
    #import ipdb; ipdb.set_trace() 
    colours = list(mcolors.TABLEAU_COLORS)[:len(norm_dicts)]
    offsets = [idx-(len(data)//2)+1 for idx in range(len(data))]
    rects1 = [ plt.bar(index+(bar_width*offsets[idx]), datum, bar_width, alpha=opacity, color=colours[idx], label=dict_labels[idx]) for idx, datum in enumerate(data) ]
    
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.xticks(index + bar_width, ('None', '==0', '0-0.2', '0.2-0.4','0.4-0.6','0.6-0.8','0.8-1','1<'))
    plt.legend(bbox_to_anchor=(0, 1.15))#loc=0)
    
    plt.tight_layout()
    #plt.show()
    if save_path == "DEFAULT":
        raise ValueError(f"Set the save_path directory please")
    plt.savefig(save_path)


def vocab2norm_stats(vocab, norm="conc-m"):
    """
    Takes a vocab, and returns the norm_stats
    """
    try:
        vocab.remove("")
    except:
        pass
    have_word = [word for word in vocab if word in norm_dict.words.keys()]
    have_conc_norm = [ word for word in have_word if norm in norm_dict.words[word].keys() ] # Collect all words in vocab we have a concreteness norm for
    havent_conc_norm = [ word for word in vocab if word not in have_conc_norm ]
    # Vocab stats
    print(f"We have concreteness for {100*len(have_conc_norm)/len(vocab):.2f}% of vocab")
    _, vocab_norm_stats = line_to_stats(" ".join(vocab), norm)
    vocab_norm_stats = norm_stats(vocab_norm_stats, norm)
    return vocab_norm_stats



################
# Run Functions
################
def avsd(args):
    avsd_path = os.path.join( os.path.dirname(__file__), "data/AVSD" )
    avsd_train = os.path.join( avsd_path, "qa_and_options", "avsd_train.json" )
    avsd_val = os.path.join( avsd_path, "qa_and_options", "avsd_val.json" )
    avsd_train_options = os.path.join( avsd_path, "qa_and_options", "train_options.json" )
    avsd_val_options = os.path.join( avsd_path, "qa_and_options", "val_options.json" )

    avsd_train = myutils.load_json(avsd_train)
    avsd_val = myutils.load_json(avsd_val)
    avsd_train_options = myutils.load_json(avsd_train_options)
    avsd_val_options = myutils.load_json(avsd_val_options)
    #avsd_dialogs = h5py.File( os.path.join( avsd_path, "features/dialogs.h5" ) , "r") 
    #params = myutils.load_json( os.path.join( avsd_path, "features/params.json" ) )
    #i2w = params["ind2word"]

    questions = avsd_val_options["data"]["questions"] + avsd_train_options["data"]["questions"]
    answers = avsd_val_options["data"]["answers"] + avsd_train_options["data"]["answers"]
    summaries = [ ele["summary"].replace('\n','') for ele in avsd_val.values() if ele["summary"] != None ] + [ ele["summary"].replace('\n','') for ele in avsd_train.values() if ele["summary"] != None ]
    captions = [ ele["caption"] for ele in avsd_val_options["data"]["dialogs"] ] + [ ele["caption"] for ele in avsd_train_options["data"]["dialogs"] ]

    # Vocab
    questions   = [ clean_word(word) for sentence in questions for word in sentence.split() ]
    answers     = [ clean_word(word) for sentence in answers for word in sentence.split() ]
    summaries   = [ clean_word(word) for sentence in summaries for word in sentence.split() ]
    captions    = [ clean_word(word) for sentence in captions for word in sentence.split() ]

    question_vocab = list(set(questions))
    answer_vocab = list(set(answers))
    summary_vocab = list(set(summaries))
    captions_vocab = list(set(captions))
    total_vocab = list(set(question_vocab+answer_vocab+summary_vocab+captions_vocab))
    
    total_conc_stats = vocab2norm_stats(total_vocab, "conc-m")
    question_conc_stats = vocab2norm_stats(question_vocab, "conc-m")
    answer_conc_stats = vocab2norm_stats(answer_vocab, "conc-m")
    summary_conc_stats = vocab2norm_stats(summary_vocab, "conc-m")
    caption_conc_stats = vocab2norm_stats(captions_vocab, "conc-m")

    #import ipdb; ipdb.set_trace()
    # Plotting
    #import ipdb; ipdb.set_trace()
    plot_dicts = [total_conc_stats, question_conc_stats, answer_conc_stats, summary_conc_stats, caption_conc_stats]
    plot_labels = [ "Total Vocab", "Question Vocab", "Answer Vocab", "Summary Vocab", "Caption Vocab"]
    normdict2plot(plot_dicts, plot_labels, title="AVSD Concreteness (Vocab)", xlab="Conc Range", ylab="%", 
            save_path=os.path.join( os.path.dirname(__file__) , "plots_n_stats/all/2_improved_concreteness_distribution/AVSD_conc_vocab.png" ) )

    # Full dataset
    # Plotting
    #import ipdb; ipdb.set_trace()
    question_conc_stats = vocab2norm_stats(questions, "conc-m")
    answer_conc_stats = vocab2norm_stats(answers, "conc-m")
    summary_conc_stats = vocab2norm_stats(summaries, "conc-m")
    caption_conc_stats = vocab2norm_stats(captions, "conc-m")

    plot_dicts = [question_conc_stats, answer_conc_stats, summary_conc_stats, caption_conc_stats]
    plot_labels = ["Question Vocab", "Answer Vocab", "Summary Vocab", "Caption Vocab"]
    normdict2plot(plot_dicts, plot_labels, title="AVSD Concreteness", xlab="Conc Range", ylab="%", 
            save_path=os.path.join( os.path.dirname(__file__) , "plots_n_stats/all/2_improved_concreteness_distribution/AVSD_conc_feat.png" ) )
    print(f"Overlap of each vocab")

    

def pvse(args):
    pvse_path = os.path.join( os.path.dirname(__file__), "data/pvse" )
    mrw_data = myutils.load_json( os.path.join( pvse_path, "mrw/mrw-v1.0.json" ) )
    mrw_sentences = [ ele["sentence"] for ele in mrw_data ]
    mrw_vocab = set([ clean_word(word) for sentence in mrw_sentences for word in sentence.split() ])
    mrw_vocab.remove("")
    have_word = [word for word in mrw_vocab if word in norm_dict.words.keys()]
    have_conc_norm = [ word for word in have_word if "conc-m" in norm_dict.words[word].keys() ] # Collect all words in vocab we have a concreteness norm for
    havent_conc_norm = [ word for word in mrw_vocab if word not in have_conc_norm ]
    # Vocab stats
    print(f"PVSE: We have concreteness for {100*len(have_conc_norm)/len(mrw_vocab):.2f}% of vocab")
    _, vocab_conc_stats = line_to_stats(" ".join(mrw_vocab), "conc-m")
    pvse_vocab_conc_stats = norm_stats(vocab_conc_stats, "conc-m")
    print_markdown_table("Concreteness", pvse_vocab_conc_stats)

    # Sentences
    pvse_sentence_conc_stats = [norm_stats(line_to_stats(sentence, "conc-m")[1], "conc-m") for sentence in mrw_sentences]
    pvse_sentence_conc_stats = dset_stats( pvse_sentence_conc_stats , "conc-m")
    print_markdown_table("Concreteness", pvse_sentence_conc_stats)

    # Plotting
    #import ipdb; ipdb.set_trace()
    plot_dicts = [pvse_vocab_conc_stats, pvse_sentence_conc_stats]
    plot_labels = [ "Vocabulary", "Sentence Density" ]
    normdict2plot(plot_dicts, plot_labels, title="PVSE Concreteness", xlab="Conc Range", ylab="%", 
            save_path=os.path.join( os.path.dirname(__file__) , "plots_n_stats/all/2_improved_concreteness_distribution/PVSE_conc.png" ) )
    print("Fireball")
    pass

def tvqa(args):
    tvqa_path = os.path.join( os.path.dirname(__file__), "tvqa/tvqa_modality_bias/data")
    tvqa_data = myutils.load_pickle(os.path.join( tvqa_path , "total_dict.pickle" ))
    vcpts = myutils.load_pickle(os.path.join( tvqa_path, "vcpt_features/det_visual_concepts_hq.pickle" ))
    #import ipdb; ipdb.set_trace()
    print(f"Processing TVQA Questions...")

    # Generate objects for plots
    questions, answers, correct_answers, vcpts_collect, ts_subtitles, nots_subtitles = [], [], [], [], [], []

    for qidx, qdict in tqdm(enumerate(tvqa_data.values()), total=len(tvqa_data) if args.n_examples==-1 else args.n_examples ):
        #import ipdb; ipdb.set_trace()
        if len(qdict) == 15:
            a0, a1, a2, a3, a4, answer_idx, q, qid, show_name, ts, vid_name, sub_text, sub_time, located_frame, located_sub_text = qdict.values()
        elif len(qdict) == 14:
            a0, a1, a2, a3, a4, q, qid, show_name, ts, vid_name, sub_text, sub_time, located_frame, located_sub_text = qdict.values()
        else:
            raise ValueError("How did this happen?")
        vcpt = vcpts[vid_name][located_frame[0]:located_frame[1]]
        vcpt = " ".join(list(set([cpt.split()[-1] for line in vcpt for cpt in line.split(" , ") if cpt!="" ])))
        questions.append(q)
        ans = [a0,a1,a2,a3,a4]
        answers += ans
        if len(qdict) == 15:
            correct_answers.append(ans[answer_idx])
        vcpts_collect.append(vcpt)
        ts_subtitles.append(located_sub_text)
        nots_subtitles.append(sub_text)   
    ### Features
    questions = [ clean_word(word) for sentence in questions for word in sentence.split() ]
    answers = [ clean_word(word) for sentence in answers for word in sentence.split() ]
    correct_answers = [ clean_word(word) for sentence in correct_answers for word in sentence.split() ]
    vcpts_collect = [ clean_word(word) for sentence in vcpts_collect for word in sentence.split() ]
    ts_subtitles = [ clean_word(word) for sentence in ts_subtitles for word in sentence.split() ]
    nots_subtitles = [ clean_word(word) for sentence in nots_subtitles for word in sentence.split() ]

    questions_conc = vocab2norm_stats(questions, "conc-m")
    answers_conc = vocab2norm_stats(answers, "conc-m")
    correct_answers_conc = vocab2norm_stats(correct_answers, "conc-m")
    vcpts_collect_conc = vocab2norm_stats(vcpts_collect, "conc-m")
    ts_subtitles_conc = vocab2norm_stats(ts_subtitles, "conc-m")
    nots_subtitles_conc = vocab2norm_stats(nots_subtitles, "conc-m")
    # Plots
    plot_dicts = [questions_conc, answers_conc, correct_answers_conc, vcpts_collect_conc, ts_subtitles_conc, nots_subtitles_conc]
    plot_labels = [ "Questions", "Answers", "Correct Answers", "Vcpts", "Subtitles_ts", "Subtitles_no-ts" ]
    normdict2plot(plot_dicts, plot_labels, title="TVQA Concreteness", xlab="Conc Range", ylab="%", 
            save_path=os.path.join( os.path.dirname(__file__) , "plots_n_stats/all/2_improved_concreteness_distribution/TVQA_conc_feat.png" ) )

    ### Vocabs
    questions_vocab = list(set(questions))
    answers_vocab = list(set(answers))
    correct_answers_vocab = list(set(answers))
    vcpts_collect_vocab = list(set(vcpts_collect))
    ts_subtitles_vocab = list(set(ts_subtitles))
    nots_subtitles_vocab = list(set(nots_subtitles))
    total_vocab = list(set( questions_vocab + answers_vocab + correct_answers_vocab + vcpts_collect_vocab + ts_subtitles_vocab + nots_subtitles_vocab ))
    # Plots
    total_vocab_conc = vocab2norm_stats(total_vocab, "conc-m")
    questions_vocab_conc = vocab2norm_stats(questions_vocab, "conc-m")
    answers_vocab_conc = vocab2norm_stats(answers_vocab, "conc-m")
    correct_answers_vocab_conc = vocab2norm_stats(correct_answers_vocab, "conc-m")
    vcpts_collect_vocab_conc = vocab2norm_stats(vcpts_collect_vocab, "conc-m")
    ts_subtitles_vocab_conc = vocab2norm_stats(ts_subtitles_vocab, "conc-m")
    nots_subtitles_vocab_conc = vocab2norm_stats(nots_subtitles_vocab, "conc-m")
    # Plots
    plot_dicts = [total_vocab_conc, questions_vocab_conc, answers_vocab_conc, correct_answers_vocab_conc, vcpts_collect_vocab_conc, ts_subtitles_vocab_conc, nots_subtitles_vocab_conc]
    plot_labels = [ "Total Vocab", "Questions", "Answers", "Correct Answers", "Vcpts", "Subtitles_ts", "Subtitles_no-ts" ]
    normdict2plot(plot_dicts, plot_labels, title="TVQA Vocab Concreteness", xlab="Conc Range", ylab="%", 
            save_path=os.path.join( os.path.dirname(__file__) , "plots_n_stats/all/2_improved_concreteness_distribution/TVQA_conc_vocab.png" ) )


    # For iterating over examples
    #for qidx, qdict in tqdm(enumerate(tvqa_data.values()), total=len(tvqa_data) if args.n_examples==-1 else args.n_examples ):
    #    if qidx == args.n_examples: break
    #    a0, a1, a2, a3, a4, answer_idx, q, qid, show_name, ts, vid_name, sub_text, sub_time, located_frame, located_sub_text = qdict.values()
    #    vcpt = vcpts[vid_name][located_frame[0]:located_frame[1]]
    #    vcpt = " ".join(list(set([cpt.split()[-1] for line in vcpt for cpt in line.split(" , ") if cpt!="" ])))

    import ipdb; ipdb.set_trace()
    print("We're here at least")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument_group("Dataset Options")
    parser.add_argument("--dsets", type=str, nargs="+", choices=["TVQA", "PVSE", "AVSD"], required=True, help="Which dataset(s) to consider")
    parser.add_argument("--n_examples", type=int, default=-1, help="How many examples per dataset to iterate across")
    args = parser.parse_args()
    
    assert ((args.n_examples == -1) or (args.n_examples > 0)), f"n_examples: {args.n_examples} invalid. Leave to iterate across ALL examples. Otherwise set to positive integers"

    if "TVQA" in args.dsets:
        tvqa(args)
    if "PVSE" in args.dsets:
        pvse(args)
    if "AVSD" in args.dsets:
        avsd(args)
