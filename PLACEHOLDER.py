import os, sys
import argparse
from tqdm import tqdm as tqdm
from nltk.corpus import stopwords
import spacy

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
def word_to_stats(word):
    if word in english_stopwords:
        stat = "STOPWORD"
    else:
        stat = norm_dict.words.get(word, {'conc-m':{"avg":"NO_NORM"}}).get("conc-m",{"avg":"NO_NORM"})["avg"]  # Neatly handle words that have no norm
    return stat

def line_to_stats(line, eos=False, downcase=True):
    eos_word = "<eos>"
    words = line.lower().split() if downcase else line.split()
    # !!!! remove comma here, since they are too many of them
    words = [w for w in words if w != ","]
    words = words + [eos_word] if eos else words
    stats = [word_to_stats(word) for word in words]
    assert(len(words) == len(stats))
    return words, stats

def conc_stats(sentence):
    stats = {
        "has_norm"      :len([ word for word in sentence if type(word) in [float, int]]),
        "hasnt_norm"    :sentence.count("NO_NORM"),
        "length"        :len(sentence),
        "n_stopwords"   :sentence.count("STOPWORD")
    }
    sentence = [word for word in sentence if word not in ['STOPWORD', "NO_NORM"]]
    sentence_no0 = [word for word in sentence if word not in ["STOPWORD", "NO_NORM", 0]]
    stats["conc==0"]        = sentence.count(0)
    stats["0<conc<=0.2"]    = len([score for score in sentence if (0.2>=score) and (score>0)])
    stats["0.2<conc<=0.4"]  = len([score for score in sentence if (0.4>=score) and (score>0.2)])
    stats["0.4<conc<=0.6"]  = len([score for score in sentence if (0.6>=score) and (score>0.4)])
    stats["0.6<conc<=0.8"]  = len([score for score in sentence if (0.8>=score) and (score>0.6)])
    stats["0.8<conc<=1"]  = len([score for score in sentence if (1>=score) and (score>0.8)])
    stats["1<conc"]  = len([score for score in sentence if score>1])
    if len(sentence) == 0:
        stats["conc_density"] = None
    else:
        stats["conc_density"] = sum(sentence)/float(len(sentence))

    return stats


def print_markdown_table(norm_name, norm_stats):
    """
    Print the norm statistics in the syntax of markdown tables for EASY copy-pasting
    """
    length = norm_stats["length"]
    print(
    f"{norm_name} Range | % of Vocab\n\
:-- | --:\n\
None    | {(100*(norm_stats['n_stopwords']+norm_stats['hasnt_norm'])/length):.2f}%\n\
0       | {(100*norm_stats['conc==0']/length):.2f}%\n\
0-0.2   | {(100*norm_stats['0<conc<=0.2']/length):.2f}%\n\
0.2-0.4 | {(100*norm_stats['0.2<conc<=0.4']/length):.2f}%\n\
0.4-0.6 | {(100*norm_stats['0.4<conc<=0.6']/length):.2f}%\n\
0.6-0.8 | {(100*norm_stats['0.6<conc<=0.8']/length):.2f}%\n\
0.8-1 | {(100*norm_stats['0.8<conc<=1']/length):.2f}%\n\
1+ | {(100*norm_stats['1<conc']/length):.2f}%")


################
# Run Functions
################
def avsd(args):
    avsd_path = os.path.join( os.path.dirname(__file__), "data/AVSD" )
    #avsd_train = os.path.join( avsd_path, "qa_and_options", "avsd_train.json" )
    #avsd_val = os.path.join( avsd_path, "qa_and_options", "avsd_val.json" )
    avsd_train_options = os.path.join( avsd_path, "qa_and_options", "train_options.json" )
    avsd_val_options = os.path.join( avsd_path, "qa_and_options", "val_options.json" )

    #avsd_train = myutils.load_json(avsd_train)
    #avsd_val = myutils.load_json(avsd_val)
    avsd_train_options = myutils.load_json(avsd_train_options)
    avsd_val_options = myutils.load_json(avsd_val_options)

    questions = avsd_val_options["data"]["questions"] + avsd_train_options["data"]["questions"]
    answers = avsd_val_options["data"]["answers"] + avsd_train_options["data"]["answers"]
    dialogs = [ ele["dialog"] for ele in avsd_val_options["data"]["dialogs"] ] + [ ele["dialog"] for ele in avsd_train_options["data"]["dialogs"] ]
    captions = [ ele["caption"] for ele in avsd_val_options["data"]["dialogs"] ] + [ ele["caption"] for ele in avsd_train_options["data"]["dialogs"] ]
    
    question_vocab = []
    answer_vocab = []
    dialog_vocab = []
    captions_vocab = []
    total_vocab = question_vocab+answer_vocab+dialog_vocab+captions_vocab

    print(f"Overlap of each vocab")

    
    import ipdb; ipdb.set_trace()
    pass

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
    import ipdb; ipdb.set_trace()
    print(f"PVSE: We have concreteness for {100*len(have_conc_norm)/len(mrw_vocab):.2f}% of vocab")
    _, vocab_conc_stats = line_to_stats(" ".join(mrw_vocab))
    vocab_conc_stats = conc_stats(vocab_conc_stats)
    print_markdown_table("Concreteness", vocab_conc_stats)
    import ipdb; ipdb.set_trace()
    pass

def tvqa(args):
    tvqa_path = os.path.join( os.path.dirname(__file__), "tvqa/tvqa_modality_bias/data")
    tvqa_data = myutils.load_pickle(os.path.join( tvqa_path , "total_dict.pickle" ))
    vcpts = myutils.load_pickle(os.path.join( tvqa_path, "vcpt_features/det_visual_concepts_hq.pickle" ))
    import ipdb; ipdb.set_trace()
    print(f"Processing TVQA Questions...")
    for qidx, qdict in tqdm(enumerate(tvqa_data.values()), total=len(tvqa_data) if args.n_examples==-1 else args.n_examples ):
        if qidx == args.n_examples: break
        a0, a1, a2, a3, a4, answer_idx, q, qid, show_name, ts, vid_name, sub_text, sub_time, located_frame, located_sub_text = qdict.values()
        vcpt = vcpts[vid_name][located_frame[0]:located_frame[1]]
        vcpt = " ".join(list(set([cpt.split()[-1] for line in vcpt for cpt in line.split(" , ") if cpt!="" ])))

        pass
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
