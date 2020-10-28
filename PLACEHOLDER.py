import os, sys
import argparse
from tqdm import tqdm as tqdm


import word_norms
from word_norms import Word2Norm

import myutils

norm_dict_path = os.path.join( os.path.dirname(__file__) , "misc", "all_norms.pickle")
norm_dict = myutils.load_pickle(norm_dict_path)
print(f"Norm dict loaded!\nDatasets: {sorted(list(norm_dict.DSETS.keys()))}\nSingle word norms: {len(norm_dict.words)}\nWord Pairs: {len(norm_dict.word_pairs)}")

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

    print(f"Overlap of each vocab")

    
    import ipdb; ipdb.set_trace()
    pass

def pvse(args):
    pvse_path = os.path.join( os.path.dirname(__file__), "data/pvse" )
    mrw_data = myutils.load_json( os.path.join( pvse_path, "mrw/mrw-v1.0.json" ) )
    mrw_sentences = [ ele["sentence"] for ele in mrw_data ]
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
