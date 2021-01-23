import os, sys
import h5py
import myutils
from tqdm import tqdm
from multimodal.datasets import VQA, VQA2


def download_gqa():
    pip_mm_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/pip_multimodal")
    import time
    print("VQA Dataset not Downloaded. Downloading in 5 seconds...")
    time.sleep(5)
    DOWNLOAD = VQA(
        dir_data=pip_mm_path,
        features="coco-bottomup",
        split="train",
        label="best",
        min_ans_occ=1
    )
    DOWNLOAD = VQA(
        dir_data=pip_mm_path,
        features="coco-bottomup",
        split="val",
        label="best",
        min_ans_occ=1
    )
    DOWNLOAD = VQA(
        dir_data=pip_mm_path,
        features="coco-bottomup",
        split="test",
        label="best",
        min_ans_occ=1
    )
    # multilabel
    DOWNLOAD = VQA(
        dir_data=pip_mm_path,
        features="coco-bottomup",
        split="train",
        label="multilabel",
        min_ans_occ=1
    )
    DOWNLOAD = VQA(
        dir_data=pip_mm_path,
        features="coco-bottomup",
        split="val",
        label="multilabel",
        min_ans_occ=1
    )
    DOWNLOAD = VQA(
        dir_data=pip_mm_path,
        features="coco-bottomup",
        split="test",
        label="multilabel",
        min_ans_occ=1
    )
    import sys; sys.exit()




def store_tvqa_cleaned_tokens():
    tvqa_path = os.path.join( os.path.dirname(__file__), "tvqa/tvqa_modality_bias/data")
    tvqa_data = myutils.load_pickle(os.path.join( tvqa_path , "total_dict.pickle" ))
    vcpts = myutils.load_pickle(os.path.join( tvqa_path, "vcpt_features/det_visual_concepts_hq.pickle" ))
    #import ipdb; ipdb.set_trace()
    print(f"Collecting TVQA Questions...")

    # Generate objects for plots
    questions, answers, correct_answers, vcpts_collect, ts_subtitles, nots_subtitles = [], [], [], [], [], []

    for qidx, qdict in tqdm(enumerate(tvqa_data.values()), total=len(tvqa_data)):
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
    print("cleaning features")
    questions = [ myutils.clean_word(word) for sentence in questions for word in sentence.split() ]
    answers = [ myutils.clean_word(word) for sentence in answers for word in sentence.split() ]
    correct_answers = [ myutils.clean_word(word) for sentence in correct_answers for word in sentence.split() ]
    vcpts_collect = [ myutils.clean_word(word) for sentence in vcpts_collect for word in sentence.split() ]
    ts_subtitles = [ myutils.clean_word(word) for sentence in ts_subtitles for word in sentence.split() ]
    nots_subtitles = [ myutils.clean_word(word) for sentence in nots_subtitles for word in sentence.split() ]
    myutils.save_pickle( {
        "questions":questions,
        "answers":answers,
        "correct_answers":correct_answers,
        "vcpts":vcpts_collect,
        "subs_ts":ts_subtitles,
        "subs_nots":nots_subtitles
    }, os.path.join( os.path.dirname(__file__) , "tvqa/tvqa_modality_bias/data", "cleaned_total_tokens.pickle"))

def load_tvqa_clean_tokens():
    return myutils.load_pickle(os.path.join( os.path.dirname(__file__) , "tvqa/tvqa_modality_bias/data", "cleaned_total_tokens.pickle"))



def load_tvqa_at_norm_threshold(norm="conc-m", norm_threshold=0.7, greater_than=True, include_vid=True, unique_ans=True):
    # Get TVQA datasets
    train_tvqa_dat = myutils.load_json(os.path.abspath(f"{os.path.abspath(__file__)}/../tvqa/tvqa_modality_bias/data/tvqa_train_processed.json"))
    val_tvqa_dat = myutils.load_json(os.path.abspath(f"{os.path.abspath(__file__)}/../tvqa/tvqa_modality_bias/data/tvqa_val_processed.json"))
    if include_vid:
        vid_h5 = h5py.File(os.path.abspath(f"{os.path.abspath(__file__)}/../tvqa/tvqa_modality_bias/data/imagenet_features/tvqa_imagenet_pool5_hq.h5"), "r", driver=None)

    q_n_correcta = [{"q":qdict["q"], "cans":qdict[f"a{qdict['answer_idx']}".lower()], "vid_name":qdict["vid_name"], "located_frame":qdict["located_frame"]} for qdict in train_tvqa_dat+val_tvqa_dat]
    # Get norm dictionary
    norm_dict_path =   os.path.abspath(f"{os.path.abspath(__file__)}/../misc/all_norms.pickle")
    norm_dict = myutils.load_pickle(norm_dict_path)

    # Filter questions with answers that are of certain concreteness
    norm_ans = []
    qs_w_norm_ans = []
    print(f"Collecting TVQA Qs and As of certain concretness")
    for qa in tqdm(q_n_correcta, total=len(q_n_correcta) ):
        q,a  = qa["q"], qa["cans"]  # cans (correct answer)
        try:    # Speedily developing this code, comeback later to replace with .get
            if norm == "conc-m":
                ans_norm = norm_dict.words[a]["conc-m"]["sources"]["MT40k"]["scaled"]
                if greater_than:
                    if ans_norm > norm_threshold:
                        norm_ans.append(a)
                        qs_w_norm_ans.append(qa)
                else:
                    if ans_norm < norm_threshold:
                        norm_ans.append(a)
                        qs_w_norm_ans.append(qa)
        except KeyError:
            pass

    answers =  list(set(norm_ans))
    answers = " ".join(answers)
    if unique_ans:
        norm_seqs = ["@@".join( [qa["q"], answers]) for qa in qs_w_norm_ans]
    else:
        norm_seqs = ["@@".join( [qa["q"], qa["cans"]]) for qa in qs_w_norm_ans]

    if include_vid:  # Load visual features
        norm_imgnt = [ vid_h5[qa["vid_name"]][qa["located_frame"][0]:qa["located_frame"][1]] for qa in qs_w_norm_ans ]
        norm_seqs = (norm_seqs, norm_imgnt)
    return norm_seqs
