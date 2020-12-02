import os, sys
import myutils
from tqdm import tqdm


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
