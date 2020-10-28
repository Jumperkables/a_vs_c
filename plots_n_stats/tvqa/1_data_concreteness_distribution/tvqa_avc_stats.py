__author__ = "Jumperkables"
import sys, os
sys.path.append( os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) )

import myutils
import word_norms
from nltk.corpus import stopwords
import spacy

word2norm = word_norms.word_to_MRC(None)
english_stopwords = list(stopwords.words("english"))
nlp = spacy.load('en')


"""
UTILITY FUNCTIONS
"""
def word_to_stats(word):
    if word in english_stopwords:
        stat = "STOPWORD"
    else:
        stat = word2norm.get(word,{'conc':"NO_NORM"})['conc']
    return stat

def line_to_stats(line, eos=True, downcase=True):
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
        "has_norm"      :len([ word for word in sentence if type(word)==int]),
        "hasnt_norm"    :sentence.count("NO_NORM"),
        "length"        :len(sentence),
        "n_stopwords"   :sentence.count("STOPWORD")
    }
    sentence = [word for word in sentence if word not in ['STOPWORD', "NO_NORM"]]
    sentence_no0 = [word for word in sentence if word not in ["STOPWORD", "NO_NORM", 0]]
    stats["conc==0"]        = sentence.count(0)
    stats["0<conc<=100"]    = len([score for score in sentence if (100>=score) and (score>0)])
    stats["100<conc<=200"]  = len([score for score in sentence if (200>=score) and (score>100)])
    stats["200<conc<=300"]  = len([score for score in sentence if (300>=score) and (score>200)])
    stats["300<conc<=400"]  = len([score for score in sentence if (400>=score) and (score>300)])
    stats["400<conc<=500"]  = len([score for score in sentence if (500>=score) and (score>400)])
    stats["500<conc<=600"]  = len([score for score in sentence if (600>=score) and (score>500)])
    stats["600<conc<=700"]  = len([score for score in sentence if (700>=score) and (score>600)])
    if len(sentence) == 0:
        stats["conc_density"] = None
    else:
        stats["conc_density"] = sum(sentence)/float(len(sentence))

    if len(sentence_no0) == 0:
        stats["conc_density_no_0"] = None
    else:
        stats["conc_density_no_0"] = sum(sentence_no0)/float(len(sentence_no0))
    return stats

"""
SIGNIFICANT FUNCTIONS
"""
def json_2_avcstats(tvqa_json, vcpts):
    """
    Input:  A TVQA processed json
    Output: A dictionary of {qid : a_vs_c stats}
    """
    stat_dict = {}
    for counter, question in enumerate(tvqa_json):
        print(counter)
        q_return = {"answer_idx":question.get("answer_idx", None)}

        sub, sub_perword = line_to_stats(question['located_sub_text'])
        sub_stats = conc_stats(sub_perword)
        q_return["sub-ts"] = sub
        q_return["sub-ts_stats"] = sub_stats

        q, q_perword = line_to_stats(question['q'])
        q_stats = conc_stats(q_perword)
        q_return["q"] = q
        q_return["q_stats"] = q_stats
        
        vcpt = vcpts[question["vid_name"]][question["located_frame"][0]:question["located_frame"][1]]
        vcpt = " ".join(list(set([cpt.split()[-1] for line in vcpt for cpt in line.split(" , ") if cpt!="" ])))
        vcpt, vcpt_perword = line_to_stats(vcpt)
        vcpt_stats = conc_stats(vcpt_perword)
        q_return["vcpt"] = vcpt
        q_return["vcpt_stats"] = vcpt_stats
        
        # For each answers
        akeys = ["a0","a1","a2","a3","a4"]
        for akey in akeys:
            ans, ans_perword = line_to_stats(question[akey])
            ans_stats = conc_stats(ans_perword)
            q_return[akey] = ans
            q_return[f"{akey}_stats"] = ans_stats


        q.remove('<eos>')
        parsed = nlp(" ".join(q))
        nsubjs = [word for word in parsed if word.dep_ == "nsubj"]
        iobjs = [word for word in parsed if word.dep_ == "iobj"]
        dobjs = [word for word in parsed if word.dep_ == "dobj"]
        subjs_objs = [str(word) for word in nsubjs+iobjs+dobjs if str(word) not in english_stopwords]
        subjs_objs, subjs_objs_perword = line_to_stats(" ".join(subjs_objs))
        subjs_objs_stats = conc_stats(subjs_objs_perword)
        q_return["subj"] = subjs_objs
        q_return["subj_stats"] = subjs_objs_stats

        stat_dict[question["qid"]] = q_return
    return stat_dict


def flatten_tvqa_qid():
    qtype_dict = myutils.load_pickle('/home/jumperkables/kable_management/data/tvqa/q_type/val_q_type_dict.pickle')
    avc_stats = myutils.load_pickle("/home/jumperkables/kable_management/projects/a_vs_c/tvqa/avc_statistics/avc_stats.pickle")
    qtype_qids = {}
    #what_dict = {'after': {}, 'when': {}, 'before':{}, 'other':{}}
    #who_dict = {'after': {}, 'when': {}, 'before':{}, 'other':{}}
    #why_dict = {'after': {}, 'when': {}, 'before':{}, 'other':{}}
    #where_dict = {'after': {}, 'when': {}, 'before':{}, 'other':{}}
    #how_dict = {'after': {}, 'when': {}, 'before':{}, 'other':{}}
    #which_dict = {}
    #other_dict = {}

    # WHAT
    what = qtype_dict['what']
    what_qids = []
    what_dict = myutils.merge_two_dicts(what['after'], what['when'])
    what_dict2= myutils.merge_two_dicts(what['before'], what['other'])
    what_dict = myutils.merge_two_dicts(what_dict, what_dict2)
    what_qs = [xx['qid'] for xx in what_dict.values() ]

    # WHO
    who = qtype_dict['who']
    who_qids = []
    who_dict = myutils.merge_two_dicts(who['after'], who['when'])
    who_dict2= myutils.merge_two_dicts(who['before'], who['other'])
    who_dict = myutils.merge_two_dicts(who_dict, who_dict2)
    who_qs = [xx['qid'] for xx in who_dict.values() ]
    # why
    why = qtype_dict['why']
    why_qids = []
    why_dict = myutils.merge_two_dicts(why['after'], why['when'])
    why_dict2= myutils.merge_two_dicts(why['before'], why['other'])
    why_dict = myutils.merge_two_dicts(why_dict, why_dict2)
    why_qs = [xx['qid'] for xx in why_dict.values() ]
    # where
    where = qtype_dict['where']
    where_qids = []
    where_dict = myutils.merge_two_dicts(where['after'], where['when'])
    where_dict2= myutils.merge_two_dicts(where['before'], where['other'])
    where_dict = myutils.merge_two_dicts(where_dict, where_dict2)
    where_qs = [xx['qid'] for xx in where_dict.values() ]
    # how
    how = qtype_dict['how']
    how_qids = []
    how_dict = myutils.merge_two_dicts(how['after'], how['when'])
    how_dict2= myutils.merge_two_dicts(how['before'], how['other'])
    how_dict = myutils.merge_two_dicts(how_dict, how_dict2)
    how_qs = [xx['qid'] for xx in how_dict.values() ]

    # Which
    which = qtype_dict['which']
    which_qs = [xx['qid'] for xx in which.values() ]
    # other
    other = qtype_dict['other']
    other_qs = [xx['qid'] for xx in other.values() ]

    flat_qtype = {
        'what':what_qs,
        "who":who_qs,
        "why":why_qs,
        "where":where_qs,
        "how":how_qs,
        "which":which_qs,
        "other":other_qs
    }
    myutils.save_pickle(flat_qtype, '/home/jumperkables/kable_management/data/tvqa/q_type/val_flat_q_type_byqid_dict.pickle')
    print("ooooooofff")


def avg_conc_by_qtype():
    #temp = myutils.load_pickle("/home/jumperkables/kable_management/projects/a_vs_c/tvqa/avc_statistics/tvqa_conc_avgs_by_qtype.pickle")
    #print(temp)
    #import sys; sys.exit()
    qtype_qid = myutils.load_pickle('/home/jumperkables/kable_management/data/tvqa/q_type/val_flat_q_type_byqid_dict.pickle')
    avc_stats = myutils.load_pickle("/home/jumperkables/kable_management/projects/a_vs_c/tvqa/avc_statistics/avc_stats.pickle")
    #import ipdb; ipdb.set_trace()
    what_qids = qtype_qid['what']
    what_conc_avgs = [ avc_stats[what_qid]["q_stats"]["conc_density_no_0"] for what_qid in what_qids ]

    who_qids = qtype_qid['who']
    who_conc_avgs = [ avc_stats[who_qid]["q_stats"]["conc_density_no_0"] for who_qid in who_qids ]

    why_qids = qtype_qid['why']
    why_conc_avgs = [ avc_stats[why_qid]["q_stats"]["conc_density_no_0"] for why_qid in why_qids ]

    where_qids = qtype_qid['where']
    where_conc_avgs = [ avc_stats[where_qid]["q_stats"]["conc_density_no_0"] for where_qid in where_qids ]

    how_qids = qtype_qid['how']
    how_conc_avgs = [ avc_stats[how_qid]["q_stats"]["conc_density_no_0"] for how_qid in how_qids ]

    which_qids = qtype_qid['which']
    which_conc_avgs = [ avc_stats[which_qid]["q_stats"]["conc_density_no_0"] for which_qid in which_qids ]

    other_qids = qtype_qid['other']
    other_conc_avgs = [ avc_stats[other_qid]["q_stats"]["conc_density_no_0"] for other_qid in other_qids ]

    what_conc_avgs  = [ avg for avg in what_conc_avgs if avg != None ]
    who_conc_avgs   = [ avg for avg in who_conc_avgs if avg != None]
    why_conc_avgs   = [ avg for avg in why_conc_avgs if avg != None]
    where_conc_avgs = [ avg for avg in where_conc_avgs if avg != None]
    how_conc_avgs   = [ avg for avg in how_conc_avgs if avg != None]
    which_conc_avgs = [ avg for avg in which_conc_avgs if avg != None]
    other_conc_avgs = [ avg for avg in other_conc_avgs if avg != None]

    what_conc_avgs = sum(what_conc_avgs)/len(what_conc_avgs)
    who_conc_avgs = sum(who_conc_avgs)/len(who_conc_avgs)
    why_conc_avgs = sum(why_conc_avgs)/len(why_conc_avgs)
    where_conc_avgs = sum(where_conc_avgs)/len(where_conc_avgs)
    how_conc_avgs = sum(how_conc_avgs)/len(how_conc_avgs)
    which_conc_avgs = sum(which_conc_avgs)/len(which_conc_avgs)
    other_conc_avgs = sum(other_conc_avgs)/len(other_conc_avgs)
    #import ipdb; ipdb.set_trace()
    myutils.save_pickle({
        "what":what_conc_avgs,
        "who":who_conc_avgs,
        "why":why_conc_avgs,
        "where":where_conc_avgs,
        "how":how_conc_avgs,
        "which":which_conc_avgs,
        "other":other_conc_avgs
    }, "/home/jumperkables/kable_management/projects/a_vs_c/tvqa/avc_statistics/tvqa_conc_avgs_by_qtype.pickle")
    print("OOOFF")


def Qids_by_conc():
    #################################
    temp = myutils.load_pickle("/home/jumperkables/kable_management/projects/a_vs_c/tvqa/avc_statistics/subjsobjs_by_conc_no0.pickle")
    total = sum([len(sublist) for sublist in temp.values()])
    print(f"\
        0 : {100*len(temp['0'])/total:.5f}%\n\
        None : {100*len(temp['None'])/total:.5f}%\n\
        0-100 : {100*len(temp['0-100'])/total:.5f}%\n\
        100-200 : {100*len(temp['100-200'])/total:.5f}%\n\
        200-300 : {100*len(temp['200-300'])/total:.5f}%\n\
        300-400 : {100*len(temp['300-400'])/total:.5f}%\n\
        400-500 : {100*len(temp['400-500'])/total:.5f}%\n\
        500-600 : {100*len(temp['500-600'])/total:.5f}%\n\
        600-700 : {100*len(temp['600-700'])/total:.5f}%\n\
    ")
    import sys; sys.exit()
    #################################
    avc_stats = myutils.load_pickle("/home/jumperkables/kable_management/projects/a_vs_c/tvqa/avc_statistics/avc_stats.pickle")
    qid_by_conc = {
        "0":[],
        "0-100":[],
        "100-200":[],
        "200-300":[],
        "300-400":[],
        "400-500":[],
        "500-600":[],
        "600-700":[],
        "None":[]
    }
    for qid in avc_stats.keys():
        correct_id = avc_stats.get(qid)["answer_idx"]
        if correct_id != None:
            correct_id = f"a{correct_id}_stats"
            density = avc_stats.get(qid, {f"subj_stats":{"conc_density_no_0":"MISSING_Q"}})[f"subj_stats"]["conc_density_no_0"]
            #density = avc_stats.get(qid, {f"a{correct_id}_stats":{"conc_density_no_0":"MISSING_Q"}})[f"a{correct_id}_stats"]["conc_density_no_0"]
            #density_temp = [ avc_stats.get(qid, {f"{ansid}":{"conc_density_no_0":"MISSING_Q"}})[f"{ansid}"]["conc_density_no_0"] for ansid in ["a0_stats", "a1_stats", "a2_stats", "a3_stats", "a4_stats"] if ansid != correct_id ]
        else:
            density = "MISSING_Q"
        #density = avc_stats.get(qid, {"vcpt_stats":{"conc_density_no_0":"MISSING_Q"}})["vcpt_stats"]["conc_density_no_0"]
        if density == "MISSING_Q":
            pass
        else:
            #for density in density_temp:
            if density == 0:
                qid_by_conc["0"].append(qid)
            elif density == None:
                qid_by_conc["None"].append(qid)
            elif (density > 0) and (density<=100):
                qid_by_conc["0-100"].append(qid)
            elif (density > 100) and (density<=200):
                qid_by_conc["100-200"].append(qid)
            elif (density > 200) and (density<=300):
                qid_by_conc["200-300"].append(qid)
            elif (density > 300) and (density<=400):
                qid_by_conc["300-400"].append(qid)
            elif (density > 400) and (density<=500):
                qid_by_conc["400-500"].append(qid)
            elif (density > 500) and (density<=600):
                qid_by_conc["500-600"].append(qid)
            elif (density > 600) and (density<=700):
                qid_by_conc["600-700"].append(qid)
            else:
                raise ValueError(f"Density of {density} is out of expected 0-700 range")
    import ipdb; ipdb.set_trace()
    myutils.save_pickle(qid_by_conc, "/home/jumperkables/kable_management/projects/a_vs_c/tvqa/avc_statistics/subjsobjs_by_conc_no0.pickle")

def question_subject_conc():
    avc_stats = myutils.load_pickle("/home/jumperkables/kable_management/projects/a_vs_c/tvqa/avc_statistics/avc_stats.pickle")

    for qid in avc_stats.keys():
        question = avc_stats[qid]['q']
        question.remove('<eos>')
        parsed = nlp(" ".join(question))
        nsubjs = [word for word in parsed if word.dep_ == "nsubj"]
        iobjs = [word for word in parsed if word.dep_ == "iobj"]
        dobjs = [word for word in parsed if word.dep_ == "dobj"]
        subjs_objs = [word for word in nsubj+iobjs+dobjs if word not in english_stopwords]
        print(subjs_objs)
        obj_conc = conc_stats(subjs_objs)

        #print(f"Q:{parsed}.\nNsubjs:{nsubjs}\nIobjs:{iobjs}\nDobjs:{dobjs}")
        import ipdb; ipdb.set_trace()
        print("Waiting")



if __name__ == "__main__":
    #flatten_tvqa_qid()
    ############################
    #print("Collecting TVQA a_vs_c statistics")
    #root_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tvqa_modality_bias/data" )
    #train = os.path.join(root_data_dir, "tvqa_train_processed.json")
    #test  = os.path.join(root_data_dir, "tvqa_test_public_processed.json")
    #val   = os.path.join(root_data_dir, "tvqa_val_processed.json")
    #vcpts = os.path.join(root_data_dir, "vcpt_features/det_visual_concepts_hq.pickle")

    #train = myutils.load_json(train)
    #test  = myutils.load_json(test)
    #val   = myutils.load_json(val)
    #vcpts = myutils.load_pickle(vcpts)

    #train = json_2_avcstats(train, vcpts)
    #test  = json_2_avcstats(test, vcpts)
    #val   = json_2_avcstats(val, vcpts)

    #overall = myutils.merge_two_dicts(train, test)
    #overall = myutils.merge_two_dicts(overall, val)

    #save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),  "avc_stats.pickle")
    #myutils.save_pickle(overall, save_path)
    ##############################
    #avg_conc_by_qtype()
    #############################
    Qids_by_conc()
    #############################
    #question_subject_conc()
    #############################

