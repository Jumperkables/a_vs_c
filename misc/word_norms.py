__author__ = "Jumperkables"

import pandas as pd
import argparse
import os, sys, copy
import pyreadr
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re, math

# Project imports
from . import myutils
#import myutils
from .USF_teonbrooks_free import USF_Free
#from USF_teonbrooks_free import USF_Free
from .MRC_samzhang_extract import MRC_Db
#from MRC_samzhang_extract import MRC_Db

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stpwrds = stopwords.words("english")

pos_translation = {
    # USF notation
    'PP':"prepostion",
    'AD':"adverb",
    'V':"verb",
    'N':"noun",
    'QPS':"cue",
    'P':"pronoun", 
    'AJ':"adjective", 
    'I':"interjection", 
    'C':"conjunction",
    'TPS':"target",
    "ADV":"adverb",
    "AV":"adverb",
    "PRP":"prepostion",
    "ADJ":"adjective",
    "A":"adjective?",
    "INT":"interjection",
    "":None,
}
mrc_pos_trans = {
    # Part of MRC notation
    "N":"noun",
    "J":"adjective",
    "V":"verb",
    "A":"adverb",
    "O":"other",
    "R":"prepostion",
    "C":"conjunction",
    "U":"pronoun",
    "I":"interjection",
    "P":"past participle",
    " ":None,
}
simlex_pos_trans = {
    "A":"adjective",
    "N":"noun",
    "V":"verb"
}
vinson_pos_trans = {
    "object":"noun",
    "actionN":"action-noun",
    "actionV":"verb"
}
simverb_type = {
    'ANTONYMS':"antonyms", 
    'COHYPONYMS':"cohyponyms", 
    'HYPER/HYPONYMS':"hyper/hyponyms", 
    'NONE':None, 
    'SYNONYMS':"synonyms"
}
kup_reilly_pos_trans = ['Conjunction',"Name","Verb","Adverb","Determiner","Pronoun","Adjective","Abbreviation","Interjection","Preposition","Noun","Number"]


########################################################################
# Running functions
########################################################################
# Assoc & SimLex words
is_assoc_or_simlex = []
assoc_or_simlex_word_pairs = []
simlex_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/SimLex-999/SimLex-999.txt")
simlex999 = pd.read_csv(simlex_path, delimiter="\t")
seen = []
to_drop = []
# Remove the duplicate entries
for idx, row in enumerate(simlex999.iterrows()):
    w1 = row[1]['word1'].lower()
    w2 = row[1]['word2'].lower()
    is_assoc_or_simlex.append(w1)
    is_assoc_or_simlex.append(w2)
    assoc_or_simlex_word_pairs.append(f"{w1}|{w2}")
    assoc_or_simlex_word_pairs.append(f"{w2}|{w1}")
is_assoc_or_simlex = list(set(is_assoc_or_simlex))
assoc_or_simlex_word_pairs = list(set(assoc_or_simlex_word_pairs))


def wordlist_is_expanded_norm(wordlist):
    norm_dict = myutils.load_norms_pickle( os.path.join(os.path.dirname(__file__), "all_norms.pickle"))
    to_keep = []
    #from USF_teonbrooks_free import USF_Free
    #fa_file = "/home/jumperkables/a_vs_c/data/USF/teonbrooks/free_association.txt"
    #USF_Free = USF_Free(fa_file)
    #usf_words = USF_Free.db
    #SimVerb_data = pd.read_csv("/home/jumperkables/a_vs_c/data/SimVerb/SimVerb-3500.txt", sep="\t")
    for w1 in wordlist:
        for w2 in wordlist:
            w1 = w1.lower()
            w2 = w2.lower()
            nd = norm_dict.word_pairs.get(f"{w1}|{w2}", None)
            if nd != None:
                if "sim" in nd:
                    #breakpoint()
                    if nd["sim"]["avg"] != None:
                        #breakpoint()
                        pass
                if "str" in nd:
                    if nd["str"]["avg"] != None:
                        #breakpoint()
                        pass

                ########################
                try:
                    assoc = nd["assoc"]['sources']['USF']['scaled']
                except KeyError:
                    assoc = None
                ########################
                try:
                    simlex = nd["simlex999-m"]["sources"]["SimLex999"]["scaled"]
                except KeyError:
                    simlex = None
                ########################
                try:
                    sim = nd["sim"]["avg"]
                except KeyError:
                    sim = None
                ########################
                try:
                    usf_str = nd["str"]["avg"]
                except KeyError:
                    usf_str = None
                print(f"assoc: {assoc} | simlex: {simlex} | sim: {sim} | str: {usf_str}")
                if assoc != None or sim != None or usf_str != None or simlex != None:
                    if assoc == None:
                        assoc = 0
                    if simlex == None:
                        simlex = 0
                    if sim == None:
                        sim = 0
                    if usf_str == None:
                        usf_str = 0
                    if assoc >= 0.7 or sim >= 0.7 or usf_str >= 0.7 or simlex >= 0.7:
                        to_keep.append(w1)
                        to_keep.append(w2)
    to_keep = list(set(to_keep))
    return to_keep

def wordlist_is_assoc_or_simlex(wordlist):
    to_keep = []
    for w1 in wordlist:
        for w2 in wordlist:
            w1 = w1.lower()
            w2 = w2.lower()
            if (f"{w1}|{w2}" in assoc_or_simlex_word_pairs) or (f"{w2}|{w1}" in assoc_or_simlex_word_pairs):
                to_keep.append(w1)
                to_keep.append(w2)
    to_keep = list(set(to_keep))
    return to_keep


def word_is_assoc_or_simlex(word):
    if word in is_assoc_or_simlex:
        return True
    else:
        return False


def word_is_cOrI(norm_dict, word):
    is_conc = norm_dict.words.get(f"{word}", {}).get("conc-m", {}).get("avg", None)
    is_imgbl = norm_dict.words.get(f"{word}", {}).get("imag-m", {}).get("avg", None)
    if (is_conc in [None,0,0.]) and (is_imgbl in [None,0,0.]):
        return False
    else:
        return True

def word_is_Conc(norm_dict, word):
    is_conc = norm_dict.words.get(f"{word}", {}).get("conc-m", {}).get("avg", None)
    # check for if it has a word-pair norm
    print("Allowing conc scores of 0")
    if (is_conc in [None]):
        return False
    else:
        return True

def word_to_MRC(args):
    args = _if_main(args)
    MRC_dict    = MRC_Db(os.path.join(args.MRC_path, "samzhang111/mrc2.dct")) 
    mrc_keys = MRC_dict.mrc_keys
    word_to_MRC = MRC_dict.MRC_dict # We keep the class structure intact incase we want the SQLAlchemy session later
    return word_to_MRC

def word_to_norms(args):
    word_to_norms = {}
    args = _if_main(args)
    return word_to_norms

def get_concrete_words(args):
    args = _if_main(args)
    conc_words = []
    # MT40k
    MT40k = pd.read_csv(args.MT40k_path, sep="\t")
    mt40k_words = MT40k.query('`Conc.M` > 4')['Word']
    mt40k_words = [word for word in mt40k_words]
    conc_words += mt40k_words
   
    # CSLB
    CSLB_norms = pd.read_csv(os.path.join(args.CSLB_path, "norms.dat"), sep="\t")  
    #CSLB_feature_matrix = pd.read_csv(os.path.join(args.CSLB_path, "feature_matrix.dat"), sep="\t")
    cslb_words = np.unique(CSLB_norms['concept'].values).tolist()
    conc_words += cslb_words

    # USF
    USF_free_assoc = USF_Free(os.path.join(args.USF_path, "teonbrooks/free_association.txt"))
    usf_words = USF_free_assoc.db
    usf_words = usf_words.query("`TCON` > '5'")['TARGET']
    usf_words = list(set(usf_words.values))
    conc_words += usf_words

    # Clark and Paivio
    CP_a = pd.read_fwf(os.path.join(args.CP_path, "cp2004a.txt"))
    #CP_b = pd.read_fwf(os.path.join(args.CP_path, "cp2004b.txt"))
    CP_a = myutils.df_firstrow_to_header(CP_a)[:-1].drop(columns=['ITM'])
    #CP_b = myutils.df_firstrow_to_header(CP_b)[:-1]
    CP_a_words = list(CP_a.query("`CON` > '5'")["WORD"].values)
    conc_words += CP_a_words

    # Toront Word Pool
    TWP = pyreadr.read_r(os.path.join(args.TWP_path, "TWP.RData"))
    TWP = TWP['TWP']
    TWP_words = list(TWP.query("`concreteness` > 5")["word"].values)
    conc_words += TWP_words

    # Cleaning
    conc_words = [word.lower() for word in conc_words]

    return conc_words


######## Classes to create full norm dictionary
#class Wnorm():
#    def __init__(self, word):
#        self.word = word
#        dsets_dict = {
#            "MT40k":{"scaled":None, "original":None}
#        }
#        self.norms = {
#            "conc-m":   copy.deepcopy(dsets_dict),
#            "conc-sd":  copy.deepcopy(dsets_dict),
#        }
#
#    def __str__(self):
#        return f"Word: '{self.word}'\nNorms: \n{self.norms}"
#    
#    def __repr__(self):
#        return f"Word: '{self.word}'\nNorms: \n{self.norms}"
#
#    def __getitem__(self, norm):
#        return self.norms.get("norm", None)
#    
#    def update(self, norm, scaled, original, dset):
#        self.norms[norm][dset]["scaled"] = scaled
#        self.norms[norm][dset]["original"] = original

class Word2Norm():
    def __init__(self):
        self.NORMS = {
                'conc-m':   "Mean concreteness", 
                "conc-sd":  "Standard deviation of concreteness", 
                "nphon":    "Number of phonemes", 
                "nsyl":     "Number of syllables",
                "imag-m":   "Mean imagability",
                "imag-sd":  "Standard deviation of imagability",

                "vfreq":    "Verbal frequency",
                "kf_freq":  "Kucera & Francis frequency",

                "fam":      "Familiarity",
                "mean":     "Meaningfullness",
                "pleas":    "Pleasantness",
                "categ":    "Categorisability",
                "emo":      "Emotionality",
                "toglia":   "The 'Toglia & Battig metric (1978)' provided via in Cortese (2004)",
                "dist":     "Distinctiveness as in McRae (excluding taxonomics)",
                "eod":      "Ease of definition",

                "rt-m":     "Mean reaction time (Cortese 2004)",
                "rt-sd":    "Standard deviation of reaction time",
                "rt-nam":   "Naming reaction time (Reilly ELP)",
                "rt-lex_dec":"Alternative reaction time (Reilly ELP)",

                "simlex999-m":  "SimLex999 metric",
                "simlex999-sd": "Standard deviation of SimLex999",
                "assoc":    "Word association metric",
                "str":      "USF directional 'strength'",
                "rstr":     "USF Resonant strength",
                "oastr":    "USF overlapping strength",
                "mstr":     "USF mediated strength",
                "pos":      "Part-of-speech",
                "modal":    "Modality",
                "sim":      "Similarity, the similarity and USF strength metrics may be related",
                "sem_rel":  "Type of semantic relationship e.g. hyponym, antonym etc..",

                "sem_dens": "Semantic density (Reilly ELP)",
                "sem_neig": "Semantic neighbors (Reilly ELP)",
                "sem_div": "Semantic diversity (Reilly ELP)",

                "num_func":     "Number of functional features",
                "num_vis_mot":  "Number of visual-motor features",
                "num_vis_fs":	"number of visual form and surface features",
                "num_vis_col":	"number of visual colour features",
                "num_sound":	"number of sound features",
                "num_taste":	"number of taste features",
                "num_smell":	"number of smell features",
                "num_tact":	"number of tactile features",
                "num_ency":	"number of encyclopedic features",
                "num_tax":	"number of taxonomic features",
                "num_mean":     "number of meanings",

                "mm_vis":       "Kastner's multimodal 'visual' metric",
                "mm_txt":       "Kastner's multimodal 'textual' metric",
                "mm_pho":       "Kastner's multimodal 'phonetic' metric",
                "mm_all":       "Kastner's multimodal 'all' metric",

                # FROM SIANPAR'S INDONESIAN WORD NORMS
                "val-m":        "Mean 'Valence'",
                "val-sd":       "Standard deviation of valence",
                "arou-m":       "Mean 'Arousal'",
                "arou-sd":      "Standard deviation of arousal",
                "dom-m":        "Mean 'Dominance'",
                "dom-sd":       "Standard deviation of dominance",
                "pred-m":       "Mean 'predictability'",
                "pred-sd":      "Standard deviation of predictability",

                # From the 'LNC' dataset in Reilly's compilation
                "lnc-audi":     "Auditory",
                "lnc-gust":     "Gustatory",
                "lnc-hapt":     "Haptic",
                "lnc-interoc":  "Interoceptive",
                "lnc-olfact":   "Olfactory",
                "lnc-foot_leg": "Foot-leg",
                "lnc-hand_arm": "Hand-arm",
                "lnc-head":     "Head",
                "lnc-mouth":    "Mouth",
                "lnc-torso":    "Torso",
                "lnc-dom_sense":"'Dominant' sense (i assume)",
        }
        self.DSETS = {
            "MT40k":    {"norms":["conc-m","conc-sd"], 
                         "description":"Pending"},
            "MRC":      {"norms":["conc-m","imag-m","nphon","nsyl","vfreq","kf_freq","fam","mean","pos"],
                         "description": "Verbal frequency are from 'Brown'. Meaningfullness values are the non-zero average of Paivio and Colarado norms"},
            "USF":      {"norms":["assoc","conc-m","str","rstr","oastr","mstr","pos"],
                         "description": "USF association metrics of some kind were provided by SimLex999 dataset, how to properly normalise them is unclear. I have currently decided to normalise them as if the distribution is ranged 0-10. NOTE, There are plenty more norms we have not added so far"},
            "SimLex999":{"norms":["conc-m","simlex999-m","simlex999-sd"],
                         "description": ""},
            "Vinson":   {"norms":["pos","modal"],
                         "description":"Pending"},
            "McRae":    {"norms":["kf_freq","num_func","num_vis_mot","num_vis_fs","num_vis_col","num_sound","num_taste","num_smell","num_tact","num_ency","num_tax"],
                         "description":"Pending"},
            "SimVerb":  {"norms":["sim","sem_rel"],
                         "description":"The similarity and USF strength metrics may be related"},
            "CP":       {"norms":["conc-m","kf_freq","imag-m","mean","nsyl","emo","pleas","num_mean","fam","eod"],
                         "description":"The Clark-Paivio word norms"},
            "TWP":      {"norms":["imag-m","conc-m","kf_freq"],
                         "description":"TWP Word norms"},
            "Battig":   {"norms":["kf_freq","nsyl"],
                         "description":"Battig Word norms"},
            "Cortese":  {"norms":["rt-m","rt-sd","imag-m","imag-sd","toglia"],
                         "description":"Cortese dataset"},
            "Reilly":   {"norms":["conc-m","imag-m","kf_freq","nphon","nsyl","rt-m","fam"],
                         "description":"2 parts of this dataset exist. One from Reilly and another an updated augmentation of norms also provided by Reilly"},
            "mm_imgblty":{"norm":["mm_vis","mm_txt","mm_pho","mm_all"],
                         "description":"A multimodal imagability feature dataset supplied by one kind Marc Kastner"},
            "sianpar_indo":{"norm":["val-m","val-sd","arou-m","arou-sd","conc-m","conc-sd","dom-m","dom-sd","pred-m","pred-sd"],
                         "description":"Sianpar's dataset of norms for indonesian words, (english translations included)"},
            "yee_chinese":{"norm":["val-m", "arou-m", "imag-m", "conc-m"],
                         "description":"Yee's word norm dataset for Chinese-translated words"},
            "megahr":   {"norm":["conc-m", "imag-m"],
                         "description":"MEGAHR Cross Lingual Database word norms"},
            "glasgow":  {"norm":["conc-m", "imag-m", "arou-m", "val-m", "fam-m", "dom-m"],
                         "description":"Glasgow Norms of 5500 words"},
            "GLS-Reilly":  {"norm":["arou-m", "dom-m", "imag-m", "val-m", "conc-m", "fam"],
                         "description":"The GLS dataset split provided by Reilly"},
            "KUP-Reilly":  {"norm":["pos", "nphon", "nsyl"],
                         "description":"The KUP dataset split provided by Reilly"},
            "ELP-Reilly":  {"norm":["rt-nam","rt-lex_dec","sem_dens","sem_neig","sem_div"], # FINISH THIS
                         "description":"The ELP dataset split provided by Reilly"},
            "BRYS-Reilly":  {"norm":["conc-m", "pos"],
                         "description":"The BRYS dataset split provided by Reilly"},
            "LNC-Reilly":  {"norm":["lnc-audi","lnc-gust","lnc-hapt","lnc-interoc","lnc-olfact","lnc-foot_leg","lnc-hand_arm","lnc-head","lnc-mouth","lnc-torso","lnc-dom_sense"], # FINISH THIS
                         "description":"The LNC dataset split provided by Reilly"},
            "WAR-Reilly":  {"norm":["val-m", "arou-m", "dom-m"],
                         "description":"The WAR dataset split provided by Reilly"},
        }
        self.words = {}
        self.word_pairs = {}

    def __len__(self):
        return(len(self.words)+len(self.word_pairs))

    def __str__(self):
        return_str = ""
        for idx, wrd in enumerate(self.words.keys()):
            if idx>3:
                return_str+="..."
                break
            return_str+=f"'{wrd}':\n"
            for nrm, nrm_dict in self.words[wrd].items():
                return_str+=f"    '{nrm}':\n        'avg': {nrm_dict['avg']}\n        'source': {nrm_dict['sources']}\n"
        return f"{return_str}\n\nContained Norms: {self.NORMS}\nContained Datasets: {self.DSETS}"

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, word):
        if type(word) == tuple and len(word) == 2:
            word0, word1 = word
            return self.word_pairs[f"{word0}|{word1}"]
        else:
            return self.words[word]

    def update(self, word, norm, scaled, original, dset):
        assert norm in self.NORMS.keys(), f"{norm}  is not a recognised norm"
        assert dset in self.DSETS.keys(), f"{dset} is not recognised"
        if (scaled != scaled) or (original != original):
            return None # SKIP NaN values
        if type(word) == tuple and len(word) == 2:
            word = f"{word[0]}|{word[1]}"
            if word not in self.word_pairs.keys():
                self.word_pairs[word] = {}
            if norm not in self.word_pairs[word].keys():
                self.word_pairs[word][norm] = {"avg":None, "sources":{}}
            if dset in self.word_pairs[word][norm]["sources"].keys():
                return None
            self.word_pairs[word][norm]["sources"][dset] = {"scaled":scaled, "original":original}
            current_norms = [ self.word_pairs[word][norm]["sources"][dst]["scaled"] for dst in self.word_pairs[word][norm]["sources"].keys() ]

            if type(scaled) in [float, int]: # The avergage of non-numeric norms isn't considered here
                current_norms = [ele for ele in current_norms if ele != 0]
                self.word_pairs[word][norm]["avg"] = list_avg(current_norms)           
        else:
            if word not in self.words.keys():
                self.words[word] = {}
            if norm not in self.words[word].keys():
                self.words[word][norm] = {"avg":None, "sources":{}}
            if dset in self.words[word][norm]["sources"].keys():
                return None
            self.words[word][norm]["sources"][dset] = {"scaled":scaled, "original":original}
            current_norms = [ self.words[word][norm]["sources"][dst]["scaled"] for dst in self.words[word][norm]["sources"].keys() ]
            
            if type(scaled) in [float, int]: # The avergage of non-numeric norms isn't considered here
                current_norms = [ele for ele in current_norms if ele != 0]
                self.words[word][norm]["avg"] = list_avg(current_norms)


def list_avg(lst):
    if len(lst) == 0:
        return 0
    else:
        return sum(lst)/len(lst)

def clean_word(word):
    if word != word:
        return None
    word = word.lower()
    word = re.sub('[^a-z0-9]+', '', word)
    return word

def shift_n_scale(value, offset, divisor):
    if value == 0:
        return 0
    else:
        return (value+offset)/divisor

def assert_float(value):
    try:
        value = float(value)
    except ValueError:
        return None
    return value

def explore_dsets(args):
    args = _if_main(args)
    print(f"Loading {'all ' if args.all else ''}{len(args.dsets)} dataset{'s' if len(args.dsets) > 1 else ''}: {args.dsets}")
    #breakpoint()
    flag_MT40k  = ("MT40k" in args.dsets) or args.all
    flag_CSLB   = ("CSLB" in args.dsets) or args.all
    flag_USF    = ("USF" in args.dsets) or args.all
    flag_MRC    = ("MRC" in args.dsets) or args.all
    flag_SimLex999  = ("SimLex999" in args.dsets) or args.all
    flag_Vinson = ("Vinson" in args.dsets) or args.all
    flag_McRae  = ("McRae" in args.dsets) or args.all
    flag_SimVerb= ("SimVerb" in args.dsets) or args.all
    flag_imSitu = ("imSitu" in args.dsets) or args.all
    flag_CP     = ("CP" in args.dsets) or args.all
    flag_TWP    = ("TWP" in args.dsets) or args.all
    flag_Battig = ("Battig" in args.dsets) or args.all
    flag_EViLBERT   = ("EViLBERT" in args.dsets) or args.all
    flag_Cortese    = ("Cortese" in args.dsets) or args.all
    flag_Reilly     = ("Reilly" in args.dsets) or args.all
    flag_MM_imgblty = ("MM_imgblty" in args.dsets) or args.all
    flag_sianpar_indo = ("sianpar_indo" in args.dsets) or args.all
    flag_yee_chinese = ("yee_chinese" in args.dsets) or args.all
    flag_megahr_crossling = ("megahr_crossling" in args.dsets) or args.all
    flag_glasgow = ("glasgow" in args.dsets) or args.all
    
    full_dict = Word2Norm()

    

    if flag_MT40k:
        #import ipdb; ipdb.set_trace()
        # Download from here: http://crr.ugent.be/papers/Concreteness_ratings_Brysbaert_et_al_BRM.txt
        MT40k = pd.read_csv(args.MT40k_path, sep="\t")
        with tqdm(total=len(MT40k)) as pbar:
            for key, row in MT40k.iterrows():
                pbar.set_description(f"MT40k: Processing")
                pbar.update(1)
                word = row['Word']
                concm = row["Conc.M"]
                concsd = row["Conc.SD"]
                full_dict.update(word, "conc-m", shift_n_scale(concm,-1,4), concm, "MT40k")
                full_dict.update(word, "conc-sd", shift_n_scale(concsd,0,4), concsd, "MT40k")
            #print(f"MT40k: \n {MT40k} \n\n")
            pbar.close()

    if flag_CSLB:
        #import ipdb; ipdb.set_trace()
        # Download from: http://www.csl.psychol.cam.ac.uk/propertynorms/ 
        CSLB_norms          = pd.read_csv(os.path.join(args.CSLB_path, "norms.dat"), sep="\t")  
        CSLB_feature_matrix = pd.read_csv(os.path.join(args.CSLB_path, "feature_matrix.dat"), sep="\t")
        #print(f"CSLB Norms: \n {CSLB_norms} \n\n")
        #print(f"CSLB Feature Matrix: \n {CSLB_feature_matrix} \n\n")

    if flag_USF:
        #import ipdb; ipdb.set_trace()
        # Code and txt provided by teonbrooks: https://github.com/teonbrooks/free_association
        USF_free_assoc  = USF_Free(os.path.join(args.USF_path, "teonbrooks/free_association.txt")).db
        with tqdm(total=len(USF_free_assoc)) as pbar:
            for key, row in USF_free_assoc.iterrows():
                pbar.set_description(f"USF dataset")
                pbar.update(1)
                cue, target             = row['CUE'], row["TARGET"] # word
                fsg, bsg                = assert_float(row["FSG"]), assert_float(row["BSG"])    # forward-strength
                cue_conc, t_conc        = assert_float(row["QCON"]), assert_float(row["TCON"])  # backward-strength
                connect_cue, connect_t  = assert_float(row["QMC"]), assert_float(row["TMC"])    # conc-m
                meds, overs             = assert_float(row["MSG"]), assert_float(row["OSG"])    # mediated, overlapping associated strength
                c_rs, t_rs              = assert_float(row["QRSG"]), assert_float(row["TRSG"])  # resonant strength
                cue, target             = clean_word(cue), clean_word(target)

                c_pos, t_pos = pos_translation[row["QPS"]], pos_translation[row["TPS"]]

                if c_pos is not None:
                    full_dict.update(cue, "pos", c_pos, c_pos, "USF")
                if t_pos is not None:
                    full_dict.update(cue, "pos", t_pos, t_pos, "USF")

                if cue_conc is not None:
                    full_dict.update(cue, "conc-m", shift_n_scale(cue_conc,-1,6), cue_conc, "USF")
                if t_conc is not None:
                    full_dict.update(target, "conc-m", shift_n_scale(t_conc,-1,6), t_conc, "USF")

                if c_rs is not None:
                    full_dict.update(cue, "rstr", shift_n_scale(c_rs,0,1), c_rs, "USF")
                if t_rs is not None:
                    full_dict.update(target, "rstr", shift_n_scale(t_rs,0,1), t_rs, "USF")

                if fsg is not None:
                    full_dict.update((cue,target), "str", shift_n_scale(fsg,0,1), fsg, "USF")
                if bsg is not None:
                    full_dict.update((target,cue), "str", shift_n_scale(bsg,0,1), bsg, "USF")

                if meds is not None:
                    full_dict.update((target,cue), "mstr", shift_n_scale(meds,0,1), meds, "USF")
                if overs is not None:
                    full_dict.update((target,cue), "oastr", shift_n_scale(overs,0,1), overs, "USF")

            pbar.close()



        #print(f"USF Database: \n Definitions: {USF_free_assoc.definitions} \n\n")
    
    if flag_MRC:
        #import ipdb; ipdb.set_trace()
        # Code and MRC file provided by samzhang111: https://github.com/samzhang111/mrc-psycholinguistics
        MRC_dict    = MRC_Db(os.path.join(args.MRC_path, "samzhang111/mrc2.dct"))
        #["conc-m","imag-m","nphon","nsyl","vfreq","fam","mean"]
        #full_dict.update(word, "conc-m", (concm-1)/4, concm, "MT40k")
        MRC_dict    = MRC_dict.MRC_dict # We keep the class structure intact incase we want the SQLAlchemy session later
        #kf_freq = [fdict['kf_freq'] for fdict in MRC_dict.values()]
        #print(f"Max: {max(kf_freq)}, Min:{min(kf_freq)}")
        #print( set([ MRC_dict[key]["wtype"] for key in MRC_dict.keys()]) )
        with tqdm(total=len(MRC_dict.keys())) as pbar:
            for wrd in MRC_dict.keys():
                pbar.set_description(f"MRC Database")
                pbar.update(1)
                dct = MRC_dict[wrd]
                concm,imag,nphon,nsyl,vfreq,fam,meanc,meanp = dct['conc'],dct['imag'],dct['nphon'],dct['nsyl'],dct['brown_freq'],dct['fam'],dct['meanc'],dct['meanp']
                wrd = clean_word(wrd)
                kf_freq = dct["kf_freq"]
                full_dict.update(wrd,"conc-m",shift_n_scale(concm,-100,600),concm,"MRC")
                full_dict.update(wrd,"imag-m",shift_n_scale(imag,-100,600),imag,"MRC")
                full_dict.update(wrd,"nphon",nphon,nphon,"MRC")
                full_dict.update(wrd,"nsyl",nsyl,nsyl,"MRC")
                full_dict.update(wrd,"vfreq",vfreq,vfreq,"MRC")
                full_dict.update(wrd,"kf_freq",kf_freq,kf_freq,"MRC")
                full_dict.update(wrd,"fam",fam,fam,"MRC")
                mean = [meanp,meanc]
                mean = [ele for ele in mean if ele != 0]
                mean = list_avg(mean)
                full_dict.update(wrd,"mean",shift_n_scale(mean,-100,600),mean,"MRC")
                pos = mrc_pos_trans[dct["wtype"]]
                full_dict.update(wrd,"pos",pos,pos,"MRC")
            pbar.close()
           
    if flag_SimLex999:
        #import ipdb; ipdb.set_trace()
        # Data from Felix Hill: https://fh295.github.io/simlex.html
        SimLex999   = pd.read_csv(os.path.join(args.SimLex999_path, "SimLex-999.txt"), sep="\t")
        #print(set(SimLex999["POS"]))
        with tqdm(total=len(SimLex999)) as pbar:
            for key, row in SimLex999.iterrows():
                pbar.set_description(f"SimLex999 Dataset")
                pbar.update(1)
                word1, word2 = row['word1'], row['word2']
                word1, word2 = clean_word(word1), clean_word(word2)
                conc1, conc2 = row["conc(w1)"], row["conc(w2)"]
                simlex999m = row["SimLex999"]
                simlex999sd = row["SD(SimLex)"]
                assoc_usf = row["Assoc(USF)"]
                pos = row["POS"]
                pos = simlex_pos_trans[pos]
    
                full_dict.update((word1,word2),"simlex999-m", shift_n_scale(simlex999m,0,10), simlex999m, "SimLex999")
                full_dict.update((word2,word1),"simlex999-m", shift_n_scale(simlex999m,0,10), simlex999m, "SimLex999")
    
                full_dict.update((word1,word2),"simlex999-sd", shift_n_scale(simlex999sd,0,10), simlex999sd, "SimLex999")
                full_dict.update((word2,word1),"simlex999-sd", shift_n_scale(simlex999sd,0,10), simlex999sd, "SimLex999")
    
                full_dict.update((word1,word2),"assoc", shift_n_scale(assoc_usf,0,10), assoc_usf, "USF")
                full_dict.update((word2,word1),"assoc", shift_n_scale(assoc_usf,0,10), assoc_usf, "USF")
    
                full_dict.update(word1, "conc-m", shift_n_scale(conc1,-1,4), conc1, "SimLex999")
                full_dict.update(word2, "conc-m", shift_n_scale(conc2,-1,4), conc2, "SimLex999")

                full_dict.update(word1, "pos", pos, pos, "SimLex999")
                full_dict.update(word2, "pos", pos, pos, "SimLex999")
            pbar.close()

    
    if flag_Vinson:
        #import ipdb; ipdb.set_trace()
        # Download from: https://static-content.springer.com/esm/art%3A10.3758%2FBRM.40.1.183/MediaObjects/Vinson-BRM-2008a.zip

        Vinson_word_cats    = pd.read_csv(os.path.join(args.Vinson_path, "word_categories.txt"), sep="\t")
        Vinson_features     = pd.read_csv(os.path.join(args.Vinson_path, "feature_list_and_types.txt"), sep="\t")
        Vinson_weight_mat0  = pd.read_csv(os.path.join(args.Vinson_path, "feature_weight_matrix_1_256.txt"), sep="\t")
        Vinson_weight_mat1  = pd.read_csv(os.path.join(args.Vinson_path, "feature_weight_matrix_257_456.txt"), sep="\t")
        with tqdm(total=len(Vinson_word_cats)) as pbar:
            for key, row in Vinson_word_cats.iterrows():
                pbar.set_description(f"Vinson Word Types:")
                pbar.update(1)
                word = row["word"]
                word = clean_word(word)
                pos = vinson_pos_trans[row["type"]]
                full_dict.update(word, "pos", pos, pos, "Vinson")
        pbar.close()
        #import ipdb; ipdb.set_trace()
        with tqdm(total=len(Vinson_features)) as pbar:
            for key, row in Vinson_features.iterrows():
                pbar.set_description(f"Vinson Modalities:")
                pbar.update(1)
                word = row["feature"]
                word = clean_word(word)
                if word != "":
                    modal = [key for key in row.keys() if key in ["Visual","Perceptual","Functional","Motoric"] and row[key]==1 ]
                    if modal != []:
                        full_dict.update(word, "modal", modal, modal, "Vinson")
        pbar.close()


    if flag_McRae:
        #import ipdb; ipdb.set_trace()
        # Download from: https://static-content.springer.com/esm/art%3A10.3758%2FBF03192726/MediaObjects/McRae-BRM-2005.zip
        # Information in the readme
        McRae_CONCS_brm     = pd.read_csv(os.path.join(args.McRae_path, "CONCS_brm.txt"), sep="\t")
        #kf_freq = [fdict['KF'] for key, fdict in McRae_CONCS_brm.iterrows()]
        #print(f"Max:{max(kf_freq)}, Min{min(kf_freq)}")
        with tqdm(total=len(McRae_CONCS_brm)) as pbar:
            for key, row in McRae_CONCS_brm.iterrows():
                pbar.set_description(f"McRae dataset")
                pbar.update(1)
                word = row["Concept"]
                word = clean_word(word)
                kf_freq = row["KF"]
                num_func = row["Num_Func"]
                num_vis_mot = row["Num_Vis_Mot"]
                num_vis_fs = row["Num_VisF&S"]
                num_vis_col = row["Num_Vis_Col"]
                num_sound = row["Num_Sound"]
                num_taste = row["Num_Taste"]
                num_smell = row["Num_Smell"]
                num_tact = row["Num_Tact"]
                num_ency = row["Num_Ency"]
                num_tax = row["Num_Tax"]

                nsyl = row["Length_Syllables"]
                fam = row["Familiarity"]
                dist = row["Mean_Distinct_No_Tax"]

                full_dict.update(word, "kf_freq", kf_freq, kf_freq, "McRae")
                full_dict.update(word, "num_func", num_func, num_func, "McRae")

                full_dict.update(word, "nsyl", nsyl, nsyl, "McRae")
                full_dict.update(word, "fam", shift_n_scale(fam,0,10), fam, "McRae")
                full_dict.update(word, "dist", dist, dist, "McRae")

                full_dict.update(word, "num_vis_mot", num_vis_mot, num_vis_mot, "McRae")
                full_dict.update(word, "num_vis_fs", num_vis_fs, num_vis_fs, "McRae")
                full_dict.update(word, "num_vis_col", num_vis_col, num_vis_col, "McRae")
                full_dict.update(word, "num_sound", num_sound, num_sound, "McRae")
                full_dict.update(word, "num_taste", num_taste, num_taste, "McRae")
                full_dict.update(word, "num_smell", num_smell, num_smell, "McRae")
                full_dict.update(word, "num_tact", num_tact, num_tact, "McRae")
                full_dict.update(word, "num_ency", num_ency, num_ency, "McRae")
                full_dict.update(word, "num_tax", num_tax, num_tax, "McRae")


            pbar.close()
       
    if flag_SimVerb:
        #import ipdb; ipdb.set_trace()
        # Download from: https://github.com/benathi/word2gm/tree/master/evaluation_data/simverb/data
        # There is more nuance to this data yet
        SimVerb_data    = pd.read_csv(os.path.join(args.SimVerb_path, "SimVerb-3500.txt"), sep="\t")
        first_ele = SimVerb_data.columns
        first_ele = [ele for ele in first_ele]
        first_ele[3] = float(first_ele[3])
        SimVerb_data.columns = ['word1','word2','pos','sim','type']
        SimVerb_data = SimVerb_data.append({SimVerb_data.columns[i]:first_ele[i] for i in range(5)}, ignore_index=True)
        #similarity = list(SimVerb_data['sim'])
        #filter(lambda v: v==v, similarity)
        #print(f"Max: {max(similarity)}, Min:{min(similarity)}")
        with tqdm(total=len(SimVerb_data)) as pbar:
            for key, row in SimVerb_data.iterrows():
                pbar.set_description(f"SimVerb dataset")
                pbar.update(1)
                sim = row["sim"]
                sim = assert_float(sim)
                breakpoint()
                sem_rel = simverb_type[row["type"]]
                word1, word2 = clean_word(row["word1"]), clean_word(row["word2"])
                full_dict.update(word1, "pos", "verb", "verb", "SimVerb")
                full_dict.update(word2, "pos", "verb", "verb", "SimVerb")
                if sim != None:
                    full_dict.update((word1,word2), "sim", shift_n_scale(sim,0,10), sim, "SimVerb")
                    full_dict.update((word2,word1), "sim", shift_n_scale(sim,0,10), sim, "SimVerb")
                full_dict.update((word1,word2), "sem_rel", sem_rel, sem_rel, "SimVerb")
                full_dict.update((word2,word1), "sem_rel", sem_rel, sem_rel, "SimVerb")
            pbar.close()


    if flag_imSitu:
        #import ipdb; ipdb.set_trace()
        # Download imSitu here: https://public.ukp.informatik.tu-darmstadt.de/coling18-multimodalSurvey/
        # Properly separate word from embedding values
        imSitu_verb_embeddings  = pd.read_csv(os.path.join(args.imSitu_path, "imSitu_verbs_averagedEmbeddings.w2vt"), sep="\t")
        with tqdm(total=len(imSitu_verb_embeddings)) as pbar:
            for key, row in imSitu_verb_embeddings.iterrows():
                pbar.set_description(f"imSitu dataset")
                pbar.update(1)
            pbar.close()

    if flag_CP:
        #import ipdb; ipdb.set_trace()
        # Download from: https://link.springer.com/article/10.3758/BF03195584#SecESM1
        CP_a    = pd.read_csv(os.path.join(args.CP_path, "cp2004a.txt"), delim_whitespace=True)
        CP_a    = CP_a.drop(925, axis=0) # Remove erroneous NaN row
        CP_b    = pd.read_csv(os.path.join(args.CP_path, "cp2004b.txt"), delim_whitespace=True)
        CP_b    = CP_b.drop(2311, axis=0)
        with tqdm(total=len(CP_a)) as pbar:
            for key, row in CP_a.iterrows():
                pbar.set_description(f"CP_a dataset")
                pbar.update(1)
                word = clean_word(row['WORD'])
                kf_freq = 10**row['FRQKF']
                imag, mean, concm = row['IMG'], row['MNG'], row['CON']

                emo = row["EMO"]
                pleas = row["PLS"]
                num_mean = row["AMB"]
                fam = row["FAM"]
                eod = row["DEF"]
                full_dict.update(word, "emo", shift_n_scale(emo,-1,6),emo,"CP")
                full_dict.update(word, "pleas", shift_n_scale(pleas,-1,6),pleas,"CP")
                full_dict.update(word, "num_mean", shift_n_scale(emo,-1,6),num_mean,"CP")
                full_dict.update(word, "fam", shift_n_scale(fam,-1,6),fam,"CP")
                full_dict.update(word, "eod", shift_n_scale(eod,-1,6),eod,"CP")

                full_dict.update(word, "conc-m",shift_n_scale(concm,-1,6),concm,"CP")
                full_dict.update(word, "imag-m",shift_n_scale(imag,-1,6),imag,"CP")
                full_dict.update(word, "kf_freq", kf_freq, kf_freq, "CP")
                full_dict.update(word, "mean", shift_n_scale(mean, 0, 10), mean, "CP")
                #

            pbar.close()
        with tqdm(total=len(CP_b)) as pbar:
            for key, row in CP_b.iterrows():
                pbar.set_description(f"CP_b dataset: 2311")
                pbar.update(1)
                word = clean_word(row["WORD"])
                imag, kf_freq, nsyl = row["IMG"], row["LKFR"], row["SYL"]
                fam = row["FAM"]
                kf_freq = 10**kf_freq
                full_dict.update(word, "imag-m",shift_n_scale(imag,0,7) ,imag , "CP")
                full_dict.update(word, "kf_freq", kf_freq, kf_freq, "CP")
                full_dict.update(word, "nsyl", nsyl, nsyl,"CP")
                full_dict.update(word, "fam", shift_n_scale(fam,-1,6), fam,"CP")
                #imag, kf_freq, nsyl
            pbar.close()


    if flag_TWP:
        #import ipdb; ipdb.set_trace()
        # RData files supplied by friendly: https://github.com/friendly/WordPools/tree/master/R
        TWP =   pyreadr.read_r(os.path.join(args.TWP_path, "TWP.RData"))["TWP"]
        with tqdm(total=len(TWP)) as pbar:
            for key, row in TWP.iterrows():
                pbar.set_description(f"TWP Dataset")
                pbar.update(1)
                word = clean_word(row["word"])
                imag, conc, kf_freq = row["imagery"], row["concreteness"], row["frequency"]
                full_dict.update(word,"imag-m",shift_n_scale(imag,-1,6),imag,"TWP")
                full_dict.update(word,"conc-m",shift_n_scale(conc,-1,6),conc,"TWP")
                full_dict.update(word,"kf_freq",kf_freq,kf_freq,"TWP")
            pbar.close()


    if flag_Battig:
        #import ipdb; ipdb.set_trace()
        # RData files supplied by friendly: https://github.com/friendly/WordPools/tree/master/R
        Battig =   pyreadr.read_r(os.path.join(args.Battig_path, "Battig.RData"))["Battig"]
        with tqdm(total=len(Battig)) as pbar:
            for key, row in Battig.iterrows():
                pbar.set_description(f"Battig dataset")
                pbar.update(1)
                word = clean_word(row["word"])
                nsyl, kf_freq = row["syl"], row["frequency"]
                full_dict.update(word,"nsyl",nsyl,nsyl,"Battig")
                full_dict.update(word,"kf_freq",kf_freq,kf_freq,"Battig")
            pbar.close()

    if flag_EViLBERT:
        #import ipdb; ipdb.set_trace()
        # Downloaded from here: https://sapienzanlp.github.io/babelpic/
        EViLBERT = pd.read_csv(os.path.join(args.EViLBERT_path, "visemb-w21ts4lVQA_ext.txt"))

    if flag_Cortese:
        #import ipdb; ipdb.set_trace()
        # Downloaded from: https://link.springer.com/article/10.3758/BF03195585#SecESM1
        Cortese = pd.read_csv(os.path.join(args.Cortese_path, "cortese2004norms.csv"))#, delim_whitespace=True)
        Cortese = Cortese.drop([0,1,2,3,4,5,6,7,8], axis=0)
        Cortese.columns = ['item','rt','sd_rt','num','rating','sd_rating', 'toglia']

        with tqdm(total=len(Cortese)) as pbar:
            for key, row in Cortese.iterrows():
                pbar.set_description(f"Cortese dataset")
                pbar.update(1)
                word = clean_word(row["item"])
                if word != None:
                    rt, sd_rt = assert_float(row["rt"]), assert_float(row["sd_rt"])
                    imagm, imagsd = assert_float(row["rating"]), assert_float(row["sd_rating"])
                    toglia = assert_float(row["toglia"])
                    full_dict.update(word, "rt-m", rt, rt, "Cortese")
                    full_dict.update(word, "rt-sd", sd_rt, sd_rt, "Cortese")
                    full_dict.update(word, "imag-m", shift_n_scale(imagm,-1,6), imagm, "Cortese")
                    full_dict.update(word, "imag-sd", shift_n_scale(imagsd,0,6), imagsd,"Cortese")
                    full_dict.update(word, "toglia",shift_n_scale(toglia,-1,6), toglia,"Cortese")

            pbar.close()

    
    if flag_Reilly:
        #import ipdb; ipdb.set_trace()
        # Request from author, see README (author is kind =) )
        RCortese =       pd.read_csv(os.path.join(args.Reilly_path, "Reilly_LexDbases_Merged_v1.csv"))
        Reilly_comp =   pd.read_excel(os.path.join(args.Reilly_path, "Reilly_Noun_Imageability_Dataset_2013.xls"))
        with tqdm(total=len(RCortese)) as pbar:
            for key, row in RCortese.iterrows():
                pbar.set_description(f"Reilly-Cortese dataset")
                pbar.update(1)
                word = clean_word(row["word"])

                # GLS
                arou, dom, imag, vale, conc, fam = assert_float(row["gls_arousal"]), assert_float(row["gls_dominance"]), assert_float(row["gls_img"]), assert_float(row["gls_valence"]), assert_float(row["gls_cnc"]), assert_float(row["gls_fam"])
                # FORGIVE MY USAGE OF COLONS BUT THERE ARE MANY NORMS HERE I WOULD LIKE THIS SECTION TIDY
                if arou != None: full_dict.update(word, "arou-m", shift_n_scale(arou,0,9) , arou, "GLS-Reilly") # Custom scaling
                if dom  != None: full_dict.update(word, "dom-m" , shift_n_scale(dom ,0,9) , dom , "GLS-Reilly") # Custom scaling
                if imag != None: full_dict.update(word, "imag-m", shift_n_scale(imag,-1,7), imag, "GLS-Reilly")
                if vale != None: full_dict.update(word, "val-m" , shift_n_scale(vale,0,9) , vale, "GLS-Reilly") # Custom scaling
                if conc != None: full_dict.update(word, "conc-m", shift_n_scale(conc,-1,7), conc, "GLS-Reilly")
                if fam  != None: full_dict.update(word, "fam"   , shift_n_scale(fam ,-1,7), fam , "GLS-Reilly")

                # KUP
                pos, nphon, nsyl = row["kup_pos"], assert_float(row["kup_nphon"]), assert_float(row["kup_nsyll"])
                if pos in kup_reilly_pos_trans:
                    pos = pos.lower() # All the entries are already in my chosen format when lowered
                else:
                    pos = None
                if pos   != None: full_dict.update(word, "pos"  , pos, pos     , "KUP-Reilly")
                if nphon != None: full_dict.update(word, "nphon", nphon, nphon , "KUP-Reilly")
                if nsyl  != None: full_dict.update(word, "nsyl" , nsyl, nsyl   , "KUP-Reilly")

                # ELP
                rt_lexdec, rt_nam = assert_float(row["elp_lex_dec_rt"]), assert_float(row["elp_naming_rt"])
                sem_dens, sem_neig, sem_div = row["elp_sem_density"], row["elp_sem_neighbors"], row["elp_sem_diversity"]
                if rt_lexdec != None: full_dict.update(word, "rt-lex_dec", rt_lexdec, rt_lexdec, "ELP-Reilly")
                if rt_nam    != None: full_dict.update(word, "rt-nam"    , rt_nam   , rt_nam  , "ELP-Reilly")
                if sem_neig == sem_neig:
                    sem_neig = str.replace(sem_neig,',','')
                sem_dens, sem_neig, sem_div = assert_float(sem_dens), assert_float(sem_neig), assert_float(sem_div)
                if sem_dens != None: full_dict.update(word, "sem_dens", shift_n_scale(sem_dens,0,1), sem_dens, "ELP-Reilly") # range:0.08-0.724
                if sem_neig != None: full_dict.update(word, "sem_neig" , shift_n_scale(sem_neig,0,1), sem_neig, "ELP-Reilly") # range:0-9931
                if sem_div  != None: full_dict.update(word, "sem_div"  , shift_n_scale(sem_div,0,2.5) , sem_div,  "ELP-Reilly") # range: 0.175 - 2.413

                # BRYS
                conc, pos = assert_float(row["brys_concreteness"]), row["brys_pos"]
                if pos in kup_reilly_pos_trans:
                    pos = pos.lower()
                else:
                    pos = None
                if pos  != None: full_dict.update(word, "pos"   , pos , pos   , "BRYS-Reilly")
                if conc != None: full_dict.update(word, "conc-m", shift_n_scale(conc,-1,4), conc  , "BRYS-Reilly")

                # LNC
                audi, gust, hapt, interoc, olfact, foot_leg, hand_arm, head, mouth, torso, dom_sense = assert_float(row["lnc_auditory"]), assert_float(row["lnc_gustatory"]), assert_float(row["lnc_haptic"]), assert_float(row["lnc_interoceptive"]), assert_float(row["lnc_olfactory"]), assert_float(row["lnc_footleg"]), assert_float(row["lnc_handarm"]), assert_float(row["lnc_head"]), assert_float(row["lnc_mouth"]), assert_float(row["lnc_torso"]), row["lnc_domsense"]
                if dom_sense == dom_sense:
                    dom_sense = str.replace(dom_sense,'_','-').lower()
                if audi != None: full_dict.update(word, "lnc-audi", shift_n_scale(audi,0,5), audi, "LNC-Reilly") # Range: 0-5
                if gust != None: full_dict.update(word, "lnc-gust", shift_n_scale(gust,0,5), gust, "LNC-Reilly") # Range: 0-5
                if hapt != None: full_dict.update(word, "lnc-hapt", shift_n_scale(hapt,0,5), hapt, "LNC-Reilly") # Range: 0-5
                if interoc  != None: full_dict.update(word, "lnc-interoc",  shift_n_scale(interoc,0,5), interoc, "LNC-Reilly")
                if olfact   != None: full_dict.update(word, "lnc-olfact",   shift_n_scale(olfact,0,5), olfact, "LNC-Reilly")
                if foot_leg != None: full_dict.update(word, "lnc-foot_leg", shift_n_scale(foot_leg,0,5), foot_leg, "LNC-Reilly")
                if hand_arm != None: full_dict.update(word, "lnc-hand_arm", shift_n_scale(hand_arm,0,5), hand_arm, "LNC-Reilly")
                if head     != None: full_dict.update(word, "lnc-head",     shift_n_scale(head,0,5), head, "LNC-Reilly")
                if mouth    != None: full_dict.update(word, "lnc-mouth",    shift_n_scale(mouth,0,5), mouth, "LNC-Reilly")
                if torso    != None: full_dict.update(word, "lnc-torso",    shift_n_scale(torso,0,5), torso, "LNC-Reilly")
                if dom_sense != None: full_dict.update(word,"lnc-dom_sense",dom_sense, dom_sense, "LNC-Reilly")

                # WAR
                val, arou, dom = assert_float(row["war_valence"]), assert_float(row["war_arousal"]), assert_float(row["war_dominance"])
                if val  != None: full_dict.update(word, "val-m" , shift_n_scale(val ,0,9), val , "WAR-Reilly") # Custom scaling
                if arou != None: full_dict.update(word, "arou-m", shift_n_scale(arou,0,9), arou, "WAR-Reilly") # Custom scaling
                if dom  != None: full_dict.update(word, "dom-m" , shift_n_scale(dom ,0,9), dom , "WAR-Reilly") # Custom scaling

            pbar.close()
        #import ipdb; ipdb.set_trace()
        with tqdm(total=len(Reilly_comp)) as pbar:
            for key, row in Reilly_comp.iterrows():
                pbar.set_description(f"Reilly Imagability dataset")
                pbar.update(1)
                word = clean_word(row["WORD"])
                concm, imagm = assert_float(row["CNC"]), assert_float(row["IMG"])
                kf_freq = assert_float(row["KFFRQ"])
                nphon, nsyl = assert_float(row["nphon"]), assert_float(row["NSYL"])
                rt = assert_float(row["I_Mean_RT"])
                fam = assert_float(row["FAM"])

                if concm is not None:
                    full_dict.update(word, "conc-m", shift_n_scale(concm,-100,600), concm,"Reilly")
                if imagm is not None:
                    full_dict.update(word, "imag-m", shift_n_scale(imagm,-100,600), imagm,"Reilly")
                if kf_freq is not None:
                    full_dict.update(word, "kf_freq", shift_n_scale(kf_freq,0,1), kf_freq,"Reilly")
                if nphon is not None:
                    full_dict.update(word, "nphon", shift_n_scale(nphon,0,1), nphon,"Reilly")
                if nsyl is not None:
                    full_dict.update(word, "nsyl", shift_n_scale(nsyl,0,1), nsyl,"Reilly")
                if rt is not None:
                    full_dict.update(word, "rt-m", shift_n_scale(rt,0,1), rt,"Reilly")
                if fam is not None:
                    full_dict.update(word, "fam", shift_n_scale(fam,-100,600), fam,"Reilly")

            pbar.close()

    if flag_MM_imgblty:
        #import ipdb; ipdb.set_trace()
        # Recently released by author (who is kind)
        MM_imgblty = pd.read_csv(os.path.join(args.MM_imgblty_path, "corpus.csv"))
        with tqdm(total=len(MM_imgblty)) as pbar:
            for key, row in MM_imgblty.iterrows():
                pbar.set_description(f"Multimodal-Imagability dataset")
                pbar.update(1)
                word = clean_word(row["Word"])
                mm_vis, mm_txt, mm_pho, mm_all = row["Visual"], row["Textual"], row["Phonetic"], row["All"]
                full_dict.update(word, "mm_vis", shift_n_scale(mm_vis, 0, 100), mm_vis, "mm_imgblty")
                full_dict.update(word, "mm_txt", shift_n_scale(mm_txt, 0, 100), mm_txt, "mm_imgblty")
                full_dict.update(word, "mm_pho", shift_n_scale(mm_pho, 0, 100), mm_pho, "mm_imgblty")
                full_dict.update(word, "mm_all", shift_n_scale(mm_all, 0, 100), mm_all, "mm_imgblty")
            pbar.close()

    if flag_sianpar_indo:
        #import ipdb; ipdb.set_trace()
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5138238/
        sianpar_indo = pd.read_excel(os.path.join(args.sianpar_indo_path, "Data_Sheet_1.XLSX")) 
        with tqdm(total=len(sianpar_indo)) as pbar:
            for key, row in sianpar_indo.iterrows():
                pbar.set_description(f"Sianpar Indonesian word dataset")
                pbar.update(1)
                word = clean_word(row["Words (English)"])
                concm, concsd = row["ALL_Concreteness_Mean"], row["ALL_Concreteness_SD"]
                aroum, arousd = row["ALL_Arousal_Mean"], row["ALL_Arousal_SD"]
                valm, valsd = row["ALL_Valence_Mean"], row["ALL_Valence_SD"]
                domm, domsd = row["ALL_Dominance_Mean"], row["ALL_Dominance_SD"]
                predm, predsd = row["ALLPredictability_Mean"], row["ALL_Predictability_SD"]

                # Normalise 1-7 -> 0-1
                full_dict.update(word, "conc-m", shift_n_scale(concm,-1,6), concm, "sianpar_indo")
                full_dict.update(word, "conc-sd", shift_n_scale(concsd,0,6), concsd, "sianpar_indo")

                # assumed normalisation of 0-8 -> 0-1??
                full_dict.update(word, "arou-m", shift_n_scale(aroum,0,8), aroum, "sianpar_indo")
                full_dict.update(word, "arou-sd", shift_n_scale(arousd,0,8), arousd, "sianpar_indo")
                full_dict.update(word, "val-m", shift_n_scale(valm,0,8), valm, "sianpar_indo")
                full_dict.update(word, "val-sd", shift_n_scale(valsd,0,8), valsd, "sianpar_indo")
                full_dict.update(word, "dom-m", shift_n_scale(domm,0,8), domm, "sianpar_indo")
                full_dict.update(word, "dom-sd", shift_n_scale(domsd,0,8), domsd, "sianpar_indo")
                full_dict.update(word, "pred-m", shift_n_scale(predm,0,8), predm, "sianpar_indo")
                full_dict.update(word, "pred-sd", shift_n_scale(predsd,0,8), predsd, "sianpar_indo")
            pbar.close()

    if flag_yee_chinese:
        #import ipdb; ipdb.set_trace()
        # https://www.ebi.ac.uk/biostudies/studies/S-EPMC5367816?xr=true
        yee_chinese = pd.read_excel(os.path.join(args.yee_chinese_path, "pone.0174569.s001.xlsx")) 
        yee_chinese2= pd.read_excel(os.path.join(args.yee_chinese_path, "pone.0174569.s002.xlsx"))
        with tqdm(total=len(yee_chinese)) as pbar:
            for key, row in yee_chinese.iterrows():
                pbar.set_description(f"Yee's Chinese word norms 1")
                pbar.update(1)
                word = clean_word(row["English translation"])
                valm, aroum, imagm, concm = row["Valence"], row["Arousal"], row["Imageability"], row["Concreteness"]
                valm, aroum, imagm, concm = assert_float(valm), assert_float(aroum), assert_float(imagm), assert_float(concm)
                if valm != None:
                    # All norms are in range 1-5, scaling to 0-1
                    full_dict.update(word, "val-m", shift_n_scale(valm,-1,4), valm, "yee_chinese")
                if aroum != None:
                    full_dict.update(word, "arou-m", shift_n_scale(aroum,-1,4), aroum, "yee_chinese")
                if imagm != None:
                    full_dict.update(word, "imag-m", shift_n_scale(imagm,-1,4), imagm, "yee_chinese")
                if concm != None:
                    full_dict.update(word, "conc-m", shift_n_scale(concm,-1,4), concm, "yee_chinese")
            pbar.close()
        #with tqdm(total=len(yee_chinese2)) as pbar:
        #    for key, row in yee_chinese2.iterrows():
        #        pbar.set_description(f"Yee's Chinese word norms 2")
        #        pbar.update(1)
        #    pbar.close()

    if flag_megahr_crossling:
        #import ipdb; ipdb.set_trace()
        # Predicted word norms from the hard work of these guys: https://github.com/clarinsi/megahr-crossling
        megahr_wf   = pd.read_csv(os.path.join(args.megahr_crossling_path, "megahr.en"), delim_whitespace=True) # a file sorted by decreasing word frequency
        first_ele = megahr_wf.columns
        first_ele = [ele for ele in first_ele]
        first_ele[1], first_ele[2] = float(first_ele[1]), float(first_ele[2])
        megahr_wf.columns = ['word', 'conc', 'imag']
        megahr_wf = megahr_wf.append({megahr_wf.columns[i]:first_ele[i] for i in range(3)}, ignore_index=True)
        #megahr_conc = pd.read_csv(os.path.join(args.megahr_crossling_path, "megahr.en.sort.c")) # a file sorted by concretenss predictions
        #megahr_imag = pd.read_csv(os.path.join(args.megahr_crossling_path, "megahr.en.sort.i")) # a file sorted by imageability predictions
        with tqdm(total=len(megahr_wf)) as pbar:
            for key, row in megahr_wf.iterrows():
                pbar.set_description(f"MEGAHR Cross Lingual Word norms:")
                pbar.update(1)
                word = clean_word(row["word"])
                # As mentioned on the official GitHub, word predictions should be 1-5, but can sometimes go beyond. I will round overpredictions down to 5
                concm, imagm = row['conc'], row['imag']
                full_dict.update(word, "conc-m", shift_n_scale(min(5,concm),-1,4), concm,"megahr")
                full_dict.update(word, "imag-m", shift_n_scale(min(5,imagm),-1,4), imagm,"megahr")


            pbar.close()

    if flag_glasgow:
        #import ipdb; ipdb.set_trace()
        # https://osf.io/ud367/
        glasgow = pd.read_excel(os.path.join(args.glasgow_path, "GlasgowNorms.xlsx"))
        with tqdm(total=len(glasgow)) as pbar:
            for key, row in glasgow.iterrows():
                pbar.set_description(f"Glasgow word norms")
                # Scaling is unclear. IMAG, FAM and CNC all in range 1-7, other 3 however are in range 1-8.3ish?? 
                pbar.update(1)
                word = row["Unnamed: 0"]
                if type(word) == str:
                    word = clean_word(word)
                    aroum = assert_float(row['AROU'])
                    valm = assert_float(row['VAL'])
                    domm = assert_float(row['DOM'])
                    concm = assert_float(row['CNC'])
                    fam = assert_float(row['FAM'])
                    imagm = assert_float(row['IMAG'])
                    # Arousal, valence and dominance scaling unclear, so will simply divide by 9
                    if aroum != None:   
                        full_dict.update(word, "arou-m", shift_n_scale(aroum,0,9), aroum, "glasgow")
                    if valm != None:   
                        full_dict.update(word, "val-m", shift_n_scale(valm,0,9), valm, "glasgow")
                    if domm != None:   
                        full_dict.update(word, "dom-m", shift_n_scale(domm,0,9), domm, "glasgow")
                    if concm != None:   
                        full_dict.update(word, "conc-m", shift_n_scale(concm,-1,6), concm, "glasgow")
                    if fam != None:   
                        full_dict.update(word, "fam", shift_n_scale(fam,-1,6), fam, "glasgow")
                    if imagm != None:   
                        full_dict.update(word, "imag-m", shift_n_scale(imagm,-1,6), imagm, "glasgow")
            pbar.close()
    #import ipdb; ipdb.set_trace()
    return full_dict



########################################################################
# Util functions
########################################################################
def _resolve_path(path):
    """
    Resolve the relative path of this main.py
    Inputs:
        path: path to the file relative to main
    """
    return(Path(__file__).parent.parent.resolve() / path)


def _if_main(args):
    if __name__ == "__main__":
        return args
    else:
        args = _parse()
        return args

def _parse():
    parser = argparse.ArgumentParser()
    # Which datasets
    parser.add_argument_group("Main running arguments")
    parser.add_argument("--purpose", type=str, default="explore_dsets",
            choices=["explore_dsets", "get_concrete_words", "word_to_MRC", "word_to_norms"],
            help="how to run this script as main")

    parser.add_argument_group("Explore datasets options")
    parser.add_argument("--all", action="store_true", help="Load all datasets")
    parser.add_argument("--dsets", type=str, nargs="+", default=[],
            choices=["MT40k", "CSLB", "USF", "MRC", "SimLex999", "Vinson", "McRae", "SimVerb", "imSitu", "CP", "TWP", "Battig", "EViLBERT", "Cortese", "Reilly", "MM_imgblty", "sianpar_indo", "yee_chinese", "megahr_crossling", "glasgow"], 
            help="If not all, which datasets")

    parser.add_argument_group('Data paths')
    parser.add_argument("--MT40k_path", type=str, default="data/MT40k/Concreteness_ratings_Brysbaert_et_al_BRM.txt", help="Path to MT40k text file")
    parser.add_argument("--CSLB_path", type=str, default="data/CSLB_prop_norms", help="Path to the CSLB data directory")
    parser.add_argument("--USF_path", type=str, default="data/USF", help="Path to USF root dir")
    parser.add_argument("--MRC_path", type=str, default="data/MRC", help="Path to MRC root dir")
    parser.add_argument("--SimLex999_path", type=str, default="data/SimLex-999", help="Path to SimLex-999 gold standard root dir")
    parser.add_argument("--Vinson_path", type=str, default="data/Vinson/Vinson-BRM-2008", help="Path to Vinson root dir")
    parser.add_argument("--McRae_path", type=str, default="data/McRae/McRae-BRM-InPress", help="Path to McRae root dir")
    parser.add_argument("--SimVerb_path", type=str, default="data/SimVerb", help="Path to SimVerb root dir")
    parser.add_argument("--imSitu_path", type=str, default="data/imSitu", help="Path to imSitu root dir")
    parser.add_argument("--CP_path", type=str, default="data/PYM_and_CP/Clark-BRMIC-2004", help="Clark and Paivio 2004 word norms (extended)")   
    parser.add_argument("--TWP_path", type=str, default="data/TWP", help="Path to Toronto Word Pool root dir")
    parser.add_argument("--Battig_path", type=str, default="data/Battig", help="Path to Battig word norm root dir")
    parser.add_argument("--EViLBERT_path", type=str, default="data/BabelPic/EViLBERT", help="Path to EViLBERT embeddings")
    parser.add_argument("--Cortese_path", type=str, default="data/2004_cortese/Cortese-BRMIC-2004", help="Path to Cortese 2004 monosyllabic concepts")
    parser.add_argument("--Reilly_path", type=str, default="data/Reilly", help="Path to Cortese 2004 monosyllabic concepts")
    parser.add_argument("--MM_imgblty_path", type=str, default="data/mm_imgblty", help="Path to multimodal imageability corpus")
    parser.add_argument("--sianpar_indo_path", type=str, default="data/sianpar_indonesian", help="Path to Sianpar's indonesian word norm corpus")
    parser.add_argument("--yee_chinese_path", type=str, default="data/yee_chinese", help="Yee's valence, arousal, concs etc.. norms word norm corpus")
    parser.add_argument("--megahr_crossling_path", type=str, default="data/megahr_crossling", help="MEGAHR cross-lingual word predictions")
    parser.add_argument("--glasgow_path", type=str, default="data/glasgow", help="")

    if __name__ == "__main__":    
        args = parser.parse_args()
    else:
        args, unknown = parser.parse_known_args()
    
    # Resolve all paths supplied
    args.MT40k_path     = _resolve_path(args.MT40k_path)
    args.CSLB_path      = _resolve_path(args.CSLB_path)
    args.USF_path       = _resolve_path(args.USF_path)
    args.MRC_path       = _resolve_path(args.MRC_path)
    args.SimLex999_path = _resolve_path(args.SimLex999_path)
    args.Vinson_path    = _resolve_path(args.Vinson_path)
    args.McRae_path     = _resolve_path(args.McRae_path)
    args.SimVerb_path   = _resolve_path(args.SimVerb_path)
    args.imSitu_path    = _resolve_path(args.imSitu_path)
    args.CP_path        = _resolve_path(args.CP_path)
    args.TWP_path       = _resolve_path(args.TWP_path)
    args.Battig_path    = _resolve_path(args.Battig_path)
    args.EViLBERT_path  = _resolve_path(args.EViLBERT_path)
    args.Cortese_path   = _resolve_path(args.Cortese_path)
    args.Reilly_path    = _resolve_path(args.Reilly_path)
    args.MM_imgblty_path= _resolve_path(args.MM_imgblty_path)
    args.sianpar_indo_path  = _resolve_path(args.sianpar_indo_path)
    args.yee_chinese_path   = _resolve_path(args.yee_chinese_path)
    args.megahr_crossling_path  = _resolve_path(args.megahr_crossling_path)
    args.glasgow_path  = _resolve_path(args.glasgow_path)

    return args


#args = _parse()
#if args.purpose == "explore_dsets":
#    full_dict = explore_dsets(args)
#    myutils.save_pickle(full_dict, os.path.join(os.path.dirname(__file__), "misc", "all_norms.pickle"))

########################################################################
# Main
########################################################################
if __name__ == "__main__":
    args = _parse()
    #norm_dict_path = os.path.join( "/home/jumperkables/kable_management/projects/a_vs_c/misc" , "all_norms.pickle")
    #norm_dict = myutils.load_norms_pickle(norm_dict_path)
    #breakpoint()
    if args.purpose == "explore_dsets":
        full_dict = explore_dsets(args)
        myutils.save_pickle(full_dict, os.path.join(os.path.dirname(__file__), "misc", "all_norms.pickle"))
    #elif args.purpose == "get_concrete_words":
    #    conc_words = get_concrete_words(args)
    #elif args.purpose == "word_to_MRC":
    #    word_to_MRC(args) 
    #elif args.purpose == "word_to_norms":
    #    word_to_norms(args)
    #else:
    #    print(f"Purpose: '{args.purpose}' is not recognised")
    #    sys.exit()
