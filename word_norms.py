__author__ = "Jumperkables"

import myutils
import pandas as pd
import argparse
import os, sys
import pyreadr
import numpy as np
from pathlib import Path

# Project imports
from extraction.USF_teonbrooks_free import USF_Free
from extraction.MRC_samzhang_extract import MRC_Db
#from extraction.MRC




########################################################################
# Running functions
########################################################################
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




def explore_dsets(args):
    args = _if_main(args)
    print(f"Loading {'all ' if args.all else ''}{len(args.dsets)} dataset{'s' if len(args.dsets) > 1 else ''}: {args.dsets}")
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

    if flag_MT40k:
        # Download from here: http://crr.ugent.be/papers/Concreteness_ratings_Brysbaert_et_al_BRM.txt
        MT40k = pd.read_csv(args.MT40k_path, sep="\t") 
        print(f"MT40k: \n {MT40k} \n\n")

    if flag_CSLB:
        # Download from: http://www.csl.psychol.cam.ac.uk/propertynorms/ 
        CSLB_norms          = pd.read_csv(os.path.join(args.CSLB_path, "norms.dat"), sep="\t")  
        CSLB_feature_matrix = pd.read_csv(os.path.join(args.CSLB_path, "feature_matrix.dat"), sep="\t")
        print(f"CSLB Norms: \n {CSLB_norms} \n\n")
        print(f"CSLB Feature Matrix: \n {CSLB_feature_matrix} \n\n")

    if flag_USF:
        # Code and txt provided by teonbrooks: https://github.com/teonbrooks/free_association
        USF_free_assoc  = USF_Free(os.path.join(args.USF_path, "teonbrooks/free_association.txt"))

        print(f"USF Database: \n Definitions: {USF_free_assoc.definitions} \n\n")

    if flag_MRC:
        # Code and MRC file provided by samzhang111: https://github.com/samzhang111/mrc-psycholinguistics
        MRC_dict    = MRC_Db(os.path.join(args.MRC_path, "samzhang111/mrc2.dct")) 
        MRC_dict    = MRC_dict.MRC_dict # We keep the class structure intact incase we want the SQLAlchemy session later
        
    if flag_SimLex999:
        # Data from Felix Hill: https://fh295.github.io/simlex.html
        SimLex999   = pd.read_csv(os.path.join(args.SimLex999_path, "SimLex-999.txt"), sep="\t")  

    if flag_Vinson:
        # Download from: https://static-content.springer.com/esm/art%3A10.3758%2FBRM.40.1.183/MediaObjects/Vinson-BRM-2008a.zip
        Vinson_word_cats    = pd.read_csv(os.path.join(args.Vinson_path, "word_categories.txt"), sep="\t")
        Vinson_features     = pd.read_csv(os.path.join(args.Vinson_path, "feature_list_and_types.txt"), sep="\t")
        Vinson_weight_mat0  = pd.read_csv(os.path.join(args.Vinson_path, "feature_weight_matrix_1_256.txt"), sep="\t")
        Vinson_weight_mat1  = pd.read_csv(os.path.join(args.Vinson_path, "feature_weight_matrix_257_456.txt"), sep="\t")
    
    if flag_McRae:
        # Download from: https://static-content.springer.com/esm/art%3A10.3758%2FBF03192726/MediaObjects/McRae-BRM-2005.zip
        # Information in the readme
        McRae_CONCS_brm     = pd.read_csv(os.path.join(args.McRae_path, "CONCS_brm.txt"), sep="\t")
    
    if flag_SimVerb:
        # Download from: https://github.com/benathi/word2gm/tree/master/evaluation_data/simverb/data
        # There is more nuance to this data yet
        SimVerb_data    = pd.read_csv(os.path.join(args.SimVerb_path, "SimVerb-3500.txt"), sep="\t")

    if flag_imSitu:
        # Download imSitu here: https://public.ukp.informatik.tu-darmstadt.de/coling18-multimodalSurvey/
        # Properly separate word from embedding values
        imSitu_verb_embeddings  = pd.read_csv(os.path.join(args.imSitu_path, "imSitu_verbs_averagedEmbeddings.w2vt"), sep="\t")

    if flag_CP:
        # Download from: https://link.springer.com/article/10.3758/BF03195584#SecESM1
        CP_a    = pd.read_csv(os.path.join(args.CP_path, "cp2004a.txt"), sep="\t")
        CP_b    = pd.read_csv(os.path.join(args.CP_path, "cp2004b.txt"), sep="\t")
        
    if flag_TWP:
        # RData files supplied by friendly: https://github.com/friendly/WordPools/tree/master/R
        TWP =   pyreadr.read_r(os.path.join(args.TWP_path, "TWP.RData"))

    if flag_Battig:
        # RData files supplied by friendly: https://github.com/friendly/WordPools/tree/master/R
        Battig =   pyreadr.read_r(os.path.join(args.Battig_path, "Battig.RData"))

    if flag_EViLBERT:
        # Downloaded from here: https://sapienzanlp.github.io/babelpic/
        EViLBERT = pd.read_csv(os.path.join(args.EViLBERT_path, "visemb-w21ts4lVQA_ext.txt"))

    if flag_Cortese:
        # Downloaded from: https://link.springer.com/article/10.3758/BF03195585#SecESM1
        Cortese = pd.read_csv = pd.read_csv(os.path.join(args.Cortese_path, "cortese2004norms.csv"))

    if flag_Reilly:
        # Request from author, see README (author is kind =) )
        Cortese =       pd.read_csv(os.path.join(args.Reilly_path, "Reilly_LexDbases_Merged_v1.csv"))
        Reilly_comp =   pd.read_excel(os.path.join(args.Reilly_path, "Reilly_Noun_Imageability_Dataset_2013.xls"))
        import ipdb; ipdb.set_trace()

    if flag_MM_imgblty:
        # Recently released by author (who is kind)
        MM_imgblty = pd.read_csv(os.path.join(args.MM_imgblty_path, "corpus.csv"))

    return None



########################################################################
# Util functions
########################################################################
def _resolve_path(path):
    """
    Resolve the relative path of this main.py
    Inputs:
        path: path to the file relative to main
    """
    return(Path(__file__).parent.resolve() / path)


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
    parser.add_argument("--dsets", type=str, nargs="+",
            choices=["MT40k", "CSLB", "USF", "MRC", "SimLex999", "Vinson", "McRae", "SimVerb", "imSitu", "CP", "TWP", "Battig", "EViLBERT", "Cortese", "Reilly", "MM_imgblty"], 
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

    return args



########################################################################
# Main
########################################################################
if __name__ == "__main__":
    args = _parse()
    if args.purpose == "explore_dsets":
        explore_dsets(args)
    elif args.purpose == "get_concrete_words":
        conc_words = get_concrete_words(args)
    elif args.purpose == "word_to_MRC":
        word_to_MRC(args) 
    elif args.purpose == "word_to_norms":
        word_to_norms(args)
    else:
        print(f"Purpose: '{args.purpose}' is not recognised")
        sys.exit()
