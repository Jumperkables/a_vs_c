__author__ = "Jumperkables"

import utils
import pandas as pd
import argparse
import os, sys
import pyreadr
from pathlib import Path

# Project imports
from extraction.USF_teonbrooks_free import USF_Free
from extraction.MRC_samzhang_extract import MRC_Db
#from extraction.MRC



def resolve_path(path):
    """
    Resolve the relative path of this main.py
    Inputs:
        path: path to the file relative to main
    """
    return(Path(__file__).parent.resolve() / path)


def compile_wordlist(args):
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
        import ipdb; ipdb.set_trace()
        
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

    return(True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Which datasets
    parser.add_argument("--all", action="store_true", help="Load all datasets")
    parser.add_argument("--dsets", type=str, nargs="+", 
            choices=["MT40k", "CSLB", "USF", "MRC", "SimLex999", "Vinson", "McRae", "SimVerb", "imSitu", "CP", "TWP", "Battig", "EViLBERT", "Cortese", "Reilly", "MM_imgblty"], 
            help="If not all, which datasets")
    # Data paths
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

    args = parser.parse_args()

    # Resolve all paths supplied
    args.MT40k_path     = resolve_path(args.MT40k_path)
    args.CSLB_path      = resolve_path(args.CSLB_path)
    args.USF_path       = resolve_path(args.USF_path)
    args.MRC_path       = resolve_path(args.MRC_path)
    args.SimLex999_path = resolve_path(args.SimLex999_path)
    args.Vinson_path    = resolve_path(args.Vinson_path)
    args.McRae_path     = resolve_path(args.McRae_path)
    args.SimVerb_path   = resolve_path(args.SimVerb_path)
    args.imSitu_path    = resolve_path(args.imSitu_path)
    args.CP_path        = resolve_path(args.CP_path)
    args.TWP_path       = resolve_path(args.TWP_path)
    args.Battig_path    = resolve_path(args.Battig_path)
    args.EViLBERT_path  = resolve_path(args.EViLBERT_path)
    args.Cortese_path   = resolve_path(args.Cortese_path)
    args.Reilly_path    = resolve_path(args.Reilly_path)
    args.MM_imgblty_path= resolve_path(args.MM_imgblty_path)

    compile_wordlist(args)
