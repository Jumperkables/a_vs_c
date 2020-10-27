import os, sys
sys.path.append( os.path.dirname(os.path.dirname(os.path.abspath(__file__))) )

import myutils
import word_norms
from avsd.dataloader import restrict_vocab





if __name__ == "__main__":
    w2i_path = os.path.join( os.path.dirname(os.path.abspath(__file__)), "tvqa_modality_bias/data/cache/word2idx.pickle" )
    tvqa_word2idx = myutils.load_pickle( w2i_path )
    word2norm = word_norms.word_to_MRC(None)

    # Conc lt 300
    #conclt300 = restrict_vocab(tvqa_word2idx, word2norm, ["conc-lt-500"], ["<pad>", "<unk>", "<eos>"], '<unk>')
    #myutils.save_pickle(conclt300, os.path.join( os.path.dirname(w2i_path) , "word2idx_conclt500.pickle") )

    # Conc gt 300
    #concgt300 = restrict_vocab(tvqa_word2idx, word2norm, ["conc-gt-500"], ["<pad>", "<unk>", "<eos>"], '<unk>')
    #myutils.save_pickle(concgt300, os.path.join( os.path.dirname(w2i_path) , "word2idx_concgt500.pickle") )

    ## Imag lt 500
    imaglt500 = restrict_vocab(tvqa_word2idx, word2norm, ["imag-lt-500"], ["<pad>", "<unk>", "<eos>"], '<unk>')
    myutils.save_pickle(imaglt500, os.path.join( os.path.dirname(w2i_path) , "word2idx_imaglt500.pickle") )

    # Imag gt 500
    imaggt500 = restrict_vocab(tvqa_word2idx, word2norm, ["imag-gt-500"], ["<pad>", "<unk>", "<eos>"], '<unk>')
    myutils.save_pickle(imaggt500, os.path.join( os.path.dirname(w2i_path) , "word2idx_imaggt500.pickle") )

    print("Loaded")
