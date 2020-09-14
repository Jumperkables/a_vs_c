__author__ = "samzhang111"
# https://github.com/samzhang111/mrc-psycholinguistics
# Adapted by Jumperkables

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Integer

Base = declarative_base()

class Word(Base):
    __tablename__ = 'word'

    # word id (row in the database)
    wid = Column(Integer, primary_key=True)
    # number of letters in word
    nlet = Column(Integer)
    # number of phonemes in word
    nphon = Column(Integer)
    # number of syllables in word
    nsyl = Column(Integer)
    # Kucera and Francis written frequency
    kf_freq = Column(Integer)
    # Kucera and Francis number of categories
    kf_ncats = Column(Integer)
    # K&F number of samples
    kf_nsamp = Column(Integer)
    # Thorndike-Lorge frequency
    tl_freq = Column(Integer)
    # Brown verbal frequency
    brown_freq = Column(Integer)
    # Familiarity
    fam = Column(Integer)
    # Concreteness
    conc = Column(Integer)
    # Imagery
    imag = Column(Integer)
    # Mean Colerado Meaningfulness
    meanc = Column(Integer)
    # Mean Pavio Meaningfulness
    meanp = Column(Integer)
    # Age of Acquisition
    aoa = Column(Integer)
    # Type
    tq2 = Column(String)
    # Part of speech
    wtype = Column(String)
    # PD Part of speech
    pdwtype = Column(String)
    # Alphasyllable
    alphasyl = Column(String)
    # Status
    status = Column(String)
    # Variant phoneme
    var = Column(String)
    # Written capitalized
    cap = Column(String)
    # Irregular plural
    irreg = Column(String)
    # The actual word
    word = Column(String)
    # Phonetic transcription
    phon = Column(String)
    # Edited phonetic transcription
    dphon = Column(String)
    # Stress pattern
    stress = Column(String)

    def __init__(self, 
        wid,
        nlet,
        nphon,
        nsyl,
        kf_freq,
        kf_ncats,
        kf_nsamp,
        tl_freq,
        brown_freq,
        fam,
        conc,
        imag,
        meanc,
        meanp,
        aoa,
        tq2,
        wtype,
        pdwtype,
        alphasyl,
        status,
        var,
        cap,
        irreg,
        word,
        phon,
        dphon,
        stress):
        keys = [
            "id",
            "nlet",
            "nphon",
            "nsyl",
            "kf_freq",
            "kf_ncats",
            "kf_nsamp",
            "tl_freq",
            "brown_freq",
            "fam",
            "conc",
            "imag",
            "meanc",
            "meanp",
            "aoa",
            "tq2",
            "wtype",
            "pdwtype",
            "alphasyl",
            "status",
            "var",
            "cap",
            "irreg",
            "word",
            "phon",
            "dphon",
            "stress"
        ] 
        values = [
            wid,
            nlet,
            nphon,
            nsyl,
            kf_freq,
            kf_ncats,
            kf_nsamp,
            tl_freq,
            brown_freq,
            fam,
            conc,
            imag,
            meanc,
            meanp,
            aoa,
            tq2,
            wtype,
            pdwtype,
            alphasyl,
            status,
            var,
            cap,
            irreg,
            word,
            phon,
            dphon,
            stress
        ]
        l = len(values)
        assert(l == len(keys))
        self.word = {keys[i]:values[i] for i in range(l)} 
