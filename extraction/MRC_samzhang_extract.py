__author__ = "samzhang111"
# https://github.com/samzhang111/mrc-psycholinguistics
# Adapted by Jumperkables

from .MRC_samzhang_wordmodel import Word

#from sqlalchemy import create_engine
#from sqlalchemy.orm import sessionmaker


class MRC_Db():
    def __init__(self, path):
        """
        This object was designed to provide a SQLAlchemy session object. This still can. I have altered it to give a dictionary object instead
        """
        #engine = create_engine('sqlite:///{name}.db'.format(name='mrc2'))

        # uncomment to drop table first:
        #Word.__table__.drop(engine, checkfirst=True)
        #Word.__table__.create(engine, checkfirst=True)
        
        self.MRC_dict = {}
        self.mrc_keys = [
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
        #Session = sessionmaker(bind=engine)
        #self.session = Session()
        
        f = open(path, 'r')
        i = 0
        for line in f:
            line = line.strip()
            
            # see wordmodel.py for blurbs of each variable
            # or even better, mrc2.doc
            
            word, phon, dphon, stress = line[51:].split('|')
        
            w = Word(
                    wid = i,
                    nlet = int(line[0:2]),
                    nphon = int(line[2:4]),
                    nsyl = int(line[4]),
                    kf_freq = int(line[5:10]),
                    kf_ncats = int(line[10:12]),
                    kf_nsamp = int(line[12:15]),
                    tl_freq = int(line[15:21]),
                    brown_freq = int(line[21:25]),
                    fam = int(line[25:28]),
                    conc = int(line[28:31]),
                    imag = int(line[31:34]),
                    meanc = int(line[34:37]),
                    meanp = int(line[37:40]),
                    aoa = int(line[40:43]),
                    tq2 = line[43],
                    wtype = line[44],
                    pdwtype = line[45],
                    alphasyl = line[46],
                    status = line[47],
                    var = line[48],
                    cap = line[49],
                    irreg = line[50],
                    word=word,
                    phon=phon,
                    dphon=dphon,
                    stress=stress)
            w = w.word
            # Clean these words
            self.MRC_dict[word.lower()] = w
            #self.session.add(w)
            i+=1
        
        #    if i%1000 == 0:
        #        print(i, w)
        #        self.session.commit()
        #self.session.commit()
