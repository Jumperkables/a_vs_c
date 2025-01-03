# Abstract-vs-Concrete Guided Multimodal Machine Learning...
![Alt text](misc/imgs/a-c_scale.png?raw=true "Abstract -> Concrete")
#### ...and many other psycholinguistic norms for good measure

## Noura, Alistair and Sarah:
### Things for you to check/consider/care-about etc...
* Regenerated concreteness plots for TVQA/AVSD/PVSE: See `plots_n_stats/all/2_improved_concreteness_distribution`
Observation:
* Feature concreteness for each dataset appears intuitive
* Inparitulcar vcpts from TVQA
* Much of AVSD is concrete. Very interesting

## Plan:
(Recent History)
* ~~TVQA's statistics~~ (Early October)
* ~~Kastner's Thesis~~ (Mid October)
* ~~Create the word norm dictionary~~ (Late October)
* ~~Clean, centralise and 'clone ready' clean GitHub~~ (Late October)
* ~~Check examples of datasets with massively improved norm compilation:~~ (Late October)
    - ~~TVQA~~
    - ~~AVSD~~
    - ~~PVSE~~
* ~~Polish and extend norm-dict functionality~~ (End of October)
* ~~Redo Concreteness distribution:~~ (Beginning of November)
    - ~~TVQA~~
    - ~~AVSD~~
    - ~~PVSE~~
* ~~Isolate norm information in existing models. Using Hopfield network's BERT analysis~~ (End of November)
* Create associative/categorical network draft (NEXT)
* Norm pretraining experimental results (NEXT)
    - TVQA
    - AVSD
    - PVSE
* Network refinement
...
* Vcpts and regional features from both nautral co-reference. Segmentation map one part and coreference another!!!

## Dont Forget:
* [Object Oriented Ontology](`https://en.wikipedia.org/wiki/Object-oriented_ontology`)
* GET THE QA FOR ABSTRACT CONCEPTS FOR A GOOD AMOUNT OF ABSTRACT CONCEPTS
* TVQA statistics with BERT
* Redo TVQA's `1_data_concreteness_distribution` with extended full norm dictionary this time
* Create larger transformer model
* [Hopfield networks](`https://ml-jku.github.io/hopfield-layers/`) for associative learning
* Dean as an honorable mention for pointing out Hopfield networks to me
* (Hudson/Dean suggestion) (Pending)
* Emulate the Hopfield network's idea to get softmaxes of concepts that i have and see how close they are
    - Follow this up by associatively overtraining these elements to create proper convergence around these fuzzy terms
    - Do something similar for concrete terms
    - Find out which fine-tuned things are specialising in what
* Storing patterns is potentially better suited to associating linguistic labels with concrete instantiations of objects

## Paper:
![Too big to display](misc/imgs/Abstract_vs_Concrete.pdf)
 

## Introduction:
* This project was initially inspired by the [dual coding theory](https://www.taylorfrancis.com/books/9781315798868) paradigm. 
* The Neurologists and Psycholinguists have worked hard to isolate different properties of words, concepts and information we store in our brains, so-called **Psycholinguistic Word Norms**, e.g. concreteness, imagability, dominance...
* Of particular note is **concreteness**:"specific, definite and vivid", which is considered on a spectrum against **abstractness**: "vague, immaterial and incoporeal".
* Put far too simply, the most advanced intelligence we know of (the human brain) apparently sees it fit to store and handle concrete and abstract words and concepts in [structurally different representations](https://www.semanticscholar.org/paper/Abstract-and-concrete-concepts-have-structurally-Crutch-Warrington/fa8257eb0a6ca226ab65e3873577659d7be1d1a7).
* If the brain decides to engineer itself with these priors in mind, we presumptive explorers of intelligence would perhaps do well to consider how this may guide our comparatively clumsy efforts in modern machine learning.

## Contents:
* (`avsd`): AVSD implementation
* (`misc`): Holds images, the norm dictionary, and other various single-use files
* (`myutils.py`): Helper functions
* (`results`): Where runs are stored
* (`tvqa`): TVQA implementation
* (`word_norms.py`): Handling code for word norm processing and dictionary creation
* (`plots`): Directory for statistics and plots from each dataset. Noura and co should pay attention
* (`scripts`): A central location for example scripts for functionality of all code in this repository

## Contributions:
Dont bullet points just make everything nicer.
* **Big(gest?) Norm Dictionary:** We centralise many of the existing norm databases into one flexible and extensive resource. It includes concreteness values we focus on, but many, many more that others may find useful. To the best of our knowledge, this is the largest single compilation of word-norm databases available in code.
* **To be confirmed**

## Usage/Installation:
[AVSD](https://github.com/batra-mlp-lab/avsd) and [PVSE](https://github.com/yalesong/pvse) implementations are directly adapted from the official repositories. The [TVQA](https://github.com/Jumperkables/tvqa_modality_bias) implementation is one we used in another of our projects (which is in turn adapted from the [original repository](https://github.com/jayleicn/TVQA)). We thank and appreciate the authors of these repositories for their well documented implementations. If in using our implementation here you use any of the features from these 3 implementations please credit and cite the original authors and implementations as they ask.

### Central Repo and Word Norm Dictionary:
Example scripts running various experiments in this repository for all sections are centralised in (`scripts`).

0. Clone repo -> (`git clone git@github.com:Jumperkables/a_vs_c.git`)
1. Central (`a_vs_c`) virtual env: (`pip install -r requirements.txt`). You will have to edit running scripts to source your virtual environment. You may find it useful to create a symlink: (`ln -s /your/virtual/envs venvs`).
2. The Word Norm Dictionary. Use the supplied pickle file yourself, or generate your own and browse the other leftover norms those datasets have to offer:
    * (`misc/all_norms.pickle`) ('Word2Norm' class defined in (`word_norms.py`)
    * Gather a_vs_c data into a single directory. Follow the links below, and cross check with the path arguments in (`word_norms.py`) for appropriate subdirectory names. (`ln -s /single/directory/you/just/made data`) (This will take some time). When you're done, run (`scripts/extraction/create_norm_dict.sh`)

You will prepare several virtual environments for different implementations. You may skip any of these if you don't plan on using those parts of my implementation.
* Best to make some results directories (`mkdir results results/avsd results/pvse results/tvqa`)

### AVSD
0. 'avsd' virtual env: (`pip install -r avsd/requirements.txt`)
1. Follow the data download instructions in (`avsd/README.md`)
2. Take your pick: (`scripts/avsd/runs`)

### PVSE
0. 'pvse' virtual env: (`pip install -r pvse/requirements.txt`)
1.  Follow the data download instructions in (`pvse/README.md`)
2. ADDITIONALLY, (`ln -s /where/you/saved/pvse/data pvse/data`)
3. Bon appetite: (`scripts/pvse/runs`)

### TVQA
A bit more involved, because you'll have to set up my other TVQA repository and symlink it in here.

0. In a different location or directory (anywhere really), follow my full instructions for setting up my [tvqa_modality_bias](https://github.com/Jumperkables/tvqa_modality_bias) repo. **MAKE SURE TO CLONE THE a-vs-c BRANCH, NOT MASTER**
1. Now back in the root (`a_vs_c`) repo, create a symlink to TVQA repo you just installed: (`ln -s /path/to/tvqa_modality_bias tvqa/tvqa_modality_bias`) 
2. Create a symlink in the `tvqa_modality_bias/models` directory to the overall a_vs_c top-level `models` directory to allow imports from my custom models
(`ln path/to/a-vs-c/models tvqa_modality_bias/models/a-vs-c_models`)
3. (`scripts/tvqa/runs`)

## Norms Datasets:
The norm dictionary we created (`misc/all_norms.pickle`) is made using the following sources. The link are not ALL the official ones:
### Included
* [MT40k](http://crr.ugent.be/papers/Concreteness_ratings_Brysbaert_et_al_BRM.txt)
* [USF](https://github.com/teonbrooks/free_association)
* [MRC](https://github.com/samzhang111/mrc-psycholinguistics)
* [SimLex999](https://fh295.github.io/simlex.html)
* [Vinson](https://static-content.springer.com/esm/art%3A10.3758%2FBRM.40.1.183/MediaObjects/Vinson-BRM-2008a.zip)
* [McRae](https://static-content.springer.com/esm/art%3A10.3758%2FBF03192726/MediaObjects/McRae-BRM-2005.zip)
* [SimVerb](https://github.com/benathi/word2gm/tree/master/evaluation_data/simverb/data)
* [CP](https://link.springer.com/article/10.3758/BF03195584#SecESM1) (includes and extends from the PYM dataset)
* [TWP](https://github.com/friendly/WordPools/tree/master/R)
* [Battig](https://github.com/friendly/WordPools/tree/master/R)
* [Cortese](https://link.springer.com/article/10.3758/BF03195585#SecESM1)
* [Imageability Corpus](https://github.com/mkasu/imageabilitycorpus)
* [Reilly's Compilation](https://www.reilly-coglab.com/data) from "Formal Distinctiveness of High- and Low-Imageability Nouns:Analyses and Theoretical Implications" (contact the author to request access)
* [Sianpar's Indonesian Norms](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5138238/) (property norms for indonesian, but English translations are included)
* [Chinese Word Norm Corpus](https://www.ebi.ac.uk/biostudies/studies/S-EPMC5367816?xr=true) (norms for Chinese words, with English translations)
* [MEGAHR Facebook Cross-Lingual](https://github.com/clarinsi/megahr-crossling/)
* [Glasgow Word Norms](https://osf.io/ud367/)

### Honourable Mentions:
* [CSLB](http://www.csl.psychol.cam.ac.uk/propertynorms/) (the property norms are too specific for our use)
* [imSitu](https://public.ukp.informatik.tu-darmstadt.de/coling18-multimodalSurvey/) (wordy and specific descriptions of images)
* [EViLBERT](https://sapienzanlp.github.io/babelpic/) (embeddings and images of non-concrete concepts)

# Repo ToDo:
* Instructions for TVQA vocab changing
* Doublecheck that AVSD doesnt need extra external work done
* Mention and thank Remi's multimodal package.

# Citation:
Published at somewhere<br><br>
@inproceedings{avscmm,<br>
  title={TBC},<br>
  author={Winterbottom, T. and Xiao, S. and McLean, A. and Al Moubayed, N.},<br>
  booktitle={},<br>
  year={202X}<br>
}

# Help:
Feel free to contact me @ (`thomas.i.winterbottom@durham.ac.uk`) if you have any criticisms you'd like me to hear out or would like any help
