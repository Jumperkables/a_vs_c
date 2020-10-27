# Abstract-vs-Concrete Guided Multimodal Machine Learning...
![Alt text](misc/imgs/a-c_scale.png?raw=true "Abstract -> Concrete")
#### ...and many other psycholinguistic norms for good measure

## Plan:

## Paper:
![Current State of the Paper](misc/imgs/Abstract_vs_Concrete.pdf =600)

## Introduction:
* This project was initially inspired by the [dual coding theory](https://www.taylorfrancis.com/books/9781315798868) paradigm. 
* The Neurologists and Psycholinguists have worked hard to isolate different properties of words, concepts and information we store in our brains, so-called **Psycholinguistic Word Norms**, e.g. concreteness, imagability, dominance...
* Of particular note is **concreteness**:"specific, definite and vivid", which is considered on a spectrum against **abstractness**: "vague, immaterial and incoporeal".
* Put far too simply, the most advanced intelligence we know of (the human brain) apparently sees it fit to store and handle concrete and abstract words and concepts in [structurally different representations](https://www.semanticscholar.org/paper/Abstract-and-concrete-concepts-have-structurally-Crutch-Warrington/fa8257eb0a6ca226ab65e3873577659d7be1d1a7).
* If the brain decides to engineer itself with these priors in mind, we presumptive explorers of intelligence would perhaps do well to consider how this may guide our comparatively clumsy efforts in modern machine learning.

## Contributions:
Dont bullet points just make everything nicer.
* **Big(gest?) Norm Dictionary:** We centralise many of the existing norm databases into one flexible and extensive resource. It includes concreteness values we focus on, but many, many more that others may find useful. To the best of our knowledge, this is the largest single compilation of word-norm databases available in code.
* **To be confirmed**


## Norms Datasets:
The following links are not ALL the official ones, merely where my implmenetation has drawn from. Word datasets that are currently collected and ready to use are:
* [MT40k](http://crr.ugent.be/papers/Concreteness_ratings_Brysbaert_et_al_BRM.txt)
* [CSLB](http://www.csl.psychol.cam.ac.uk/propertynorms/)
* [USF](https://github.com/teonbrooks/free_association)
* [MRC](https://github.com/samzhang111/mrc-psycholinguistics)
* [SimLex999](https://fh295.github.io/simlex.html)
* [Vinson](https://static-content.springer.com/esm/art%3A10.3758%2FBRM.40.1.183/MediaObjects/Vinson-BRM-2008a.zip)
* [McRae](https://static-content.springer.com/esm/art%3A10.3758%2FBF03192726/MediaObjects/McRae-BRM-2005.zip)
* [SimVerb](https://github.com/benathi/word2gm/tree/master/evaluation_data/simverb/data)
* [imSitu](https://public.ukp.informatik.tu-darmstadt.de/coling18-multimodalSurvey/)
* [CP](https://link.springer.com/article/10.3758/BF03195584#SecESM1) (includes and extends from the PYM dataset)
* [TWP](https://github.com/friendly/WordPools/tree/master/R)
* [Battig](https://github.com/friendly/WordPools/tree/master/R)
* [EViLBERT](https://sapienzanlp.github.io/babelpic/)
* [Cortese](https://link.springer.com/article/10.3758/BF03195585#SecESM1)
* [Reilly's Compilation](https://www.reilly-coglab.com/data) from "Formal Distinctiveness of High- and Low-Imageability Nouns:Analyses and Theoretical Implications" (contact the author to request access)
* [Imageability Corpus](https://github.com/mkasu/imageabilitycorpus)
<!---
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
-->

## Plan
### 08/09+ Start on a familiar dataset (AVSD):
* Consider extra resources recently supplied
* Begin exploring 'reducing abstract bias'. (To get used to handling these concepts in deep learning)
    - Prepare AVSD
    - Prepare a list of abstract and concrete concepts by rating
    - Consider performance of AVSD with abstract concepts removed from dialog, then concrete concepts removed
    - Ablate this with different levels of concreteness, cutoffs from 0-5 scores

### 15/09+ The ranked retrieval metric:
The proof of concept currently discussed with Alistair is image-ranked retrieval metric
* This metric can be facilitated through ranked-retrieval image tasks
* There are 
