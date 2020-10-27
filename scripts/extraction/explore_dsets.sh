#!/bin/bash
source ../../venvs/a_vs_c/bin/activate
python ../../word_norms.py \
    --purpose explore_dsets \
    --dsets MT40k USF MRC SimLex999 Vinson McRae SimVerb CP TWP Battig Cortese MM_imgblty sianpar_indo yee_chinese megahr_crossling glasgow
# TODO Reilly (First half)
# INCOMPLETE: USF, McRae, CP, TWP, Reilly
# INGORE CSLB imSitu EViLBERT
## CONSIDER PLEASANTNESS AS A NORM
