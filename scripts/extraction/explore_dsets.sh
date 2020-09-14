#!/bin/bash
source ../../venvs/a_vs_c/bin/activate
python ../../word_norms.py \
    -purpose explore_dsets \
    --dsets CSLB USF MT40k CP TWP  
