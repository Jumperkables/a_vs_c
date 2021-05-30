#!/bin/bash
source ../venvs/a_vs_c/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python explore_gqa.py \
    --dataset GQA \
    --num_workers 0 \
    --loss avsc
