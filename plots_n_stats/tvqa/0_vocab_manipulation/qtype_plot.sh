#!/bin/bash
source ../venvs/mk8-tvqa/bin/activate
python ./tvqa_modality_bias/tools/question_type.py \
    --action magnum_opus \
    --jobname Concreteness
