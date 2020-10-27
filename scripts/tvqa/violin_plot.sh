#!/bin/bash
source ../venvs/mk8-tvqa/bin/activate
python ./tvqa_modality_bias/tools/violin_plot.py \
    --lanecheck_path=/home/jumperkables/kable_management/projects/a_vs_c/tvqa/vi_concgt500/lanecheck_dict.pickle_valid \
    --jobname MRC_concgt500
