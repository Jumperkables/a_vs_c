#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source ../../../venvs/a_vs_c/bin/activate
python ../../../misc/bert_analysis.py \
    --purpose bertqa_logits \
    --model lxmert-qa \
    --device 0 \
    --dataset tvqa \
    --plot_title "lxmert-qa @@ final layer answer logits" \
    --plot_save_path "/home/jumperkables/kable_management/projects/a_vs_c/plots_n_stats/bert/7_qa_logits/lxmert@@.png" \
    --threshold 0.5
