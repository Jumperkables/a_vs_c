#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source ../../../venvs/a_vs_c/bin/activate
python ../../../misc/BERT_analysis.py \
    --purpose bertqa_logits \
    --model lxmert-qa \
    --device 0 \
    --dataset TVQA \
    --plot_title "LXMERT-QA @@ Final Layer Answer Logits" \
    --plot_save_path "/home/jumperkables/kable_management/projects/a_vs_c/plots_n_stats/BERT/7_qa_logits/lxmert@@.png" \
    --threshold 0.95
