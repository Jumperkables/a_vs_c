#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source ../../../venvs/a_vs_c/bin/activate
python ../../../misc/BERT_analysis.py \
    --purpose bertqa_logits \
    --model bert-qa \
    --device 0 \
    --dataset TVQA \
    --plot_title "BERT-QA @@ Final Layer Answer Logits" \
    --plot_save_path "/home/jumperkables/kable_management/projects/a_vs_c/plots_n_stats/BERT/7_qa_logits/bert@@.png" \
    --threshold 0.5

python ../../../misc/BERT_analysis.py \
    --purpose bertqa_logits \
    --model bert-qa \
    --device 0 \
    --dataset TVQA \
    --plot_title "BERT-QA @@ Final Layer Answer Logits" \
    --plot_save_path "/home/jumperkables/kable_management/projects/a_vs_c/plots_n_stats/BERT/7_qa_logits/bert@@.png" \
    --threshold 0.8

python ../../../misc/BERT_analysis.py \
    --purpose bertqa_logits \
    --model bert-qa \
    --device 0 \
    --dataset TVQA \
    --plot_title "BERT-QA @@ Final Layer Answer Logits" \
    --plot_save_path "/home/jumperkables/kable_management/projects/a_vs_c/plots_n_stats/BERT/7_qa_logits/bert@@.png" \
    --threshold 0.9

python ../../../misc/BERT_analysis.py \
    --purpose bertqa_logits \
    --model bert-qa \
    --device 0 \
    --dataset TVQA \
    --plot_title "BERT-QA @@ Final Layer Answer Logits" \
    --plot_save_path "/home/jumperkables/kable_management/projects/a_vs_c/plots_n_stats/BERT/7_qa_logits/bert@@.png" \
    --threshold 0.95

python ../../../misc/BERT_analysis.py \
    --purpose bertqa_logits \
    --model bert-qa \
    --device 0 \
    --dataset TVQA \
    --plot_title "BERT-QA @@ Final Layer Answer Logits" \
    --plot_save_path "/home/jumperkables/kable_management/projects/a_vs_c/plots_n_stats/BERT/7_qa_logits/bert@@.png" \
    --threshold 0.99
