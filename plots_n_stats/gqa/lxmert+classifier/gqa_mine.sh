#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source ../../../venvs/a_vs_c/bin/activate
python ../../../misc/BERT_analysis.py \
    --purpose normqs \
    --dataset GQA \
    --norm conc-m \
    --model lxmert+classifier \
    --model_path "/home/jumperkables/kable_management/projects/a_vs_c/checkpoints/gqa_lxmert_unfreeze-heads-epoch=23-valid_acc=0.20.ckpt" \
    --device 1 \
    --max_seq_len 500 \
    --plot_title LXMERT+Classifier-Unfreeze_Heads_GQA_Concreteness \
    --plot_save_path /home/jumperkables/kable_management/projects/a_vs_c/plots_n_stats/gqa/lxmert+classifier/gqa.png \
    --high_threshold 0.96 \
    --low_threshold 0.35 \
    --threshold_mode mean
