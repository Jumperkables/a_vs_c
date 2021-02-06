#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source ../../../venvs/a_vs_c/bin/activate
python ../../../misc/BERT_analysis.py \
    --purpose normqs \
    --dataset VQACP2 \
    --norm conc-m \
    --model lxmert+classifier \
    --model_path "/home/jumperkables/kable_management/projects/a_vs_c/checkpoints/vqacp2_lxmert_unfreeze-heads_topk-500_lr-8e5-epoch=22-valid_acc=0.19.ckpt" \
    --device 1 \
    --max_seq_len 500 \
    --plot_title LXMERT+Classifier-Unfreeze_Heads_VQACP2-topk500_Concreteness \
    --plot_save_path /home/jumperkables/kable_management/projects/a_vs_c/plots_n_stats/gqa/lxmert+classifier/vqacp2-topk500.png \
    --high_threshold 0.96 \
    --low_threshold 0.35 \
    --threshold_mode mean
