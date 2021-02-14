#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source ../../../../venvs/a_vs_c/bin/activate
python ../../../../misc/BERT_analysis.py \
    --purpose tvqaconcqs \
    --model default \
    --comp_pool abstract \
    --device 0 \
    --max_seq_len 500 \
    --plot_title BERT_Concreteness_POOL-Abs \
    --plot_save_path /home/jumperkables/kable_management/projects/a_vs_c/plots_n_stats/BERT/5_qnas_conc/bert/abspool.png \
    --threshold 0.9 \
    --threshold_mode mean