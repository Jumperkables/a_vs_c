#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source ../../../../venvs/a_vs_c/bin/activate
python ../../../../misc/BERT_analysis.py \
    --purpose tvqaconcqs \
    --model lxmert-qa \
    --comp_pool concrete \
    --device 1 \
    --max_seq_len 500 \
    --plot_title LXMERT-QA_Concreteness_POOL-conc \
    --plot_save_path /home/jumperkables/kable_management/projects/a_vs_c/plots_n_stats/BERT/5_qnas_conc/lxmert-qa/concpool.png \
    --threshold 0.9 \
    --threshold_mode mean
