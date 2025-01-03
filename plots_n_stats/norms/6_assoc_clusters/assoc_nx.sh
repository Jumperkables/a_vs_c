#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source ../../../venvs/a_vs_c/bin/activate
python ../../../misc/norm_analysis.py \
    --norm "assoc" \
    --norm_threshold 0 \
    \
    \
    --purpose "G_2_nx" \
    --draw_style "default" \
    --title "USF Assoc Metric Network: Assoc > 0" \
    --save \
    --save_path "/home/jumperkables/kable_management/projects/a_vs_c/plots_n_stats/norms/6_assoc_clusters/usf_assoc_gt0.png"

