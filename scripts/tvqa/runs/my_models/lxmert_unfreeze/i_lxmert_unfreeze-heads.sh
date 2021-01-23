#!/bin/bash
#SBATCH -p part0
#SBATCH --job-name lxmert_unfreezeheads_i_lxmert 
#SBATCH --ntasks 6
#SBATCH --gres gpu:1 
#SBATCH -o lxmert_unfreezeheads_i_lxmert.out

source /home/jumperkables/kable_management/python_venvs/mk8-tvqa/bin/activate
cd ../../../../../tvqa/tvqa_modality_bias
python -W ignore main.py \
    --input_streams imagenet \
    --jobname=lxmert_unfreezeheads_i_lxmert \
    --results_dir_base=.results/lxmert_unfreezeheads_i_lxmert \
    --modelname=tvqa_avsc.Lxmert_adapt \
    --lrtype radam \
    --bsz 32 \
    --log_freq 1900 \
    --test_bsz 100 \
    --word2idx data/cache/word2idx.pickle \
    --lanecheck_path .results/lxmert_unfreezeheads_i_lxmert/lanecheck_dict.pickle \
    --mload_path /home/jumperkables/kable_management/projects/a_vs_c/tvqa/tvqa_modality_bias/.results/lxmert_unfreezeheads_i_lxmert/best_valid.pth \
    --bert lxmert \
    --lrtype radam \
    --max_es_cnt 10 \
    --n_epoch 300 \
    --device 1 \
    --wandb \
    --unfreeze heads
