#!/bin/bash
#SBATCH -p part0
#SBATCH --job-name mk0_i_lxmert 
#SBATCH --ntasks 6
#SBATCH --gres gpu:1 
#SBATCH -o mk0_i_lxmert.out

source /home/jumperkables/kable_management/python_venvs/mk8-tvqa/bin/activate
cd ../../../../../tvqa/tvqa_modality_bias
python -W ignore main.py \
    --input_streams imagenet \
    --jobname=mk0_i_lxmert \
    --results_dir_base=.results/mk0_i_lxmert \
    --modelname=tvqa_avsc.Lxmert_adapt \
    --lrtype radam \
    --bsz 32 \
    --log_freq 800 \
    --test_bsz 100 \
    --word2idx data/cache/word2idx.pickle \
    --lanecheck_path .results/mk0_i_lxmert/lanecheck_dict.pickle \
    --bert lxmert \
    --lrtype radam \
    --wandb \
    --max_es_cnt 5 \
    --n_epoch 300
