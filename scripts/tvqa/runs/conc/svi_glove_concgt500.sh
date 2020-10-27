#!/bin/bash
#SBATCH -p part0
#SBATCH --job-name svi_glove_concgt500 
#SBATCH --ntasks 6
#SBATCH --gres gpu:1 
#SBATCH -o svi_glove_concgt500.out

source /home/jumperkables/kable_management/python_venvs/mk8-tvqa/bin/activate

python -W ignore tvqa_modality_bias/main.py \
    --input_streams sub vcpt imagenet \
    --jobname=nstpwrds_svi_glove_concgt500 \
    --results_dir_base=concgt500 \
    --modelname=tvqa_abc_bert_nofc \
    --lrtype radam \
    --bsz 32 \
    --log_freq 800 \
    --test_bsz 100 \
    --lanecheck True \
    --word2idx tvqa_modality_bias/data/cache/word2idx_concgt500.pickle \
    --lanecheck_path concgt500/lanecheck_dict.pickle 
