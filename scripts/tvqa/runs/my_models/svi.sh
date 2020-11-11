#!/bin/bash
#SBATCH -p part0
#SBATCH --job-name svi_glove_concgt300 
#SBATCH --ntasks 6
#SBATCH --gres gpu:1 
#SBATCH -o svi_glove_concgt300.out

source /home/jumperkables/kable_management/python_venvs/mk8-tvqa/bin/activate

python -W ignore tvqa_modality_bias/main.py \
    --input_streams sub vcpt imagenet \
    --jobname=nstpwrds_svi_glove_concgt300 \
    --results_dir_base=concgt300 \
    --modelname=tvqa_abc_bert_nofc \
    --lrtype radam \
    --bsz 32 \
    --log_freq 800 \
    --test_bsz 100 \
    --lanecheck True \
    --word2idx tvqa_modality_bias/data/cache/word2idx.pickle \
    --lanecheck_path concgt300/lanecheck_dict.pickle 
