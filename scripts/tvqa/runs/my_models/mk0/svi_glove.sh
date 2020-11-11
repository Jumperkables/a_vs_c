#!/bin/bash
#SBATCH -p part0
#SBATCH --job-name mk0_svi_glove 
#SBATCH --ntasks 6
#SBATCH --gres gpu:1 
#SBATCH -o mk0_svi_glove.out

source /home/jumperkables/kable_management/python_venvs/mk8-tvqa/bin/activate
cd ../../../../../tvqa/tvqa_modality_bias
python -W ignore main.py \
    --input_streams sub vcpt imagenet \
    --jobname=mk0_svi_glove \
    --results_dir_base=mk0_svi_glove \
    --modelname=a-vs-c_models.assoc_vs_ctgrcl \
    --lrtype radam \
    --bsz 32 \
    --log_freq 800 \
    --test_bsz 100 \
    --lanecheck True \
    --word2idx data/cache/word2idx.pickle \
    --lanecheck_path mk0_svi_glove/lanecheck_dict.pickle 
