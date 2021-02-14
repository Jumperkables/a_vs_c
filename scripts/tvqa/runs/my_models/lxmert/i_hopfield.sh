#!/bin/bash
#SBATCH -p part0
#SBATCH --job-name mk0_i_hopfield 
#SBATCH --ntasks 6
#SBATCH --gres gpu:1 
#SBATCH -o mk0_i_hopfield.out

source /home/jumperkables/kable_management/python_venvs/mk8-tvqa/bin/activate
cd ../../../../../tvqa/tvqa_modality_bias
python -W ignore main.py \
    --input_streams imagenet \
    --jobname=mk0_i_hopfield \
    --results_dir_base=.results/mk0_i_hopfield \
    --modelname=tvqa_avsc.Hopfield \
    --lrtype radam \
    --bsz 32 \
    --log_freq 800 \
    --test_bsz 100 \
    --word2idx data/cache/word2idx.pickle \
    --lanecheck_path .results/mk0_i_hopfield/lanecheck_dict.pickle \
    --bert lxmert \
    --lrtype radam \
