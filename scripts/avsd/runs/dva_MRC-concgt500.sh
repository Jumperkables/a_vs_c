#!/bin/bash
#SBATCH --ntasks 6
#SBATCH --mem 18G
#SBATCH -p part0
#SBATCH --job-name dva_MRC-concgt500 
#SBATCH --gres gpu:1 
#SBATCH -o dva_MRC-concgt500.out

source ../../../venvs/avsd/bin/activate
python ../../../avsd/train.py \
    --input_type question_video_audio \
    --gpuid 0 \
    --jobname dva_MRC-concgt500_train \
    --save_path .results/dva_MRC-concgt500 \
    --num_epochs 20 \
    --save_step 2 \
    --mrc_norms_conditions conc-gt-500 \
    --log

#python ../../../avsd/evaluate.py \
#    --input_type question_video_audio \
#    --split val \
#    --jobname dva_MRC-concgt500_val \
#    --save_ranks \
#    --load_path .results/dva_MRC-concgt500/20.pth \
#    --mrc_norms_conditions conc-gt-500 \
#    --save_path .results/dva_MRC-concgt500/20_rank.json \
#    --log
