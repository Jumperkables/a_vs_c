#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 18G
#SBATCH -p res-gpu-small
#SBATCH --job-name dva_MRC-concgt300 
#SBATCH --gres gpu:1 
#SBATCH -o dva_MRC-concgt300.out

source ../../../venvs/avsd/bin/activate
python ../../../avsd/train.py \
    --input_type question_video_audio \
    --gpuid 0 \
    --jobname dva_MRC-concgt300_train \
    --save_path .results/dva_MRC-concgt300 \
    --num_epochs 20 \
    --save_step 2 \
    --mrc_norms_conditions conc-gt-300 \
    --log

#python ../../../avsd/evaluate.py \
#    --input_type question_video_audio \
#    --split val \
#    --jobname dva_MRC-concgt300_val \
#    --save_ranks \
#    --load_path .results/dva_MRC-concgt300/20.pth \
#    --mrc_norms_conditions conc-gt-300 \
#    --save_path .results/dva_MRC-concgt300/20_rank.json \
#    --log
