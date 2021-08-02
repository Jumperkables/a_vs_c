#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 18G
#SBATCH -p res-gpu-small
#SBATCH --job-name va 
#SBATCH --gres gpu:1 
#SBATCH -o va.out

source ../../../venvs/avsd/bin/activate
python ../../../avsd/train.py \
    --input_type question_video_audio \
    --gpuid -1 \
    --jobname va_train \
    --save_path .results/va \
    --num_epochs 20 \
    --save_step 2 \
    --log

#python ../../../avsd/evaluate.py \
#    --input_type question_video_audio \
#    --split val \
#    --jobname va_val \
#    --save_ranks \
#    --load_path .results/va/20.pth \
#    --save_path .results/va/20_rank.json \
#    --log
