#!/bin/bash
#SBATCH -p part0
#SBATCH --job-name default 
#SBATCH --ntasks 6
#SBATCH --gres gpu:1 
#SBATCH -o default.out

source ../../../venvs/avsd/bin/activate
python ../../../avsd/train.py \
    --gpuid -1 \
    --jobname default_train \
    --save_path .results/default \
    --num_epochs 20 \
    --save_step 2 \
#    --log

#python ../../../avsd/evaluate.py \
#    --split val \
#    --jobname default_val \
#    --save_ranks \
#    --load_path .results/default/20.pth \
#    --save_path .results/default/20_rank.json
