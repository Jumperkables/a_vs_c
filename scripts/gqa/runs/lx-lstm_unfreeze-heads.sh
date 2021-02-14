#!/bin/bash
#SBATCH -p part0
#SBATCH --job-name gqa_lx-lstm_unfreeze-heads 
#SBATCH --ntasks 6
#SBATCH --gres gpu:1 
#SBATCH -o gqa_lx-lstm_unfreeze-heads.out

cd ../../..
source venvs/a_vs_c/bin/activate
python VQA_dsets.py \
    --jobname gqa_lx-lstm_unfreeze-heads \
    --dataset GQA \
    --model lx-lstm \
    --epochs 1000 \
    --bsz 16 \
    --val_bsz 100 \
    --device 0 \
    --unfreeze heads \
    --num_workers 0 \
    --lr 0.00008 \
    --wandb \
