#!/bin/bash
#SBATCH --qos long-high-prio
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 7-00:00
#SBATCH -x gpu[0-3]
#SBATCH --mem 12G
#SBATCH -p res-gpu-small
#SBATCH --job-name vqacp_lx-lstm_unfreeze-heads_topk-500_lr-8e5 
#SBATCH --gres gpu:1 
#SBATCH -o vqacp_lx-lstm_unfreeze-heads_topk-500_lr-8e5.out

cd ../../..
source venv/bin/activate
python main.py \
    --jobname vqacp_lx-lstm_unfreeze-heads_topk-500_lr-8e5 \
    --dataset VQACP \
    --model lx-lstm \
    --epochs 500 \
    --bsz 16 \
    --val_bsz 100 \
    --device 0 \
    --unfreeze heads \
    --num_workers 4 \
    --lr 0.00008 \
    --topk 500 \
    --wandb \
