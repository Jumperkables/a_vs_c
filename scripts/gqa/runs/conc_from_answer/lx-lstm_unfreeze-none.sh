#!/bin/bash
#SBATCH --qos long-high-prio
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 7-00:00
#SBATCH --mem 12G
#SBATCH -p res-gpu-small
#SBATCH --job-name gqa_lx-lstm_unfreeze-none 
#SBATCH --gres gpu:1 
#SBATCH -o gqa_lx-lstm_unfreeze-none.out

cd ../../..
source venvs/a_vs_c/bin/activate
python VQA_dsets.py \
    --jobname gqa_lx-lstm_unfreeze-none \
    --dataset GQA \
    --model lx-lstm \
    --epochs 1000 \
    --bsz 32 \
    --val_bsz 100 \
    --device 1 \
    --unfreeze none \
    --num_workers 0 \
    --lr 0.00008 \
    --wandb \
