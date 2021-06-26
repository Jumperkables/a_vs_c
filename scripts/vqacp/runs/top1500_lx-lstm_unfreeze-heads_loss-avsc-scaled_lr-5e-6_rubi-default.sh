#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 21G
#SBATCH -p res-gpu-small
#SBATCH --job-name vqacp_topk-1500_lx-lstm_unfreeze-heads_loss-avsc-scaled_lr-5e-6_rubi-none 
#SBATCH --gres gpu:1 
#SBATCH -o vqacp_topk-1500_lx-lstm_unfreeze-heads_loss-avsc-scaled_lr-5e-6_rubi-none.out

cd ../../..
source venvs/a_vs_c/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python -W ignore main.py \
    --jobname vqacp_topk-1500_lx-lstm_unfreeze-heads_loss-avsc-scaled_lr-5e-6_rubi-none \
    --dataset vqacp \
    --topk 1500 \
    --model lx-lstm \
    --loss avsc-scaled \
    --epochs 50 \
    --bsz 32 \
    --val_bsz 100 \
    --device 0 \
    --unfreeze heads \
    --num_workers 2 \
    --lr 0.000005 \
    --rubi none \
    --wandb 
