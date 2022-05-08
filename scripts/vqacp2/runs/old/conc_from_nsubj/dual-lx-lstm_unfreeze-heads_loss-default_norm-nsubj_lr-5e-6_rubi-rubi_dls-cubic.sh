#!/bin/bash
#SBATCH --qos long-high-prio
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 7-00:00
#SBATCH -p res-gpu-small
#SBATCH --job-name vqacp2_topk-1500_dual-lx-lstm_unfreeze-heads_loss-default_norm-nsubj_lr-5e-6_rubi-rubi_dls-4th 
#SBATCH --gres gpu:1 
#SBATCH -o vqacp2_topk-1500_dual-lx-lstm_unfreeze-heads_loss-default_norm-nsubj_lr-5e-6_rubi-rubi_dls-4th.out

cd ../../../..
source venv/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python -W ignore main.py \
    --jobname vqacp2_topk-1500_dual-lx-lstm_unfreeze-heads_loss-default_norm-nsubj_lr-5e-6_rubi-rubi_dls-4th \
    --dataset VQACP2 \
    --topk 1500 \
    --model dual-lx-lstm \
    --loss default \
    --norm_gt nsubj \
    --epochs 1000 \
    --bsz 32 \
    --val_bsz 100 \
    --device 0 \
    --unfreeze heads \
    --num_workers 2 \
    --lr 0.000005 \
    --rubi rubi \
    --dual_loss_style 4th \
    --wandb \
