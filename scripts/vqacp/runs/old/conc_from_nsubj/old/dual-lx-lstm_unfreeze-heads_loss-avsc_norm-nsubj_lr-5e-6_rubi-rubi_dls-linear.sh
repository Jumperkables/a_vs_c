#!/bin/bash
#SBATCH --qos long-high-prio
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 7-00:00
#SBATCH -x gpu[0-6]
#SBATCH --mem 28G
#SBATCH -p res-gpu-small
#SBATCH --job-name vqacp_topk-1500_dual-lx-lstm_unfreeze-heads_loss-avsc_norm-nsubj_lr-5e-6_rubi-rubi_dls-linear 
#SBATCH --gres gpu:1 
#SBATCH -o vqacp_topk-1500_dual-lx-lstm_unfreeze-heads_loss-avsc_norm-nsubj_lr-5e-6_rubi-rubi_dls-linear.out

cd ../../../..
source venv/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python -W ignore main.py \
    --jobname vqacp_topk-1500_dual-lx-lstm_unfreeze-heads_loss-avsc_norm-nsubj_lr-5e-6_rubi-rubi_dls-linear \
    --dataset VQACP \
    --topk 1500 \
    --model dual-lx-lstm \
    --loss avsc \
    --norm_gt nsubj \
    --epochs 1000 \
    --bsz 128 \
    --val_bsz 100 \
    --device 0 \
    --unfreeze heads \
    --num_workers 4 \
    --lr 0.000005 \
    --rubi rubi \
    --dual_loss_style linear \
    --wandb \
