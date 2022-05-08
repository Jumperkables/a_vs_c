#!/bin/bash
#SBATCH --qos long-high-prio
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 7-00:00
#SBATCH -x gpu[0-6]
#SBATCH --mem 28G
#SBATCH -p res-gpu-small
#SBATCH --job-name gqa_dual-lx-lstm_unfreeze-heads_loss-default_norm-nsubj_lr-5e-6_rubi-rubi_dls-cubic 
#SBATCH --gres gpu:1 
#SBATCH -o gqa_dual-lx-lstm_unfreeze-heads_loss-default_norm-nsubj_lr-5e-6_rubi-rubi_dls-cubic.out

cd ../../../..
source venv/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python -W ignore main.py \
    --jobname gqa_dual-lx-lstm_unfreeze-heads_loss-default_norm-nsubj_lr-5e-6_rubi-rubi_dls-cubic \
    --dataset GQA \
    --model dual-lx-lstm \
    --loss default \
    --norm_gt nsubj \
    --epochs 1000 \
    --bsz 128 \
    --val_bsz 100 \
    --device 0 \
    --unfreeze heads \
    --num_workers 4 \
    --lr 0.000005 \
    --rubi rubi \
    --dual_loss_style cubic \
    --wandb \
