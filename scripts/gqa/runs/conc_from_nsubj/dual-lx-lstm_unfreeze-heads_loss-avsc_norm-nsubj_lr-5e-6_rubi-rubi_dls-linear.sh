#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 22G
#SBATCH -p res-gpu-small
#SBATCH --job-name gqa_dual-lx-lstm_unfreeze-heads_loss-avsc_norm-nsubj_lr-5e-6_rubi-rubi_dls-linear 
#SBATCH --gres gpu:1 
#SBATCH -o gqa_dual-lx-lstm_unfreeze-heads_loss-avsc_norm-nsubj_lr-5e-6_rubi-rubi_dls-linear.out

cd ../../../..
source venvs/a_vs_c/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python -W ignore VQA_dsets.py \
    --jobname gqa_dual-lx-lstm_unfreeze-heads_loss-avsc_norm-nsubj_lr-5e-6_rubi-rubi_dls-linear \
    --dataset GQA \
    --model dual-lx-lstm \
    --loss avsc \
    --norm_gt nsubj \
    --epochs 1000 \
    --bsz 16 \
    --val_bsz 100 \
    --device 0 \
    --unfreeze heads \
    --num_workers 2 \
    --lr 0.000005 \
    --rubi rubi \
    --dual_loss_style linear \
    --wandb \
