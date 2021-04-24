#!/bin/bash
#SBATCH --qos long-high-prio
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 7-00:00
#SBATCH -x gpu[0-6]
#SBATCH --mem 22G
#SBATCH -p res-gpu-small
#SBATCH --job-name gqa_dual-lx-lstm_unfreeze-heads_loss-default_norm-nsubj_lr-5e-6_rubi-rubi_dls-quadr 
#SBATCH --gres gpu:1 
#SBATCH -o gqa_dual-lx-lstm_unfreeze-heads_loss-default_norm-nsubj_lr-5e-6_rubi-rubi_dls-quadr.out

cd ../../../..
source venvs/a_vs_c/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python -W ignore VQA_dsets.py \
    --jobname gqa_dual-lx-lstm_unfreeze-heads_loss-default_norm-nsubj_lr-5e-6_rubi-rubi_dls-quadr \
    --dataset GQA \
    --model dual-lx-lstm \
    --loss default \
    --norm_gt nsubj \
    --epochs 1000 \
    --bsz 64 \
    --val_bsz 100 \
    --device 0 \
    --unfreeze heads \
    --num_workers 0 \
    --lr 0.000005 \
    --rubi rubi \
    --dual_loss_style quadr \
    --wandb \
