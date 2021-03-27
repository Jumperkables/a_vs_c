#!/bin/bash
#SBATCH --qos long-high-prio
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 7-00:00
#SBATCH -x gpu[0-3]
#SBATCH --mem 12G
#SBATCH -p res-gpu-small
#SBATCH --job-name vqacp2-topk-1500_dual-lx-lstm_FIXED_unfreeze-heads_loss-avsc_norm-nsubj 
#SBATCH --gres gpu:1 
#SBATCH -o vqacp2-topk-1500_dual-lx-lstm_FIXED_unfreeze-heads_loss-avsc_norm-nsubj.out

cd ../../../..
source venvs/a_vs_c/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python -W ignore VQA_dsets.py \
    --jobname vqacp2-topk-1500_dual-lx-lstm_FIXED_unfreeze-heads_loss-avsc_norm-nsubj \
    --dataset VQACP2 \
    --topk 1500 \
    --model dual-lx-lstm \
    --loss avsc \
    --norm_gt nsubj \
    --epochs 1000 \
    --bsz 32 \
    --val_bsz 100 \
    --device 0 \
    --unfreeze heads \
    --num_workers 2 \
    --lr 0.00008 \
    --wandb \
