#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 16G
#SBATCH -p res-gpu-small
#SBATCH --job-name gqa_dual-lx-forqa_unfreeze-qa_head_loss-default_norm-nsubj_lr-8e-5 
#SBATCH --gres gpu:1 
#SBATCH -o gqa_dual-lx-forqa_unfreeze-qa_head_loss-default_norm-nsubj_lr-8e-5.out

cd ../../../..
source venv/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python -W ignore main.py \
    --jobname gqa_dual-lx-forqa_unfreeze-qa_head_loss-default_norm-nsubj_lr-8e-5 \
    --dataset GQA \
    --model dual-lxforqa \
    --loss default \
    --norm_gt nsubj \
    --epochs 1000 \
    --bsz 32 \
    --val_bsz 100 \
    --device 0 \
    --unfreeze qa_head \
    --num_workers 2 \
    --lr 0.00008 \
    --wandb \
