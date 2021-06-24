#!/bin/bash
#SBATCH --qos long-high-prio
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 7-00:00
#SBATCH --mem 16G
#SBATCH -p res-gpu-small
#SBATCH --job-name vqacp-topk-1500_dual-lx-forqa_unfreeze-qa_head_loss-default_norm-nsubj 
#SBATCH --gres gpu:1 
#SBATCH -o vqacp-topk-1500_dual-lx-forqa_unfreeze-qa_head_loss-default_norm-nsubj.out

cd ../../../..
source venvs/a_vs_c/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python -W ignore main.py \
    --jobname vqacp-topk-1500_dual-lx-forqa_unfreeze-qa_head_loss-default_norm-nsubj \
    --dataset VQACP \
    --topk 1500 \
    --model dual-lxforqa \
    --loss default \
    --norm_gt nsubj \
    --epochs 1000 \
    --bsz 32 \
    --val_bsz 100 \
    --device 0 \
    --unfreeze qa_head \
    --num_workers 0 \
    --lr 0.00001 \
    --wandb \
