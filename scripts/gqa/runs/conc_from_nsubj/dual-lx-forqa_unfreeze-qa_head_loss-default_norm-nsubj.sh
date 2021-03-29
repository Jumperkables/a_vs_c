#!/bin/bash
#SBATCH --qos long-high-prio
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 7-00:00
#SBATCH -x gpu[0-3]
#SBATCH --mem 12G
#SBATCH -p res-gpu-smqa_head
#SBATCH --job-name gqa_dual-lx-forqa_unfreeze-qa_head_loss-default_norm-nsubj 
#SBATCH --gres gpu:1 
#SBATCH -o gqa_dual-lx-forqa_unfreeze-qa_head_loss-default_norm-nsubj.out

cd ../../../..
source venvs/a_vs_c/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python -W ignore VQA_dsets.py \
    --jobname gqa_dual-lx-forqa_unfreeze-qa_head_loss-default_norm-nsubj \
    --dataset GQA \
    --model dual-lxforqa \
    --loss default \
    --norm_gt nsubj \
    --epochs 1000 \
    --bsz 32 \
    --val_bsz 100 \
    --device 1 \
    --unfreeze qa_head \
    --num_workers 8 \
    --lr 0.00001 \
    --wandb \
