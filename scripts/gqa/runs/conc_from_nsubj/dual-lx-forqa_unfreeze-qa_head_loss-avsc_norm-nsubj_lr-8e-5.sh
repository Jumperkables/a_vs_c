#!/bin/bash
#SBATCH --qos long-high-prio
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 7-00:00
#SBATCH -x gpu[0-3]
#SBATCH --mem 12G
#SBATCH -p res-gpu-small
#SBATCH --job-name gqa_dual-lx-forqa_unfreeze-qa_head_loss-avsc_norm-nusbj_lr-8e-5 
#SBATCH --gres gpu:1 
#SBATCH -o gqa_dual-lx-forqa_unfreeze-qa_head_loss-avsc_norm-nusbj_lr-8e-5.out

cd ../../../..
source venvs/a_vs_c/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python -W ignore VQA_dsets.py \
    --jobname gqa_dual-lx-forqa_unfreeze-qa_head_loss-avsc_norm-nusbj_lr-8e-5 \
    --dataset GQA \
    --model dual-lxforqa \
    --loss avsc \
    --norm_gt nsubj \
    --epochs 1000 \
    --bsz 32 \
    --val_bsz 100 \
    --device 1 \
    --unfreeze qa_head \
    --num_workers 8 \
    --lr 0.00008 \
    --wandb \
