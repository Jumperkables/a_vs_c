#!/bin/bash
#SBATCH --qos long-high-prio
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 7-00:00
#SBATCH -x gpu[0-3]
#SBATCH --mem 12G
#SBATCH -p res-gpu-small
#SBATCH --job-name gqa_dual-lx-lstm-h7l7_unfreeze-heads_loss-default_norm-nsubj 
#SBATCH --gres gpu:1 
#SBATCH -o gqa_dual-lx-lstm-h7l7_unfreeze-heads_loss-default_norm-nsubj.out

cd ../../../..
source venvs/a_vs_c/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python VQA_dsets.py \
    --jobname gqa_dual-lx-lstm-h7l7_unfreeze-heads_loss-default_norm-nsubj \
    --dataset GQA \
    --model dual-lx-lstm \
    --loss default \
    --norm_gt nsubj \
    --epochs 1000 \
    --bsz 32 \
    --val_bsz 100 \
    --device 1 \
    --unfreeze heads \
    --num_workers 4 \
    --lr 0.00008 \
    --wandb \
