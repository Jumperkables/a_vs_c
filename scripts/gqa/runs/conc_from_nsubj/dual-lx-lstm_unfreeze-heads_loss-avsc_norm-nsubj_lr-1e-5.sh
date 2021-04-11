#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -t 7-00:00
#SBATCH -p part0
#SBATCH --job-name gqa_REPEAT_dual-lx-lstm_unfreeze-heads_loss-avsc_norm-nsubj_lr-1e-5 
#SBATCH --gres gpu:1 
#SBATCH -o gqa_REPEAT_dual-lx-lstm_unfreeze-heads_loss-avsc_norm-nsubj_lr-1e-5.out

cd ../../../..
source venvs/a_vs_c/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python -W ignore VQA_dsets.py \
    --jobname gqa_REPEAT_dual-lx-lstm_unfreeze-heads_loss-avsc_norm-nsubj_lr-1e-5 \
    --dataset GQA \
    --model dual-lx-lstm \
    --loss avsc \
    --norm_gt nsubj \
    --epochs 1000 \
    --bsz 32 \
    --val_bsz 100 \
    --device 0 \
    --unfreeze heads \
    --num_workers 2 \
    --lr 0.00001 \
    --wandb \
