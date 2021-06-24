#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name vqacp_topk-1500_dual-lx-lstm_unfreeze-heads_loss-avsc_norm-nsubj_lr-5e-6_rubi-rubi_dls-quadr 
#SBATCH --gres gpu:1 
#SBATCH -o vqacp_topk-1500_dual-lx-lstm_unfreeze-heads_loss-avsc_norm-nsubj_lr-5e-6_rubi-rubi_dls-quadr.out

cd ../../../..
source venvs/a_vs_c/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python -W ignore VQA_dsets.py \
    --jobname vqacp_topk-1500_dual-lx-lstm_unfreeze-heads_loss-avsc_norm-nsubj_lr-5e-6_rubi-rubi_dls-quadr \
    --dataset VQACP \
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
    --lr 0.000005 \
    --rubi rubi \
    --dual_loss_style quadr \
    --wandb \
