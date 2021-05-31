#!/bin/bash
#SBATCH --ntasks 6
#SBATCH --mem 28G
#SBATCH -p part0
#SBATCH --job-name gqa-absMixed_dual-lx-lstm_unfreeze-heads_loss-default_norm-qtype-full_lr-1e-5_rubi-none_dls-linear 
#SBATCH --gres gpu:1 
#SBATCH -o gqa-absMixed_dual-lx-lstm_unfreeze-heads_loss-default_norm-qtype-full_lr-1e-5_rubi-none_dls-linear.out

cd ../../../..
source venvs/a_vs_c/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python VQA_dsets.py \
    --jobname gqa-absMixed_dual-lx-lstm_unfreeze-heads_loss-default_norm-qtype-full_lr-1e-5_rubi-none_dls-linear \
    --dataset GQA-absMixed \
    --model dual-lx-lstm \
    --loss default \
    --norm_gt qtype-full \
    --epochs 1000 \
    --bsz 16 \
    --val_bsz 100 \
    --device 0 \
    --unfreeze heads \
    --num_workers 4 \
    --lr 0.00001 \
    --rubi none \
    --dual_loss_style linear \
    --wandb \
