#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name vqacp_topk-1500_BUTD_loss-avsc-scaled_lr-5e-6_rubi-none 
#SBATCH --gres gpu:1 
#SBATCH -o ../../../checkpoints/vqacp_topk-1500_BUTD_loss-avsc-scaled_lr-5e-6_rubi-none.out

cd ../../..
source venvs/a_vs_c/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python -W ignore main.py \
    --jobname vqacp_topk-1500_BUTD_loss-avsc-scaled_lr-5e-6_rubi-none \
    --dataset vqacp \
    --topk 1500 \
    --model BUTD \
    --loss avsc-scaled \
    --epochs 50 \
    --bsz 16 \
    --val_bsz 100 \
    --device 0 \
    --unfreeze heads \
    --num_workers 4 \
    --lr 0.000001 \
    --rubi none \
    --wandb 
