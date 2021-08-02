#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 18G
#SBATCH -p res-gpu-small
#SBATCH --job-name vqa2_topk-1500_BUTD_loss-default_lr-6e-6_rubi-none 
#SBATCH --gres gpu:1 
#SBATCH -o ../../../checkpoints/vqa2_topk-1500_BUTD_loss-default_lr-6e-6_rubi-none.out

cd ../../..
source venvs/a_vs_c/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python main.py \
    --jobname vqa2_topk-1500_BUTD_loss-default_lr-6e-6_rubi-none \
    --dataset vqa2 \
    --topk 1500 \
    --model BUTD \
    --loss default \
    --epochs 100 \
    --bsz 64 \
    --val_bsz 64 \
    --device 0 \
    --unfreeze heads \
    --num_workers 4 \
    --lr 6e-6 \
    --rubi none \
    --wandb 
