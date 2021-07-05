#!/bin/bash
#SBATCH --qos long-high-prio
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 7-00:00
#SBATCH --mem 28G
#SBATCH -p res-gpu-small
#SBATCH --job-name vqacp_topk-1500_BUTD_loss-avsc-scaled_lr-5e-6_rubi-none 
#SBATCH --gres gpu:1 
#SBATCH -o vqacp_topk-1500_BUTD_loss-avsc-scaled_lr-5e-6_rubi-none.out

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
