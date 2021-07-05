#!/bin/bash
#SBATCH --qos long-high-prio
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 7-00:00
#SBATCH --mem 28G
#SBATCH -p res-gpu-small
#SBATCH --job-name vqacp_normonly_BUTD_loss-avsc_lr-5e-6_rubi-none 
#SBATCH --gres gpu:1 
#SBATCH -o vqacp_normonly_BUTD_loss-avsc_lr-5e-6_rubi-none.out

cd ../../..
source venvs/a_vs_c/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python -W ignore main.py \
    --jobname vqacp_normonly_BUTD_loss-avsc_lr-5e-6_rubi-none \
    --dataset vqacp \
    --min_ans_occ 2 \
    --norm_ans_only \
    --model BUTD \
    --loss avsc \
    --epochs 30 \
    --bsz 32 \
    --val_bsz 100 \
    --device 0 \
    --unfreeze heads \
    --num_workers 4 \
    --lr 0.000005 \
    --rubi none \
    --wandb 
