#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 12G
#SBATCH -p res-gpu-small
#SBATCH --job-name vqacp2_normonly_lxmert_loss-avsc_lr-2e-6_rubi-none 
#SBATCH --gres gpu:1 
#SBATCH -o ../../../checkpoints/vqacp2_normonly_lxmert_loss-avsc_lr-2e-6_rubi-none.out

cd ../../..
source venv/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python main.py \
    --jobname vqacp2_normonly_lxmert_loss-avsc_lr-2e-6_rubi-none \
    --dataset vqacp2 \
    --model lxmert \
    --loss avsc \
    --epochs 75 \
    --bsz 64 \
    --val_bsz 64 \
    --device 0 \
    --num_workers 4 \
    --lr 2e-6 \
    --rubi none \
    --min_ans_occ 9 \
    --wandb \
    --norm_ans_only \

