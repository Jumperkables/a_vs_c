#!/bin/bash
#SBATCH --qos long-high-prio
#SBATCH -N 1
#SBATCH -c 2
#SBATCH -t 7-00:00
#SBATCH --mem 12G
#SBATCH -p res-gpu-small
#SBATCH --job-name vqacp_hpf-0-h3l3_unfreeze-none_mao-3_loss-avsc 
#SBATCH --gres gpu:1 
#SBATCH -o vqacp_hpf-0-h3l3_unfreeze-none_mao-3_loss-avsc.out

cd ../../..
source venv/bin/activate
python main.py \
    --jobname vqacp_hpf-0-h3l3_unfreeze-none_mao-3_loss-avsc \
    --dataset VQACP \
    --model hpf-0 \
    --loss avsc \
    --hopfield_beta_high 0.3 \
    --hopfield_beta_low 0.3 \
    --epochs 1000 \
    --bsz 16 \
    --val_bsz 100 \
    --device 0 \
    --unfreeze none \
    --num_workers 2 \
    --lr 0.00008 \
    --min_ans_occ 3 \
    --wandb \
