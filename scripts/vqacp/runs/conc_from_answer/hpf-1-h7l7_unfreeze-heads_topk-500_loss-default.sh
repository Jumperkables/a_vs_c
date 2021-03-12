#!/bin/bash
#SBATCH --qos long-high-prio
#SBATCH -N 1
#SBATCH -c 2
#SBATCH -t 7-00:00
#SBATCH --mem 12G
#SBATCH -p res-gpu-small
#SBATCH --job-name vqacp_hpf-1-h7l7_unfreeze-heads_topk-500_loss-default 
#SBATCH --gres gpu:1 
#SBATCH -o vqacp_hpf-1-h7l7_unfreeze-heads_topk-500_loss-default.out

cd ../../..
source venvs/a_vs_c/bin/activate
python VQA_dsets.py \
    --jobname vqacp_hpf-1-h7l7_unfreeze-heads_topk-500_loss-default \
    --dataset VQACP \
    --model hpf-1 \
    --loss default \
    --hopfield_beta_high 0.7 \
    --hopfield_beta_low 0.7 \
    --epochs 1000 \
    --bsz 16 \
    --val_bsz 100 \
    --device 0 \
    --unfreeze heads \
    --num_workers 2 \
    --lr 0.00008 \
    --topk 500 \
    --wandb \