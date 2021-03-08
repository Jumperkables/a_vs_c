#!/bin/bash
#SBATCH --qos long-high-prio
#SBATCH -N 1
#SBATCH -c 2
#SBATCH -t 7-00:00
#SBATCH --mem 16G
#SBATCH -p res-gpu-small
#SBATCH --job-name vqacp2_hpf-3-h7l7_unfreeze-heads_topk-500_loss-avsc_norm-nsubj 
#SBATCH --gres gpu:1 
#SBATCH -o vqacp2_hpf-3-h7l7_unfreeze-heads_topk-500_loss-avsc_norm-nsubj.out

cd ../../..
source venvs/a_vs_c/bin/activate
python VQA_dsets.py \
    --jobname vqacp2_hpf-3-h7l7_unfreeze-heads_topk-500_loss-avsc_norm-nsubj \
    --dataset VQACP2 \
    --model hpf-3 \
    --loss avsc \
    --norm_gt answer \
    --hopfield_beta_high 0.7 \
    --hopfield_beta_low 0.7 \
    --epochs 1000 \
    --bsz 16 \
    --val_bsz 100 \
    --device 0 \
    --unfreeze heads \
    --num_workers 0 \
    --lr 0.00008 \
    --topk 500 \
    --wandb \
