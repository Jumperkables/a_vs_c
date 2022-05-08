#!/bin/bash
#SBATCH --qos long-high-prio
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 7-00:00
#SBATCH -x gpu[0-3]
#SBATCH --mem 12G
#SBATCH -p res-gpu-small
#SBATCH --job-name gqa_hpf-1-h7l7_unfreeze-none_loss-default 
#SBATCH --gres gpu:1 
#SBATCH -o gqa_hpf-1-h7l7_unfreeze-none_loss-default.out

cd ../../..
source venv/bin/activate
python main.py \
    --jobname gqa_hpf-1-h7l7_unfreeze-none_loss-default \
    --dataset GQA \
    --model hpf-1 \
    --hopfield_beta_high 0.7 \
    --hopfield_beta_low 0.7 \
    --loss default \
    --epochs 1000 \
    --bsz 16 \
    --val_bsz 100 \
    --device 0 \
    --unfreeze none \
    --num_workers 4 \
    --lr 0.00008 \
    --wandb \
