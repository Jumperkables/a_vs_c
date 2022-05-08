#!/bin/bash
#SBATCH -p part0
#SBATCH --job-name vqacp2_lxmert_unfreeze-heads_topk-1000_lr-8e5 
#SBATCH --ntasks 6
#SBATCH --gres gpu:1 
#SBATCH -o vqacp2_lxmert_unfreeze-heads_topk-1000_lr-8e5.out

cd ../../..
source venv/bin/activate
python main.py \
    --jobname vqacp2_lxmert_unfreeze-heads_topk-1000_lr-8e5 \
    --dataset VQACP2 \
    --epochs 1000 \
    --bsz 16 \
    --val_bsz 100 \
    --device 0 \
    --unfreeze heads \
    --num_workers 0 \
    --lr 0.00008 \
    --topk 1000 \
    --wandb \

