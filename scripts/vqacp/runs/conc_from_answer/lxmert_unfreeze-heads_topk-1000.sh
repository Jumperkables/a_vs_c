#!/bin/bash
#SBATCH -p part0
#SBATCH --job-name vqacp_lxmert_unfreeze-heads_topk-1000_lr-8e5 
#SBATCH --ntasks 6
#SBATCH --gres gpu:1 
#SBATCH -o vqacp_lxmert_unfreeze-heads_topk-1000_lr-8e5.out

cd ../../..
source venvs/a_vs_c/bin/activate
python main.py \
    --jobname vqacp_lxmert_unfreeze-heads_topk-1000_lr-8e5 \
    --dataset VQACP \
    --epochs 1000 \
    --bsz 16 \
    --val_bsz 100 \
    --device 1 \
    --unfreeze heads \
    --num_workers 0 \
    --lr 0.00008 \
    --topk 1000 \
    --wandb \

