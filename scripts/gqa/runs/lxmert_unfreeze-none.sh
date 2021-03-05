#!/bin/bash
#SBATCH --qos long-high-prio
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 7-00:00
#SBATCH --mem 12G
#SBATCH -p res-gpu-small
#SBATCH --job-name gqa_lxmert_unfreeze-none_lr-8e5 
#SBATCH --gres gpu:1 
#SBATCH -o gqa_lxmert_unfreeze-none_lr-8e5.out

cd ../../..
source venvs/a_vs_c/bin/activate
python VQA_dsets.py \
    --jobname gqa_lxmert_unfreeze-none_lr-8e5 \
    --dataset GQA \
    --epochs 1000 \
    --bsz 16 \
    --val_bsz 100 \
    --device 0 \
    --unfreeze none \
    --num_workers 0 \
    --wandb \
    --lr 0.00008

