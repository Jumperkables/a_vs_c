#!/bin/bash
#SBATCH -p part0
#SBATCH --job-name gqa_lxmert_unfreeze-none_lr-8e5 
#SBATCH --ntasks 6
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
    --device -1 \
    --unfreeze none \
    --num_workers 0 \
    --wandb \
    --lr 0.00008

