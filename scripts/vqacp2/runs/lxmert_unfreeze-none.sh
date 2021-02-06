#!/bin/bash
#SBATCH -p part0
#SBATCH --job-name vqacp2_lxmert_unfreeze-none 
#SBATCH --ntasks 6
#SBATCH --gres gpu:1 
#SBATCH -o vqacp2_lxmert_unfreeze-none.out

cd ../../..
source venvs/a_vs_c/bin/activate
python VQA_dsets.py \
    --jobname vqacp2_lxmert_unfreeze-none \
    --dataset VQACP2 \
    --epochs 20 \
    --bsz 16 \
    --val_bsz 100 \
    --device 0 \
    --unfreeze none \
    --num_workers 4 \
    #--wandb \

