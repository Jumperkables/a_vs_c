#!/bin/bash
#SBATCH -p part0
#SBATCH --job-name vqacp_lxmert_unfreeze-none 
#SBATCH --ntasks 6
#SBATCH --gres gpu:1 
#SBATCH -o vqacp_lxmert_unfreeze-none.out

cd ../../..
source venv/bin/activate
python main.py \
    --jobname vqacp_lxmert_unfreeze-none \
    --dataset VQACP \
    --epochs 20 \
    --bsz 16 \
    --val_bsz 100 \
    --device 0 \
    --unfreeze none \
    --num_workers 4 \
    #--wandb \

