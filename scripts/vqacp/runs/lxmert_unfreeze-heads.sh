#!/bin/bash
#SBATCH -p part0
#SBATCH --job-name vqacp_lxmert_unfreeze-heads 
#SBATCH --ntasks 6
#SBATCH --gres gpu:1 
#SBATCH -o vqacp_lxmert_unfreeze-heads.out

cd ../../..
source venvs/a_vs_c/bin/activate
python VQA_dsets.py \
    --jobname vqacp_lxmert_unfreeze-heads \
    --dataset VQACP \
    --epochs 20 \
    --bsz 16 \
    --val_bsz 100 \
    --device 0 \
    --unfreeze heads \
    #--wandb \

