#!/bin/bash
#SBATCH -p part0
#SBATCH --job-name gqa_lxmert_unfreeze-none 
#SBATCH --ntasks 6
#SBATCH --gres gpu:1 
#SBATCH -o gqa_lxmert_unfreeze-none.out

cd ../../..
source venvs/a_vs_c/bin/activate
python VQA_dsets.py \
    --jobname gqa_lxmert_unfreeze-none \
    --dataset GQA \
    --epochs 20 \
    --bsz 16 \
    --val_bsz 100 \
    --device 0 \
    --unfreeze none \
    --wandb \

