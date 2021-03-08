#!/bin/bash
#SBATCH -p part0
#SBATCH --job-name gqa_hpf-0-h7l3_unfreeze-none 
#SBATCH --ntasks 6
#SBATCH --gres gpu:1 
#SBATCH -o gqa_hpf-0-h7l3_unfreeze-none.out

cd ../../..
source venvs/a_vs_c/bin/activate
python VQA_dsets.py \
    --jobname gqa_hpf-0-h7l3_unfreeze-none \
    --dataset GQA \
    --model hpf-0 \
    --hopfield_beta_high 0.7 \
    --hopfield_beta_low 0.3 \
    --epochs 1000 \
    --bsz 16 \
    --val_bsz 100 \
    --device 1 \
    --unfreeze none \
    --num_workers 2 \
    --lr 0.00008 \
    --wandb \
