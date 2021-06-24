#!/bin/bash
#SBATCH -p part0
#SBATCH --job-name vqacp_hpf-0-h7l3_unfreeze-none_mao-3 
#SBATCH --ntasks 6
#SBATCH --gres gpu:1 
#SBATCH -o vqacp_hpf-0-h7l3_unfreeze-none_mao-3.out

cd ../../..
source venvs/a_vs_c/bin/activate
python main.py \
    --jobname vqacp_hpf-0-h7l3_unfreeze-none_mao-3 \
    --dataset VQACP \
    --model hpf-0 \
    --hopfield_beta_high 0.7 \
    --hopfield_beta_low 0.3 \
    --epochs 1000 \
    --bsz 16 \
    --val_bsz 100 \
    --device -1 \
    --unfreeze none \
    --num_workers 2 \
    --lr 0.00008 \
    --min_ans_occ 3 \
    --wandb \
