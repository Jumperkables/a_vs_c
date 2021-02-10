#!/bin/bash
#SBATCH -p part0
#SBATCH --job-name gqa_induction_unfreeze-all 
#SBATCH --ntasks 6
#SBATCH --gres gpu:1 
#SBATCH -o gqa_induction_unfreeze-all.out

cd ../../../..
source venvs/a_vs_c/bin/activate
python VQA_dsets.py \
    --jobname gqa_induction_unfreeze-all \
    --dataset GQA \
    --model induction \
    --epochs 1000 \
    --bsz 16 \
    --val_bsz 100 \
    --device 1 \
    --unfreeze all \
    --num_workers 0 \
    --lr 0.00008 \
    --wandb 
