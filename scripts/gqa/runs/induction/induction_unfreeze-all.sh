#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH -x gpu[0-3]
#SBATCH --mem 12G
#SBATCH -p res-gpu-small
#SBATCH --job-name gqa_induction_unfreeze-all 
#SBATCH --gres gpu:1 
#SBATCH -o gqa_induction_unfreeze-all.out

cd ../../../..
echo "Dont need to try this one out"
exit
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
