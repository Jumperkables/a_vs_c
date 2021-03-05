#!/bin/bash
#SBATCH --qos long-high-prio
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 7-00:00
#SBATCH --mem 12G
#SBATCH -p res-gpu-small
#SBATCH --job-name gqa_induction_unfreeze-heads 
#SBATCH --gres gpu:1 
#SBATCH -o gqa_induction_unfreeze-heads.out

cd ../../../..
source venvs/a_vs_c/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python VQA_dsets.py \
    --jobname gqa_induction_unfreeze-heads \
    --dataset GQA \
    --model induction \
    --epochs 1000 \
    --bsz 32 \
    --val_bsz 100 \
    --device 0 \
    --unfreeze heads \
    --num_workers 0 \
    --lr 0.00008 \
    --wandb 
