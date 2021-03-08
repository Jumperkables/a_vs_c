#!/bin/bash
#SBATCH -p part0
#SBATCH --job-name gqa_bert-lstm_unfreeze-heads 
#SBATCH --ntasks 6
#SBATCH --gres gpu:1 
#SBATCH -o gqa_bert-lstm_unfreeze-heads.out

cd ../../..
source venvs/a_vs_c/bin/activate
python VQA_dsets.py \
    --jobname gqa_bert-lstm_unfreeze-heads \
    --dataset GQA \
    --model bert-lstm \
    --epochs 1000 \
    --bsz 16 \
    --val_bsz 100 \
    --device 0 \
    --unfreeze heads \
    --num_workers 4 \
    --lr 0.00008 \
    --wandb \
