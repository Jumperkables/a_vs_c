#!/bin/bash
#SBATCH -p part0
#SBATCH --job-name vqacp_bert-lstm_unfreeze-heads_topk-500_lr-8e5 
#SBATCH --ntasks 6
#SBATCH --gres gpu:1 
#SBATCH -o vqacp_bert-lstm_unfreeze-heads_topk-500_lr-8e5.out

cd ../../..
source venvs/a_vs_c/bin/activate
python VQA_dsets.py \
    --jobname vqacp_bert-lstm_unfreeze-heads_topk-500_lr-8e5 \
    --dataset VQACP \
    --model bert-lstm \
    --epochs 500 \
    --bsz 16 \
    --val_bsz 100 \
    --device 1 \
    --unfreeze heads \
    --num_workers 0 \
    --lr 0.00008 \
    --topk 500 \
    --wandb \

