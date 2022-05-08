#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name vqa2_topk-1500_lx-lstm_unfreeze-heads_loss-default_lr-3e-6_rubi-none 
#SBATCH --gres gpu:1 
#SBATCH -o ../../../checkpoints/vqa2_topk-1500_lx-lstm_unfreeze-heads_loss-default_lr-3e-6_rubi-none.out

cd ../../..
source venv/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python main.py \
    --jobname vqa2_topk-1500_lx-lstm_unfreeze-heads_loss-default_lr-3e-6_rubi-none \
    --dataset vqa2 \
    --topk 1500 \
    --model lx-lstm \
    --loss default \
    --epochs 100 \
    --bsz 64 \
    --val_bsz 64 \
    --device 0 \
    --unfreeze heads \
    --num_workers 4 \
    --lr 3e-6 \
    --rubi none \
    --wandb 
