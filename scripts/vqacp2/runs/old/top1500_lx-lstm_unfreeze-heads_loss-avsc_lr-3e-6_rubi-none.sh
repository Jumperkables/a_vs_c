#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name vqacp2_topk-1500_lx-lstm_unfreeze-heads_loss-avsc_lr-3e-6_rubi-none 
#SBATCH --gres gpu:1 
#SBATCH -o ../../../checkpoints/vqacp2_topk-1500_lx-lstm_unfreeze-heads_loss-avsc_lr-3e-6_rubi-none.out

cd ../../..
source venv/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python main.py \
    --jobname vqacp2_topk-1500_lx-lstm_unfreeze-heads_loss-avsc_lr-3e-6_rubi-none \
    --dataset vqacp2 \
    --topk 1500 \
    --model lx-lstm \
    --loss avsc \
    --epochs 100 \
    --bsz 64 \
    --val_bsz 64 \
    --device 0 \
    --unfreeze heads \
    --num_workers 4 \
    --lr 3e-6 \
    --rubi none \
    --wandb 
