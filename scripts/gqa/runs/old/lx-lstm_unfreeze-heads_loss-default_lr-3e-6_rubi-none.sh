#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -t 7-00:00
#SBATCH --mem 18G
#SBATCH -p part0
#SBATCH --job-name gqa_lx-lstm_unfreeze-heads_loss-default_lr-3e-6_rubi-none 
#SBATCH --gres gpu:1 
#SBATCH -o ../../../checkpoints/gqa_lx-lstm_unfreeze-heads_loss-default_lr-3e-6_rubi-none.out

cd ../../..
source venv/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python main.py \
    --jobname gqa_lx-lstm_unfreeze-heads_loss-default_lr-3e-6_rubi-none \
    --dataset gqa \
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
