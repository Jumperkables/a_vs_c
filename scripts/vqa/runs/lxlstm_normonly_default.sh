#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name vqa_normonly_lx-lstm_loss-default_lr-3e-6_rubi-none 
#SBATCH --gres gpu:1 
#SBATCH -o ../../../checkpoints/vqa_normonly_lx-lstm_loss-default_lr-3e-6_rubi-none.out

cd ../../..
source venv/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python main.py \
    --jobname vqa_normonly_lx-lstm_loss-default_lr-3e-6_rubi-none \
    --dataset vqa \
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
    --min_ans_occ 9 \
    --norm_ans_only \
    --wandb \

