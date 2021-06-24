#!/bin/bash
#SBATCH --qos long-high-prio
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 7-00:00
#SBATCH -x gpu[0-6]
#SBATCH --mem 28G
#SBATCH -p res-gpu-small
#SBATCH --job-name vqa_topk-1500_lx-lstm_unfreeze-heads_loss-none_lr-5e-6_rubi-none 
#SBATCH --gres gpu:1 
#SBATCH -o vqa_topk-1500_lx-lstm_unfreeze-heads_loss-none_lr-5e-6_rubi-none.out

cd ../../..
source venvs/a_vs_c/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python -W ignore main.py \
    --jobname vqa_topk-1500_lx-lstm_unfreeze-heads_loss-none_lr-5e-6_rubi-none \
    --dataset vqa \
    --topk 1500 \
    --model lx-lstm \
    --loss none \
    --epochs 1000 \
    --bsz 32 \
    --val_bsz 100 \
    --device 0 \
    --unfreeze heads \
    --num_workers 2 \
    --lr 0.000005 \
    --rubi none \
    --wandb 
