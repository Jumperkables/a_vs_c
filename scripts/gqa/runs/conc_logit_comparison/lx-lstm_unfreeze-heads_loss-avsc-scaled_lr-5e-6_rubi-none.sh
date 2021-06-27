#!/bin/bash
#SBATCH --qos long-high-prio
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 7-00:00
#SBATCH -x gpu[0-6]
#SBATCH --mem 28G
#SBATCH -p res-gpu-small
#SBATCH --job-name gqa_lx-lstm_unfreeze-heads_loss-avsc-scaled_lr-5e-6_rubi-none 
#SBATCH --gres gpu:1 
#SBATCH -o gqa_lx-lstm_unfreeze-heads_loss-avsc-scaled_lr-5e-6_rubi-none.out

cd ../../../..
source venvs/a_vs_c/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python main.py \
    --jobname gqa_lx-lstm_unfreeze-heads_loss-avsc-scaled_lr-5e-6_rubi-none \
    --dataset GQA \
    --model lx-lstm \
    --loss avsc-scaled \
    --epochs 1000 \
    --bsz 64 \
    --val_bsz 100 \
    --device 0 \
    --unfreeze heads \
    --num_workers 2 \
    --lr 0.000005 \
    --rubi none \
    #--wandb \