#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name gqa_normonly_BUTD_loss-avsc_lr-3e-6_rubi-none 
#SBATCH --gres gpu:2 
#SBATCH -o ../../../checkpoints/gqa_normonly_BUTD_loss-avsc_lr-3e-6_rubi-none.out

cd ../../..
source venv/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python main.py \
    --jobname gqa_normonly_BUTD_loss-avsc_lr-3e-6_rubi-none \
    --dataset gqa \
    --norm_ans_only \
    --model BUTD \
    --loss avsc \
    --epochs 50 \
    --bsz 64 \
    --val_bsz 64 \
    --device 1 \
    --unfreeze heads \
    --num_workers 4 \
    --lr 3e-6 \
    --rubi none \
    --wandb 
