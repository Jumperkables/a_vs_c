#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name vqacp_normonly_BUTD_loss-avsc_lr-3e-6_rubi-none 
#SBATCH --gres gpu:1 
#SBATCH -o ../../../checkpoints/vqacp_normonly_BUTD_loss-avsc_lr-3e-6_rubi-none.out

cd ../../..
source venv/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python main.py \
    --jobname vqacp_normonly_BUTD_loss-avsc_lr-3e-6_rubi-none \
    --dataset vqacp \
    --min_ans_occ 2 \
    --norm_ans_only \
    --model BUTD \
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
