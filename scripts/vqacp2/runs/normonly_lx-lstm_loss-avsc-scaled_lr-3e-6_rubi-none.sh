#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name vqacp2_normonly_lx-lstm_loss-avsc-scaled_lr-3e-6_rubi-none 
#SBATCH --gres gpu:1 
#SBATCH -o ../../../checkpoints/vqacp2_normonly_lx-lstm_loss-avsc-scaled_lr-3e-6_rubi-none.out

cd ../../..
source venvs/a_vs_c/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python main.py \
    --jobname vqacp2_normonly_lx-lstm_loss-avsc-scaled_lr-3e-6_rubi-none \
    --dataset vqacp2 \
    --min_ans_occ 2 \
    --norm_ans_only \
    --model lx-lstm \
    --loss avsc-scaled \
    --epochs 100 \
    --bsz 64 \
    --val_bsz 64 \
    --device 0 \
    --unfreeze heads \
    --num_workers 4 \
    --lr 3e-6 \
    --rubi none \
    --wandb 
