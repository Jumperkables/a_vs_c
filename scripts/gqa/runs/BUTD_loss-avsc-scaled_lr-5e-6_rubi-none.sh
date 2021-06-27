#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name gqa_BUTD_loss-avsc-scaled_lr-5e-6_rubi-none 
#SBATCH --gres gpu:1 
#SBATCH -o gqa_BUTD_loss-avsc-scaled_lr-5e-6_rubi-none.out

cd ../../..
source venvs/a_vs_c/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python -W ignore main.py \
    --jobname gqa_BUTD_loss-avsc-scaled_lr-5e-6_rubi-none \
    --dataset gqa \
    --model BUTD \
    --loss avsc-scaled \
    --epochs 30 \
    --val_bsz 100 \
    --device 0 \
    --unfreeze heads \
    --num_workers 4 \
    --lr 0.000005 \
    --rubi none \
    --wandb 