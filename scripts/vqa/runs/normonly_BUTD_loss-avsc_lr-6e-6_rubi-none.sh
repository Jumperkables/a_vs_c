#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name vqa_normonly_BUTD_loss-avsc_lr-6e-6_rubi-none 
#SBATCH --gres gpu:1 
#SBATCH -o ../../../checkpoints/vqa_normonly_BUTD_loss-avsc_lr-6e-6_rubi-none.out

cd ../../..
source venvs/a_vs_c/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python main.py \
    --jobname vqa_normonly_BUTD_loss-avsc_lr-6e-6_rubi-none \
    --dataset vqa \
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
    --lr 6e-6 \
    --rubi none \
    --wandb 
