#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name gqa_full_lxmert_loss-simlex-avsc_lr-1e-6_rubi-none 
#SBATCH --gres gpu:1 
#SBATCH -o ../../../checkpoints/gqa_full_lxmert_loss-simlex-avsc_lr-1e-6_rubi-none.out

cd ../../..
source venv/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python main.py \
    --jobname gqa_full_lxmert_loss-simlex-avsc_lr-1e-6_rubi-none \
    --dataset gqa \
    --model lxmert \
    --loss avsc \
    --epochs 200 \
    --bsz 64 \
    --val_bsz 64 \
    --device 0 \
    --num_workers 4 \
    --lr 1e-6 \
    --rubi none \
    --wandb \
    #--norm_ans_only simlex \

