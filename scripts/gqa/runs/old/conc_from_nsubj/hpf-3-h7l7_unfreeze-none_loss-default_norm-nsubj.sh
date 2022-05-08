#!/bin/bash
#SBATCH --qos long-high-prio
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 7-00:00
#SBATCH -x gpu[0-3]
#SBATCH --mem 12G
#SBATCH -p res-gpu-small
#SBATCH --job-name gqa_hpf-3-h7l7_unfreeze-none_loss-default_norm-nsubj 
#SBATCH --gres gpu:1 
#SBATCH -o gqa_hpf-3-h7l7_unfreeze-none_loss-default_norm-nsubj.out

cd ../../../..
source venv/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python main.py \
    --jobname gqa_hpf-3-h7l7_unfreeze-none_loss-default_norm-nsubj \
    --dataset GQA \
    --model hpf-3 \
    --hopfield_beta_high 0.7 \
    --hopfield_beta_low 0.7 \
    --loss default \
    --norm_gt nsubj \
    --epochs 1000 \
    --bsz 32 \
    --val_bsz 100 \
    --device 1 \
    --unfreeze none \
    --num_workers 4 \
    --lr 0.00008 \
    #--wandb \