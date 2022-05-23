#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 12G
#SBATCH -p res-gpu-small
#SBATCH --job-name vqa2_mao-9_normonly-expanded-nc-pt4_lxmert_loss-avsc-scaled_lr-2e-6_rubi-none 
#SBATCH --gres gpu:1 
#SBATCH -o ../../../../checkpoints/vqa2_mao-9_normonly-expanded-nc-pt4_lxmert_loss-avsc-scaled_lr-2e-6_rubi-none.out

cd ../../../..
source venv/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python main.py \
    --jobname vqa2_mao-9_normonly-expanded-nc-pt4_lxmert_loss-avsc-scaled_lr-2e-6_rubi-none \
    --dataset vqa2 \
    --model lxmert \
    --loss avsc-scaled \
    --epochs 75 \
    --bsz 64 \
    --val_bsz 64 \
    --device 0 \
    --num_workers 4 \
    --lr 2e-6 \
    --rubi none \
    --min_ans_occ 9 \
    --wandb \
    --norm_ans_only expanded \
    --norm_clipping 0.4 \

