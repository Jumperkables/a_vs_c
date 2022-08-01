#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name vqacp2_mao-9_normonly-expanded-nc-pt7_lxmert_loss-default_lr-1e-6_rubi-none 
#SBATCH --gres gpu:1 
#SBATCH -o ../../../../checkpoints/vqacp2_mao-9_normonly-expanded-nc-pt7_lxmert_loss-default_lr-1e-6_rubi-none.out

cd ../../../..
source venv/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python main.py \
    --jobname vqacp2_mao-9_normonly-expanded-nc-pt7_lxmert_loss-default_lr-1e-6_rubi-none \
    --dataset vqacp2 \
    --model lxmert \
    --loss default \
    --epochs 300 \
    --bsz 64 \
    --val_bsz 64 \
    --device 0 \
    --num_workers 4 \
    --lr 1e-6 \
    --rubi none \
    --min_ans_occ 9 \
    --wandb \
    --norm_ans_only expanded \
    --norm_clipping 0.7 \

