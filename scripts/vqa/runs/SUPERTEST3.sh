#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 28G
#SBATCH -p res-gpu-small
#SBATCH --job-name SUPERTEST3 
#SBATCH --gres gpu:1 
#SBATCH -o SUPERTEST3.out

cd ../../..
source venvs/a_vs_c/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
export CUDA_LAUNCH_BLOCKING=1
python -W ignore main.py \
    --jobname SUPERTEST3 \
    --dataset vqa \
    --topk 1500 \
    --model lx-lstm \
    --loss avsc \
    --epochs 30 \
    --bsz 32 \
    --val_bsz 100 \
    --device 0 \
    --unfreeze heads \
    --num_workers 4 \
    --lr 0.000005 \
    --rubi none \
    --wandb 
