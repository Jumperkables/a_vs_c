#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name SUPERTEST3 
#SBATCH --gres gpu:1 
#SBATCH -o SUPERTEST3.out

cd ../../..
source venvs/a_vs_c/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python -W ignore main.py \
    --jobname SUPERTEST3 \
    --dataset gqa \
    --model lx-lstm \
    --loss avsc \
    --epochs 30 \
    --val_bsz 100 \
    --device 0 \
    --unfreeze heads \
    --num_workers 4 \
    --lr 0.000005 \
    --rubi none \
    --wandb 
