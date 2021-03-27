#!/bin/bash
cd ..
source venvs/a_vs_c/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python probe_attn_heads.py \
    --dataset VQACP \
    --topk 500 \
    --unfreeze heads \
    --model dual-lx-lstm \
#    --checkpoint_path 
#'/home/jumperkables/kable_management/projects/a_vs_c/checkpoints/gqa_dual-lx-lstm_unfreeze-heads_loss-avsc_norm-nsubj-epoch=16-valid_acc=0.48.ckpt'
