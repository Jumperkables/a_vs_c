#!/bin/bash
cd ..
source venvs/a_vs_c/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python probe_attn_heads.py \
    --dataset GQA-absMixed \
    --unfreeze heads \
    --model dual-lx-lstm \
    --checkpoint_path 'gqa-absMixed_dual-lx-lstm_unfreeze-heads_loss-avsc_norm-qtype_lr-5e-6_rubi-none_dls-linear-epoch=08-valid_acc=0.75.ckpt'
