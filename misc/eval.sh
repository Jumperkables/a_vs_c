#!/bin/bash
source ../venvs/a_vs_c/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python ../eval.py \
    --tier "val" \
    --checkpoint_path "checkpoints/gqa_dual-lx-lstm_unfreeze-heads_loss-avsc_norm-nsubj_lr-5e-6_rubi-none_dls-4th" \
    --score_file_name "high_scores.txt" \
    --scenes "val_sceneGraphs.json" \
    --questions "val_balanced_questions.json" \
    --choices "val_choices.json" \
    --predictions "high_predictions.json" \
    --attentions "high_attentions.json" \
    --consistency \
    --grounding \
    --objectFeatures 
