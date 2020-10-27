#!/bin/bash
source ../../venvs/pvse/bin/activate
cd ../../pvse
python eval.py \
    --data_name mrw \
    --num_embeds 5 \
    --img_attention \
    --txt_attention \
    --max_video_length 4 \
    --legacy \
    --ckpt ./ckpt/mrw_pvse.pth
