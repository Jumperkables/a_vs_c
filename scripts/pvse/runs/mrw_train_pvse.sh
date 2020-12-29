#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source ../../../venvs/pvse/bin/activate
cd ../../../pvse
python train.py \
    --data_name mrw \
    --data_path ../data/pvse/ \
    --vocab_path vocab/ \
    --max_video_length 4 \
    --cnn_type resnet18 \
    --wemb_type glove \
    --margin 0.1 \
    --num_embeds 4 \
    --img_attention \
    --txt_attention \
    --mmd_weight 0.01 \
    --div_weight 0.1 \
    --batch_size 128
