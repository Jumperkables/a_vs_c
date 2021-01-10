#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source ../../../../venvs/a_vs_c/bin/activate
python ../../../../misc/BERT_analysis.py \
    --purpose bertqa_logits \
    --model lxmert-qa \
    --device 0
