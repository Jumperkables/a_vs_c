#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )" # cd to scripts location to avoid any weird errors executing from different locations
source ../../venvs/a_vs_c/bin/activate
python ../../PLACEHOLDER.py \
    --dsets TVQA PVSE AVSD \
    --n_examples 5

