#!/bin/bash
cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd
source ../../venvs/a_vs_c/bin/activate
python tvqa_avc_stats.py
