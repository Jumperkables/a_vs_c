#!/bin/bash
#SBATCH -p part0
#SBATCH --job-name BERT_analysis 
#SBATCH --ntasks 6
#SBATCH --gres gpu:1
#SBATCH -o ../../.results/NA.out

# Script to test the function of assoc_vs_ctgrcl.py model

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source ../../venvs/a_vs_c/bin/activate
python ../../misc/BERT_analysis.py 
