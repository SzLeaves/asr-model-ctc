#!/bin/bash
# nohup jupyter nbconvert --to notebook --execute --allow-errors --inplace ../3.model_wavenet.ipynb > run_3.log 2>&1 &
nohup python3 3.model_wavenet.py > run_3.log 2>&1 &