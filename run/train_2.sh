#!/bin/bash
# nohup jupyter nbconvert --to notebook --execute --allow-errors --inplace ../2.model_bigru.ipynb > run_2.log 2>&1 &
nohup python3 ../2.model_bigru.py > run_2.log 2>&1 &
