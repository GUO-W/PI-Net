#!/bin/bash

export PYTHONNOUSERSITE=1

## train
#python traintest.py --gpu 0  --bz 4 --lr 1e-5 --test

## test
##python test.py --gpu 0 --bz 4 --lr 1e-5 --test_epoch all
python test.py --gpu 0 --bz 4 --lr 1e-5 --test_epoch 24

