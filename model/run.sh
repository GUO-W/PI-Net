#!/bin/bash

#cd $1
export PYTHONNOUSERSITE=1
export PYTHONPATH=$PYTHONPATH:/local_scratch/wguo/repos/3DMPPE/PINET-release

## train
#python traintest.py --gpu 0  --bz 4 --lr 1e-5 --test

## test
##python test.py --gpu 0 --bz 4 --lr 1e-5 --test_epoch all
python test.py --gpu 0 --bz 4 --lr 1e-5 --test_epoch 24

