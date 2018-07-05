#!/bin/bash
GPUID=1

echo 'start preprocess.sh'
./preprocess.sh
echo 'end preprocess.sh'

echo 'start training model'
CUDA_VISIBLE_DEVICES=$GPUID python train.py -no_CRF -gpu 0 -epoch 10 -out_dir result_noCRF_epoch10_batchsize512_hidden_size128/ -tensorboard
echo 'end training model'
