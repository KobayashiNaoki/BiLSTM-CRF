#!/bin/bash
GPUID=0

echo 'start preprocess.sh'
#./preprocess.sh
echo 'end preprocess.sh'

echo 'start training model'
CUDA_VISIBLE_DEVICES=$GPUID python train.py -no_CRF -gpu 0 -epoch 10 -out_dir result_noFeature_noCRF_epoch10/ -tensorboard\
                    -no_features &
CUDA_VISIBLE_DEVICES=$GPUID python train.py -no_CRF -gpu 0 -epoch 10 -out_dir result_noCRF_epoch10/ -tensorboard &
CUDA_VISIBLE_DEVICES=$GPUID python train.py -no_CRF -gpu 0 -epoch 10 -out_dir result_expand_noCRF_epoch10/ -tensorboard\
                    -train_file data/train_expand.txt -valid_file data/valid_expand.txt &
wait
echo 'end training model'
