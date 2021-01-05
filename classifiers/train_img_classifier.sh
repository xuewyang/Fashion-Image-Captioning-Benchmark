#!/usr/bin/env bash
DATA_FOLDER=/home/xuewyang/Xuewen/Research/data/FACAD/images

CUDA_VISIBLE_DEVICES='1,0' python train_img_classifier.py --batch_size 120 --lr 5e-4 \
--data_folder $DATA_FOLDER \
--model_folder /home/xuewyang/Xuewen/Research/model/fashion/resnet/resnet_fashion_lr_5e-4