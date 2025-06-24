#!/bin/bash

CUDA_VISIBLE_DEVICES='0' nohup python main.py \
--dataset 'MAFW' \
--workers 8 \
--epochs 25 \
--batch-size 8 \
--lr 3e-5 \
--weight-decay 1e-2 \
--print-freq 10 \
--temporal-layers 1 \
--img-size 224 \
--exper-name FINAL_224 \
--detail "" \
 > mafwtrain0.log 2>&1 &
