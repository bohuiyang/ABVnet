#!/bin/bash

CUDA_VISIBLE_DEVICES='0'  nohup python main.py \
--dataset 'DFEW' \
--workers 8 \
--epochs 25 \
--batch-size 8 \
--lr 4e-5 \
--weight-decay 0.6e-2 \
--print-freq 10 \
--milestones 25 \
--temporal-layers 1 \
--img-size 224 \
--exper-name FINAL_224 \
--detail "" \
 > dfewtrain0.log 2>&1 &