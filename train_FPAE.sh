#/bin/bash
#
# Chirag Agarwal <chiragagarwall12.gmail.com>
# 2020

CUDA_VISIBLE_DEVICES=0 python train_FPAE.py
imgcat ./label_0_FPAE.png ./label_1_FPAE.png
