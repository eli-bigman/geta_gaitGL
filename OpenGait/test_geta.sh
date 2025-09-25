#!/bin/bash
# Testing script for GaitGL with GETA
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main_geta.py --cfgs ./configs/gaitgl/gaitgl_geta.yaml --phase test