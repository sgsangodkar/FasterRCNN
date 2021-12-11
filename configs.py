#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 12:10:50 2021

@author: sagar
"""



class rpn_configs:
    in_channels=512
    out_channels=512
    anchors_per_location=9
    receptive_field=16
    scales = [8,16,32]
    ratios = [0.5,1,2]
    samples_per_mini_batch=256
    pos_ratio=0.5
    pos_iou_thresh=0.7
    neg_iou_thresh=0.3
    samples_per_min_batch=256
    pos_ratio=0.5
    
class train_configs:
    phase='Train'
    epochs=4
    lr=0.001
    

class data_configs:
    data_path = '../voc_data/VOCdevkit/VOC2012'
    data_type = 'trainval'
    min_size = 600
    random_flips = True