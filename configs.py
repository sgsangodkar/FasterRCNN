#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 12:10:50 2021

@author: sagar
"""
            


class config:
    in_chan=512
    out_chan=512
    anchors_per_location=9
    num_classes = 20
    
    roi_pool_size = 7
    receptive_field = 16
    
    epochs=4
    lr=0.001
    

    data_path = '../voc_data/VOCdevkit/VOC2012'
    data_type = 'trainval'
    min_size = 600
    random_flips = True
    
    visualize = True
    log = True