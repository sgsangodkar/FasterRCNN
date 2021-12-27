#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 12:10:50 2021

@author: sagar
"""
            


class config:
    
    num_classes = 20
    train_batch_size = 2
    epochs=4
    lr=0.001
    

    data_path = '../voc2007/VOCdevkit/VOC2007/'
    
    visualize = True
    log = True
    resume_prefix = None