#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 17:59:22 2021

@author: sagar
"""
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

    
def generate_anchors(stride, scales, ratios):  
    anchors = np.zeros((len(scales)*len(ratios),4))
    
    base_anchor = np.array([0,0,stride-1,stride-1])
    w = base_anchor[2]-base_anchor[0] + 1
    h = base_anchor[3]-base_anchor[1] + 1
    cx = base_anchor[0]+(w-1)/2
    cy = base_anchor[1]+(h-1)/2

    for i in range(len(ratios)):
        w_a = np.round(np.sqrt((w*h)/ratios[i]))
        h_a = np.round(w_a*ratios[i])
        for j in range(len(scales)):
            anchors[len(scales)*i+j] = np.array([
                            cx-0.5*(w_a*scales[j]-1), cy-0.5*(h_a*scales[j]-1),
                            cx+0.5*(w_a*scales[j]-1), cy+0.5*(h_a*scales[j]-1)
                        ])
    
    return anchors

if __name__ == '__main__': 
    t = time.time()
    anchors = generate_anchors(16, [8,16,32], [0.5,1,2])
    print(time.time() - t)
    print(anchors)
    
    img = np.zeros((800,800))
    anchors+=400
    
    for idx in range(len(anchors)):
        start = (int(anchors[idx,0]), int(anchors[idx,1]))
        end = (int(anchors[idx,2]), int(anchors[idx,3]))
        img = cv2.rectangle(img, start, end, (255, 255, 255), 10)
        
    plt.imshow(img, 'gray')