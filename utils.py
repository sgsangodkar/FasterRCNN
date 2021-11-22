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

def rm_cross_boundary_anchors(anchors, img_size):
    sizex, sizey = img_size
    new_anchors = []
    for anchor in anchors:
        if(anchor[0]>0 and
           anchor[1]>0 and
           anchor[2]<sizex and
           anchor[3]<sizey): 
               new_anchors.append(anchor)
    return new_anchors
    
def generate_anchors(img_size, receptive_field, scales, ratios):  
    cx, cy = ((receptive_field-1)/2, (receptive_field-1)/2)  
    img_sizex, img_sizey = img_size  
    f_sizex = img_sizex//receptive_field
    f_sizey = img_sizey//receptive_field
   
    num_centers = (f_sizex*f_sizey)
    anchors = np.zeros((num_centers*len(scales)*len(ratios),4))
    ctrs = np.zeros((num_centers,2))
    idx = 0
    for x in range(f_sizex):
        for y in range(f_sizey):
            ctrs[idx, 0] = cx + x*receptive_field
            ctrs[idx, 1] = cy + y*receptive_field
            idx +=1       
    for idx in range(num_centers):
        cx, cy = ctrs[idx]
        for i in range(len(ratios)):
            w_a = np.round(np.sqrt((16*16)/ratios[i]))
            h_a = np.round(w_a*ratios[i])          
            for j in range(len(scales)):
                anchors[idx*len(scales)*len(ratios)+len(ratios)*i+j] = [
                                cx-0.5*(w_a*scales[j]-1), cy-0.5*(h_a*scales[j]-1),
                                cx+0.5*(w_a*scales[j]-1), cy+0.5*(h_a*scales[j]-1)
                            ]
    anchors = rm_cross_boundary_anchors(anchors, img_size)
    return anchors

def calculate_iou(anchor, bbox):
    int_x = min(anchor[2], bbox[2]) - max(anchor[0], bbox[0])
    int_y = min(anchor[3], bbox[3]) - max(anchor[1], bbox[1])
    if int_x>0 and int_y>0:
        intersection = int_x*int_y
    else:
        return 0
    
    anchor_area = (anchor[2]-anchor[0])*(anchor[3]-anchor[1])
    bbox_area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
    union = anchor_area + bbox_area - intersection
    
    return intersection/union

def obtain_positive_anchors(anchors, bboxes):
    positives = []
    for bbox in bboxes:
        max_iou = 0
        for anchor in anchors:
            iou = calculate_iou(anchor, bbox)
            if iou > max_iou:
                max_iou = iou
                max_anchor = anchor
            if iou > 0:
                positives.append(anchor)
        if max_iou <= 0.7:
            positives.append(max_anchor)
    return positives

  
if __name__ == '__main__': 
    t = time.time()
    anchors = generate_anchors((600, 600), 16, [8,16,32], [0.5,1,2])
    refined_anchors = rm_cross_boundary_anchors(anchors, (600,600))

    print(time.time() - t)
    
    img = np.zeros((600,600))
    
    for idx in range(len(refined_anchors)):
        start = (int(refined_anchors[idx][0]), int(refined_anchors[idx][1]))
        end = (int(refined_anchors[idx][2]), int(refined_anchors[idx][3]))
        img = cv2.rectangle(img, start, end, (255, 255, 255), 1)
        
    plt.imshow(img, 'gray')