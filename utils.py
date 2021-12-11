#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 12:31:47 2021

@author: sagar
"""
import numpy as np

def generate_anchors(img_size, receptive_field, scales, ratios):  
    
    cx, cy = ((receptive_field-1)/2, (receptive_field-1)/2)  
    img_sizey, img_sizex = img_size  
    f_sizex = img_sizex//receptive_field
    f_sizey = img_sizey//receptive_field
   
    num_centers = (f_sizex*f_sizey)
    ctrs = np.zeros((num_centers,2))
    idx = 0
    for x in range(f_sizex):
        for y in range(f_sizey):
            ctrs[idx, 0] = cx + x*receptive_field
            ctrs[idx, 1] = cy + y*receptive_field
            idx +=1    
            
    anchors = []
    for idx in range(num_centers):
        cx, cy = ctrs[idx]
        for i in range(len(ratios)):
            w_a = np.round(np.sqrt((receptive_field*receptive_field)/ratios[i]))
            h_a = np.round(w_a*ratios[i])          
            for j in range(len(scales)):
                anchors.append([
                                int(cx-0.5*(w_a*scales[j]-1)), int(cy-0.5*(h_a*scales[j]-1)),
                                int(cx+0.5*(w_a*scales[j]-1)), int(cy+0.5*(h_a*scales[j]-1))
                              ])
    return np.array(anchors)


 
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

def obtain_iou_matrix(bboxes1, bboxes2):
    iou_matrix = np.zeros((len(bboxes1), len(bboxes2)))
    for i, bbox1 in enumerate(bboxes1):
        for j, bbox2 in enumerate(bboxes2):
            iou_matrix[i,j] = calculate_iou(bbox1, bbox2)
    return iou_matrix

def unmap(data, count, index, fill=-1):
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret



def get_xywh(bboxes):
    w = bboxes[:,2] - bboxes[:,0]
    h = bboxes[:,3] - bboxes[:,1]
    x = bboxes[:,0] + 0.5*w
    y = bboxes[:,1] + 0.5*h      
    return x, y, w, h

def bbox2reg(bboxes, anchors):
    x, y, w, h = get_xywh(bboxes)
    xa, ya, wa, ha = get_xywh(anchors)
    tw = np.log(w/wa)
    th = np.log(h/ha)
    tx = (x-xa)/wa
    ty = (y-ya)/ha
    return np.stack((tx, ty, tw, th), axis=1)
    
   
