#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 12:31:47 2021

@author: sagar
"""
import torch
import numpy as np
 
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
    iou_matrix = torch.zeros((len(bboxes1), len(bboxes2)))
    for i, bbox1 in enumerate(bboxes1):
        for j, bbox2 in enumerate(bboxes2):
            iou_matrix[i,j] = calculate_iou(bbox1, bbox2)
    return iou_matrix

def unmap(data, count, index, fill=-1):
    if len(data.shape) == 1:
        ret = torch.empty((count,), dtype=data.dtype)
        ret.fill_(fill)
        ret[index] = data
    else:
        ret = torch.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill_(fill)
        ret[index, :] = data
    return ret



def get_xywh(x1,y1,x2,y2):
    w = x2 - x1
    h = y2 - y1
    x = x1 + 0.5*w
    y = y1 + 0.5*h      
    return x,y,w,h

def bbox2reg(anchors, bboxes):
    x1,y1,x2,y2 = torch.split(bboxes,1,dim=1)
    x, y, w, h = get_xywh(x1,y1,x2,y2)
    x1a,y1a,x2a,y2a = torch.split(anchors,1,dim=1)
    xa, ya, wa, ha = get_xywh(x1a,y1a,x2a,y2a)  
    tw = torch.log(w/wa)
    th = torch.log(h/ha)
    tx = (x-xa)/wa
    ty = (y-ya)/ha
    return torch.hstack((tx, ty, tw, th))
    
def get_x1y1x2y2(x,y,w,h):
    x1 = x-0.5*w
    y1 = y-0.5*h
    x2 = x+0.5*w
    y2 = y+0.5*h
    return x1,y1,x2,y2

def reg2bbox(anchors, reg_coor):
    x1a,y1a,x2a,y2a = torch.split(anchors,1,dim=1)
    xa, ya, wa, ha = get_xywh(x1a,y1a,x2a,y2a)
    x = reg_coor[:,0:1]*wa + xa
    y = reg_coor[:,1:2]*ha + ya
    w = torch.pow(np.e, reg_coor[:,2:3])*wa
    h = torch.pow(np.e, reg_coor[:,3:4])*ha    
    x1,y1,x2,y2 = get_x1y1x2y2(x,y,w,h)
    return torch.hstack((x1, y1, x2, y2))


def flip_bboxes(bbox, size):
    H, W = size
    bbox = bbox.copy()
    x_max = W - bbox[:, 1]
    x_min = W - bbox[:, 3]
    bbox[:, 1] = x_min
    bbox[:, 3] = x_max
    return bbox


def flip_img(img):
    # Makes a copy automatically
    img = torch.flip(img, dims=[1])
    return img