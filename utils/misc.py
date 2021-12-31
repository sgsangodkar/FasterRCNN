#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 12:31:47 2021

@author: sagar
"""
import torch
import numpy as np
import cv2

def obtain_iou_matrix(bbox_a, bbox_b):
    # top left
    tl = torch.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = torch.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = torch.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = torch.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = torch.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


def unmap(data, count, index, fill=-1):
    if len(data.shape) == 1:
        ret = torch.zeros((count,), dtype=data.dtype)
        ret = ret.to(data.device)
        ret.fill_(fill)
        ret[index] = data
    else:
        ret = torch.zeros((count,) + data.shape[1:], dtype=data.dtype)
        ret = ret.to(data.device)
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


def hflip_bboxes(bbox, size):
    H, W = size
    bbox_clone = bbox.clone()
    bbox_clone[:, 0] = W - bbox[:,2]
    bbox_clone[:, 2] = W - bbox[:,0]
    return bbox_clone


def hflip_img(img):
    # Makes a copy automatically
    img = torch.flip(img, dims=[2])
    return img

def visualize_bboxes(img, bboxes):
    img = img.squeeze().permute(1,2,0).cpu()
    img_np = np.ascontiguousarray(img)
    means = np.array((0.485, 0.456, 0.406))
    stds = np.array((0.229, 0.224, 0.225))
    img_np = (img_np*stds)+means
    img_np = np.clip(img_np, 0,1)
    img_np = (img_np*255).astype(np.uint8)
   
    for i in range(len(bboxes)):
        x1,y1,x2,y2 = np.array(bboxes[i], dtype=np.uint16)
        img_np = cv2.rectangle(img_np, (x1,y1), (x2,y2), (0,255,0), 3)
            
    return img_np.copy()
