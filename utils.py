#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 17:59:22 2021

@author: sagar
"""
import cv2
import copy
import time
import numpy as np
import matplotlib.pyplot as plt

    
def generate_anchors(img_size, anchor_params):  
    
    receptive_field = anchor_params['receptive_field']
    scales = anchor_params['scales']
    ratios = anchor_params['ratios']
    
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

def obtain_iou_matrix(anchors, bboxes):
    iou_matrix = np.zeros((len(anchors), len(bboxes)))
    for i, anchor in enumerate(anchors):
        for j, bbox in enumerate(bboxes):
            iou_matrix[i,j] = calculate_iou(anchor, bbox)
    return iou_matrix

def assign_label_and_gt_bbox(anchors, bboxes, img_size, allowed_border=0):
    anchor_labels = np.empty(len(anchors), dtype=np.int32)
    anchor_labels.fill(-1)
    
    iou_matrix = obtain_iou_matrix(anchors, bboxes)
    max_iou_per_anchor = np.max(iou_matrix, axis=1)
    min_iou_per_anchor = np.min(iou_matrix, axis=1)
    anchor_labels[max_iou_per_anchor>0.7] = 1
    anchor_labels[min_iou_per_anchor<0.3] = 0
    
    if np.sum(anchor_labels==1)==0:
        max_iou_per_bbox = np.max(iou_matrix, axis=0)
        for i in range(len(bboxes)):
            anchor_labels[iou_matrix[:,i]==max_iou_per_bbox[i]] = 1

    inds_inside = np.where(
        (anchors[:, 0] >= -allowed_border) &
        (anchors[:, 1] >= -allowed_border) &
        (anchors[:, 2] < img_size[1] + allowed_border) &  # width
        (anchors[:, 3] < img_size[0] + allowed_border)    # height
    )[0]
    inds_outside = set(np.arange(len(anchors))).difference(set(inds_inside))
    inds_outside = np.array(list(inds_outside))
    anchor_labels[inds_outside] = -1
    
    return anchor_labels, np.argmax(iou_matrix,1)       

def scale_bboxes(bboxes, sx, sy):
    bboxes_scaled = copy.deepcopy(bboxes)
    for i, bbox in enumerate(bboxes):
        bboxes_scaled[i] = [int(bbox[0]*sx), 
                            int(bbox[1]*sy), 
                            int(bbox[2]*sx), 
                            int(bbox[3]*sy)
                           ]
        
    return bboxes_scaled

def get_txtytwth(anchor, bbox):
    w = bbox[2] - bbox[0]+1
    h = bbox[3] - bbox[1]+1
    x = bbox[0] + 0.5*w
    y = bbox[1] + 0.5*h
    
    w_a = anchor[2] - anchor[0]+1
    h_a = anchor[3] - anchor[1]+1
    x_a = anchor[0] + 0.5*w_a
    y_a = anchor[1] + 0.5*h_a
    
    tw = np.log(w/w_a)
    th = np.log(h/h_a)
    tx = (x-x_a)/w_a
    ty = (y-y_a)/h_a
    
    return tx, ty, tw, th

def get_t_parameters(anchors, bboxes, gt_bboxes_id):
    anchors_final = []
    for i in range(len(anchors)):
        anchors_final.append(get_txtytwth(anchors[i], bboxes[gt_bboxes_id[i]]))
    return np.array(anchors_final, dtype=np.float32)
"""
if __name__ == '__main__': 
    t = time.time()
    anchors = generate_anchors((600, 600), 16, [8,16,32], [0.5,1,2])

    print(time.time() - t)
    
    img = np.zeros((600,600))
    
    for idx in range(2):
        start = (int(anchors[idx][0]), int(anchors[idx][1]))
        end = (int(anchors[idx][2]), int(anchors[idx][3]))
        img = cv2.rectangle(img, start, end, (255, 255, 255), 1)
        
    plt.imshow(img, 'gray')
"""    
    
if __name__ == '__main__':
    anchor_params = dict(receptive_field=16,
                     scales = [8,16,32],
                     ratios = [0.5,1,2]
                )
    anchors = generate_anchors([600,600], anchor_params)

    print(anchors)
    print(anchors.shape)