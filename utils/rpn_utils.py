#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 04:55:49 2021

@author: sagar
"""
import cv2
import torch
import numpy as np
from torchvision.ops import nms
from utils.misc import bbox2reg, reg2bbox, unmap, obtain_iou_matrix
import torch.nn.functional as F
from configs import config

def gen_anchors(img_size, receptive_field, scales, ratios):  
    
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
            # removed np round
            w_a = np.sqrt((receptive_field*receptive_field)/ratios[i])
            h_a = w_a*ratios[i]         
            for j in range(len(scales)):
                anchors.append([
                                cx-0.5*(w_a*scales[j]-1), cy-0.5*(h_a*scales[j]-1),
                                cx+0.5*(w_a*scales[j]-1), cy+0.5*(h_a*scales[j]-1)
                              ])
                #removed int()
    #print(anchors[0])
    return torch.tensor(anchors, dtype=torch.float32)

"""   
def target_gen_rpn(anchors, bboxes_gt, img_size):

    num_anchors = anchors.shape[0]  
    num_bboxes = bboxes_gt.shape[0]
    
    indx_v = torch.where(
                    (anchors[:, 0] >= 0) &
                    (anchors[:, 1] >= 0) &
                    (anchors[:, 2] < img_size[1]) &  # width
                    (anchors[:, 3] < img_size[0])    # height
                    )[0]
    
    anchors_v = anchors[indx_v]

    num_anchors_v = anchors_v.shape[0]
    iou_matrix = obtain_iou_matrix(anchors_v, bboxes_gt)
    cls_gt_v = torch.empty(num_anchors_v, dtype=torch.int64) #long
        
    max_iou_per_anchor = torch.max(iou_matrix, axis=1)[0]
    min_iou_per_anchor = torch.min(iou_matrix, axis=1)[0]
    cls_gt_v[max_iou_per_anchor>0.7] = 1
    cls_gt_v[min_iou_per_anchor<0.3] = 0
            
    max_iou = torch.max(iou_matrix)
    for i in range(num_bboxes):
        cls_gt_v[iou_matrix[:,i]==max_iou] = 1
                
    bboxes_v = bboxes_gt[torch.argmax(iou_matrix, axis=1)]
    
    reg_gt_v = bbox2reg(anchors_v, bboxes_v)
    print(anchors_v.shape, bboxes_v.shape, reg_gt_v.shape)
    
    cls_gt = unmap(cls_gt_v, num_anchors, indx_v, fill=-1)
    reg_gt = unmap(reg_gt_v, num_anchors, indx_v, fill=-1)
        
    return cls_gt, reg_gt
"""

def target_gen_rpn(anchors, bboxes_gt, img_size):
    n_samples = 128
    pos_ratio = 0.5
    
    indx_v = torch.where(
                    (anchors[:, 0] >= 0) &
                    (anchors[:, 1] >= 0) &
                    (anchors[:, 2] < img_size[1]) &  # width
                    (anchors[:, 3] < img_size[0])    # height
                    )[0]
    #print(len(indx_v))
    anchors_v = anchors[indx_v]
    iou_matrix = obtain_iou_matrix(anchors_v, bboxes_gt)
    
    ## Positive and Negative Anchors Selection
    cls_gt_v = torch.zeros(len(anchors_v), dtype=torch.int64) #long      
    argmax_iou_per_anchor = torch.argmax(iou_matrix, axis=1)
    max_iou_per_anchor = iou_matrix[np.arange(len(anchors_v)), argmax_iou_per_anchor]
    cls_gt_v[max_iou_per_anchor<0.3] = 0   
    #cls_gt_v[max_iou_per_anchor>=0.7] = 1
        
    argmax_iou_per_gt = torch.argmax(iou_matrix, axis=0)
    max_iou_per_gt = iou_matrix[argmax_iou_per_gt, np.arange(len(bboxes_gt))]
    for i in range(len(bboxes_gt)):
        cls_gt_v[iou_matrix[:,i]==max_iou_per_gt[i]] = 1
        
    cls_gt_v[max_iou_per_anchor>=0.7] = 1
    #print(torch.sum(cls_gt_v), torch.sum(iou_matrix), "Hi")
    ## Anchor Selection End ##

    bboxes_v = bboxes_gt[torch.argmax(iou_matrix, axis=1)]        
    reg_gt_v = bbox2reg(anchors_v, bboxes_v)
    #print(anchors_v.shape, bboxes_v.shape, reg_gt_v.shape)
   
    cls_gt = unmap(cls_gt_v, len(anchors), indx_v, fill=-1)
    reg_gt = unmap(reg_gt_v, len(anchors), indx_v, fill=-1)

    #print(reg_gt[cls_gt==1])
    #print(reg2bbox(anchors[cls_gt==1], reg_gt[cls_gt==1]))
    
    pos_indx = torch.where(cls_gt==1)[0]
    neg_indx = torch.where(cls_gt==0)[0]
    
    n_pos_req = int(n_samples*pos_ratio)
    n_pos = min(len(pos_indx), n_pos_req)
    n_neg_req = n_samples - n_pos
    n_neg = min(len(neg_indx), n_neg_req)
    #print(n_pos, n_neg)

    pos_indx = pos_indx[torch.randperm(len(pos_indx))[:n_pos]]  
    neg_indx = neg_indx[torch.randperm(len(neg_indx))[:n_neg]] 
    indx_valid = torch.hstack([pos_indx, neg_indx])
    #uniques, counts = indx_valid.unique(return_counts=True)
    #print(len(indx_valid), len(pos_indx), len(neg_indx), len(uniques), len(counts==1))

    
    #Following can be optimised using torch.zeros() for cls_gt
    cls_gt_final = torch.zeros(cls_gt.shape, dtype=torch.long)-1
    cls_gt_final[indx_valid] = cls_gt[indx_valid]
    #print(len(pos_indx), len(neg_indx))
    #print(torch.sum(cls_gt==1), torch.sum(cls_gt==0))
    

############
    
    return cls_gt_final, reg_gt
        
"""
def assign_labels_bboxes(anchors, bboxes_gt):  
    num_bboxes = bboxes_gt.shape[0]
    num_anchors = anchors.shape[0]
    iou_matrix = obtain_iou_matrix(anchors, bboxes_gt)
    labels = torch.empty(num_anchors, dtype=torch.int64) #long
    labels.fill(-1)

    max_iou_per_anchor = torch.max(iou_matrix, axis=1)
    min_iou_per_anchor = torch.min(iou_matrix, axis=1)
    labels[max_iou_per_anchor>0.7] = 1
    labels[min_iou_per_anchor<0.3] = 0
    
    max_iou = torch.max(iou_matrix)
    for i in range(num_bboxes):
        labels[iou_matrix[:,i]==max_iou] = 1
        
    return labels, bboxes_gt[torch.argmax(iou_matrix, axis=1)]
"""
def gen_rois(cls_op, reg_op, anchors, img_size):
    nms_thresh = 0.7
    top_n = 1000
    cls_op = F.softmax(cls_op, dim=1)
    fg_scores = cls_op[:,1]
    bboxes_op = reg2bbox(anchors, reg_op)
    #print(reg_op.dtype, anchors.dtype, bboxes_op.dtype)
    
    torch.clip(bboxes_op[:,0], 0, img_size[1]-1)
    torch.clip(bboxes_op[:,1], 0, img_size[0]-1)
    torch.clip(bboxes_op[:,2], 0, img_size[1]-1)
    torch.clip(bboxes_op[:,3], 0, img_size[0]-1)
    #print(bboxes_op.shape, bboxes_op.dtype)
    #print(fg_scores.shape, fg_scores.dtype)
    
    min_size = config.roi_pool_size
    hs = bboxes_op[:, 3] - bboxes_op[:, 1]
    ws = bboxes_op[:, 2] - bboxes_op[:, 0]
    keep = np.where((hs >= min_size) & (ws >= min_size))[0]
    bboxes_op = bboxes_op[keep, :]
    fg_scores = fg_scores[keep]
    
    indices = nms(bboxes_op, fg_scores, nms_thresh)
    rois = bboxes_op[indices[:top_n]]
    
    return rois
    