# -*- coding: utf-8 -*-

import torch
from utils.misc import bbox2reg, obtain_iou_matrix

def target_gen_fast_rcnn(rois, bboxes_gt, classes_gt, n_targets):
    pos_ratio = 0.5
    iou_max = 0.5
    iou_min = 0.0
        
    iou_matrix = obtain_iou_matrix(rois, bboxes_gt)      
    max_ious, gt_id = torch.max(iou_matrix, axis=1)
    
    pos_indx = torch.where(max_ious>=iou_max)[0]   
    neg_indx = torch.where((max_ious<iou_max)&(max_ious>=iou_min))[0]  
        
    n_pos_req = int(n_targets*pos_ratio)
    n_pos = min(len(pos_indx), n_pos_req)
    n_neg_req = n_targets - n_pos
    n_neg = min(len(neg_indx), n_neg_req)
    
    if len(pos_indx)>0:
        pos_indx = pos_indx[torch.randperm(len(pos_indx))[:n_pos]] 
        
    if len(neg_indx) > 0:   
        neg_indx = neg_indx[torch.randperm(len(neg_indx))[:n_neg]]        

    indx_v = torch.hstack([pos_indx, neg_indx])

    cls_gt_v = classes_gt[gt_id[indx_v]]+1
    cls_gt_v[n_pos:]=0
    bboxes_v = bboxes_gt[gt_id[indx_v]]
    
    rois_v = rois[indx_v] #x1,y1,x2,y2  
    reg_gt_v = bbox2reg(rois_v, bboxes_v) #x,y,w,h
   
    return rois_v, cls_gt_v, reg_gt_v
