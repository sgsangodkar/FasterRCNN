# -*- coding: utf-8 -*-

import torch
from utils.misc import bbox2reg, obtain_iou_matrix, get_xywh

def target_gen_fast_rcnn(rois, bboxes_gt, classes_gt):
    n_samples = 128
    pos_ratio = 0.5
    iou_max = 0.5
    iou_min = 0.1
        
    iou_matrix = obtain_iou_matrix(rois, bboxes_gt)      
    max_ious, gt_id = torch.max(iou_matrix, axis=1)
    
    pos_indx = torch.where(max_ious>=iou_max)[0]   
    neg_indx = torch.where((max_ious<iou_max)&(max_ious>iou_min))[0]  
    
    n_pos_req = int(n_samples*pos_ratio)
    n_pos = min(len(pos_indx), n_pos_req)
    n_neg_req = n_samples - n_pos
    n_neg = min(len(neg_indx), n_neg_req)
    
    if len(pos_indx)>0:
        pos_indx = pos_indx[torch.randint(len(pos_indx), (n_pos,))] 
        
    if len(neg_indx) > 0:   
        neg_indx = neg_indx[torch.randint(len(neg_indx), (n_neg,))]        

    indx_v = torch.hstack([pos_indx, neg_indx])
       
    #cls_gt_v = torch.tensor(list(map(lambda x:classes_gt[x]+1, gt_id[indx_v])))
    #cls_gt_v[n_neg:]=0
    #bboxes_v = torch.vstack(list(map(lambda x:bboxes_gt[x], gt_id[indx_v])))
    cls_gt_v = classes_gt[gt_id[indx_v]]+1
    cls_gt_v[n_neg:]=0
    bboxes_v = bboxes_gt[gt_id[indx_v]]
    
    rois_v = rois[indx_v]       
    reg_gt_v = bbox2reg(rois_v, bboxes_v) 
    x1, y1, x2, y2 = torch.split(rois_v, 1, dim=1)
    x, y, w, h = get_xywh(x1, y1, x2, y2)
    rois_v = torch.hstack((x, y, w, h))
    """
    TO DO Normalisation
    """
    return rois_v, cls_gt_v, reg_gt_v
