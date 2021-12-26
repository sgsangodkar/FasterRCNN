# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F


def get_fast_rcnn_cls_loss(cls_op, cls_gt):
    loss = F.cross_entropy(cls_op, cls_gt)
    return loss
    
def get_fast_rcnn_reg_loss(reg_op, reg_gt, cls_gt):
    #cls_op = torch.argmax(cls_op, dim=1)
    #print(cls_gt)
    mask = cls_gt>0
    if sum(mask):
        reg_op = reg_op[mask]
        reg_gt = reg_gt[mask]
        cls_gt = cls_gt[mask]
        reg_op = reg_op.view(len(cls_gt), -1, 4)
        reg_op = reg_op[torch.arange(len(cls_gt)), cls_gt-1]
        #print(reg_op, reg_gt)
        loss = F.smooth_l1_loss(reg_op, reg_gt)
        #print("FastRCNN")
        #print(torch.abs((reg_op-reg_gt)).max())
        #print(loss, sum(mask))
    else:
        loss = torch.tensor(0.0) 

    #print(reg_op.mean(axis=0), reg_gt.mean(axis=0))
    #print('fastrcnn')
    #print(reg_op)
    #print(reg_gt)
    return loss
 
def get_fast_rcnn_loss(cls_op, cls_gt, reg_op, reg_gt): 
    #print(cls_op, cls_gt, reg_op, reg_gt)
    #print(cls_op.shape, cls_gt.shape, reg_op.shape, reg_gt.shape)
    cls_loss = get_fast_rcnn_cls_loss(cls_op, cls_gt)
    reg_loss = get_fast_rcnn_reg_loss(reg_op, reg_gt, cls_gt)
  
    return cls_loss, reg_loss
