# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F


def fast_rcnn_cls_loss(cls_op, cls_gt):
    loss = F.cross_entropy(cls_op, cls_gt)
    return loss
    
def fast_rcnn_reg_loss(reg_op, reg_gt, cls_gt):
    #cls_op = torch.argmax(cls_op, dim=1)
    #print(cls_gt)
    mask = cls_gt>0
    if sum(mask):
        reg_op = reg_op[mask]
        reg_gt = reg_gt[mask]
        cls_gt = cls_gt[mask]
        reg_op = reg_op.view(len(cls_gt), -1, 4)
        reg_op = reg_op[torch.arange(len(cls_gt)), cls_gt-1]
        loss = F.smooth_l1_loss(reg_op, reg_gt, beta=3)
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
 
def fast_rcnn_loss(cls_op, cls_gt, reg_op, reg_gt):   
    cls_loss = fast_rcnn_cls_loss(cls_op, cls_gt)
    reg_loss = fast_rcnn_reg_loss(reg_op, reg_gt, cls_gt)
  
    return cls_loss, reg_loss
