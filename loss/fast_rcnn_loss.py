# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F


def fast_rcnn_cls_loss(cls_op, cls_gt):
    loss = F.cross_entropy(cls_op, cls_gt)
    return loss
    
def fast_rcnn_reg_loss(reg_op, reg_gt, cls_op):
    cls_op = torch.argmax(cls_op, dim=1)
    reg_op = reg_op[cls_op>0]
    reg_gt = reg_gt[cls_op>0]
    cls_op = cls_op[cls_op>0]
    reg_op = reg_op.view(len(cls_op), -1, 4)
    reg_op = reg_op[torch.arange(len(cls_op)), cls_op-1]
    loss = F.smooth_l1_loss(reg_op, reg_gt)
    return loss
 
def fast_rcnn_loss(cls_op, cls_gt, reg_op, reg_gt):   
    cls_loss = fast_rcnn_cls_loss(cls_op, cls_gt)
    reg_loss = fast_rcnn_reg_loss(reg_op, reg_gt, cls_op)
  
    return cls_loss, reg_loss
