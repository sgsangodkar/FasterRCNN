#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 19:09:25 2021

@author: sagar
"""
import torch.nn.functional as F

def rpn_cls_loss(pred_logits, gt_cls):
    loss = F.cross_entropy(pred_logits, gt_cls)
    return loss

def rpn_reg_loss(pred_reg, gt_reg, gt_cls):
    loss = F.smooth_l1_loss(pred_reg[gt_cls==1], gt_reg[gt_cls==1])
    return loss

def rpn_loss(pred_logits, pred_reg, gt_cls, gt_reg):
    cls_loss = rpn_cls_loss(pred_logits, gt_cls)
    reg_loss = rpn_reg_loss(pred_reg, gt_reg, gt_cls)
    
    return cls_loss+reg_loss