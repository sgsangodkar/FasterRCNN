#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 19:09:25 2021

@author: sagar
"""
import torch
import torch.nn.functional as F

def rpn_cls_loss(pred_logits, gt_cls, ids):
    loss = F.cross_entropy(pred_logits[ids], gt_cls[ids])
    return loss

def rpn_reg_loss(pred_reg, gt_reg, gt_cls, mask):
    if len(mask[0])>0:
        loss = F.smooth_l1_loss(pred_reg[mask], gt_reg[mask], size_average=False)/len(mask[0])
        return loss
    else:
        return torch.tensor(0)

def rpn_loss(pred_logits, pred_reg, gt_cls, gt_reg):
    positives = torch.where(gt_cls==1)[0]
    negatives = torch.where(gt_cls==0)[0]
    ids = torch.empty(256, dtype=torch.long)
    mask = []
    if len(positives)<128:
        ids[:len(positives)] = positives
        ids[len(positives):256] = negatives[torch.randint(len(negatives), (256-len(positives),))] 
        mask.append(ids[:len(positives)])
    else:
        ids[:128] = positives[torch.randint(len(positives), (128,))] 
        ids[128:256] = negatives[torch.randint(len(negatives), (128,))] 
        mask.append(ids[:128])

    cls_loss = rpn_cls_loss(pred_logits, gt_cls, ids)
    reg_loss = rpn_reg_loss(pred_reg, gt_reg, gt_cls, mask)
    #print(cls_loss.item(),10*reg_loss.item())
    #print(cls_loss.dtype, reg_loss.dtype)
    return cls_loss, reg_loss

