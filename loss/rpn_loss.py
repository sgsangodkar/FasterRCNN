#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 17:31:22 2021

@author: sagar
"""
import torch.nn as nn
import torch
import torch.nn.functional as F

class RPNLoss(nn.Module):
    def __init__(self, num_samples, pos_ratio):
        super().__init__()
        self.num_samples = num_samples
        self.pos_ratio = pos_ratio

    def rpn_cls_loss(self, pred_logits, gt_labels, ids):
        loss = F.cross_entropy(pred_logits[ids], gt_labels[ids], reduction='sum')
        return tuple((loss, len(ids)))
    
    def rpn_reg_loss(self, pred_reg, gt_reg, gt_labels, mask):
        if len(mask)>0:
            loss = F.smooth_l1_loss(pred_reg[mask], gt_reg[mask], reduction='sum')
            return tuple((loss, len(mask)))
        else:
            return tuple((torch.tensor(0), 0))

    def forward(self, pred_reg, gt_reg, pred_logits, gt_labels):
        positives = torch.where(gt_labels==1)[0]
        negatives = torch.where(gt_labels==0)[0]
        ids = torch.empty(self.num_samples, dtype=torch.long)
        num_positives = int(self.num_samples*self.pos_ratio)
        
        if len(positives)<num_positives:
            ids[:len(positives)] = positives
            ids[len(positives):self.num_samples] = negatives[torch.randint(len(negatives), (self.num_samples-len(positives),))] 
            mask = ids[:len(positives)]
        else:
            ids[:num_positives] = positives[torch.randint(len(positives), (num_positives,))] 
            ids[num_positives:self.num_samples] = negatives[torch.randint(len(negatives), (self.num_samples-num_positives,))] 
            mask = ids[:num_positives]
    
        cls_loss = self.rpn_cls_loss(pred_logits, gt_labels, ids)
        reg_loss = self.rpn_reg_loss(pred_reg, gt_reg, gt_labels, mask)
      
        return cls_loss, reg_loss
