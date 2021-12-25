#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 17:31:22 2021

@author: sagar
"""
import torch
import torch.nn.functional as F

def get_rpn_cls_loss(cls_op, cls_gt):
    #print(torch.argmax(cls_op[cls_mask], axis=1), cls_gt[cls_mask])
    loss = F.cross_entropy(cls_op, cls_gt, ignore_index=-1)
    return loss
    
def get_rpn_reg_loss(reg_op, reg_gt, mask):
    loss = F.smooth_l1_loss(reg_op[mask], reg_gt[mask])
    #print(reg_op[reg_mask], reg_gt[reg_mask])
    #print("RPN")
    #print(torch.abs((reg_op[reg_mask]-reg_gt[reg_mask])).max())
    #print(loss, len(reg_mask))
    #print(reg_op[reg_mask])
    #print(reg_gt[reg_mask])
    #print(reg_op[reg_mask].mean(axis=0), reg_gt[reg_mask].mean(axis=0))
    return loss
 
def get_rpn_loss(cls_op, cls_gt, reg_op, reg_gt):
    mask = torch.where(cls_gt==1)[0]   
    cls_loss = get_rpn_cls_loss(cls_op, cls_gt)
    reg_loss = get_rpn_reg_loss(reg_op, reg_gt, mask)
  
    return cls_loss, reg_loss
"""
def rpn_cls_loss(cls_op, cls_gt, cls_mask):
    loss = F.cross_entropy(cls_op[cls_mask], cls_gt[cls_mask])
    return loss
    
def rpn_reg_loss(reg_op, reg_gt, reg_mask):
    loss = F.smooth_l1_loss(reg_op[reg_mask], reg_gt[reg_mask])
    return loss
 
def rpn_loss(cls_op, cls_gt, reg_op, reg_gt):
    num_samples = 256
    pos_ratio = 0.5

    pos = torch.where(cls_gt==1)[0]
    neg = torch.where(cls_gt==0)[0]
    cls_mask = torch.empty(num_samples, dtype=torch.long)
    req_num_pos = int(num_samples*pos_ratio)
    
    if len(pos)<req_num_pos:
        cls_mask[:len(pos)] = pos
        req_num_neg = num_samples-len(pos)
        cls_mask[len(pos):num_samples] = neg[torch.randint(len(neg), (req_num_neg,))] 
        reg_mask = cls_mask[:len(pos)]
    else:
        cls_mask[:req_num_pos] = pos[torch.randint(len(pos), (req_num_pos,))] 
        req_num_neg = num_samples-req_num_pos
        cls_mask[req_num_pos:num_samples] = neg[torch.randint(len(neg), (req_num_neg,))] 
        reg_mask = cls_mask[:req_num_pos]

    cls_loss = rpn_cls_loss(cls_op, cls_gt, cls_mask)
    reg_loss = rpn_reg_loss(reg_op, reg_gt, reg_mask)
  
    return cls_loss, reg_loss


class RPNLoss:
    def __init__(self):
        self.r_cls_loss = 0
        self.r_cls_count = 0
        
        self.r_reg_loss = 0
        self.r_reg_count = 0
        
        self.num_samples = 256
        self.pos_ratio = 0.5        

    def rpn_cls_loss(self, cls_op, cls_gt, cls_mask):
        loss = F.cross_entropy(cls_op[cls_mask], cls_gt[cls_mask])
        return tuple((loss, len(cls_mask)))
        
    def rpn_reg_loss(self, reg_op, reg_gt, reg_mask):
        loss = F.smooth_l1_loss(reg_op[reg_mask], reg_gt[reg_mask], reduction='sum')
        return tuple((loss, len(reg_mask)))
    
    def calculate(self, cls_op, cls_gt, reg_op, reg_gt):   
        pos = torch.where(cls_gt==1)[0]
        neg = torch.where(cls_gt==0)[0]
        cls_mask = torch.empty(self.num_samples, dtype=torch.long)
        req_num_pos = int(self.num_samples*self.pos_ratio)
        
        if len(pos)<req_num_pos:
            cls_mask[:len(pos)] = pos
            req_num_neg = self.num_samples-len(pos)
            cls_mask[len(pos):self.num_samples] = neg[torch.randint(len(neg), (req_num_neg,))] 
            reg_mask = cls_mask[:len(pos)]
        else:
            cls_mask[:req_num_pos] = pos[torch.randint(len(pos), (req_num_pos,))] 
            req_num_neg = self.num_samples-req_num_pos
            cls_mask[req_num_pos:self.num_samples] = neg[torch.randint(len(neg), (req_num_neg,))] 
            reg_mask = cls_mask[:req_num_pos]
    
        cls_loss, cls_count = rpn_cls_loss(cls_op, cls_gt, cls_mask)
        reg_loss, reg_count = rpn_reg_loss(reg_op, reg_gt, reg_mask)

        self.r_cls_loss += cls_loss
        self.r_cls_count += cls_count
        
        self.r_reg_loss += reg_loss
        self.r_reg_count += reg_count
        
        loss = cls_loss/cls_count + reg_loss/reg_count
        return loss 
    
    def get_running_losses(self):
        return self.r_cls_loss/self.r_cls_count, self.r_reg_loss/self.r_reg_count
"""
