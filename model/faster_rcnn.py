#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 09:14:17 2021

@author: sagar
"""

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from model.rpn import RegionProposalNetwork, TargetGeneratorRPN
from loss.rpn_loss import RPNLoss
from configs import rpn_configs

class FasterRCNN(nn.Module):
    def __init__(self, phase, device, writer=None):
        super().__init__()
        model = models.vgg16(pretrained=True, progress=False)
        self.fe = model.features[:-1]   
        
        self.rpn = RegionProposalNetwork(rpn_configs.in_channels, 
                                    rpn_configs.out_channels, 
                                    rpn_configs.anchors_per_location)
        
        if phase == 'Train':
            self.fe.train().to(device)
            self.rpn.train().to(device)
        else:
            self.fe.eval().to(device)
            self.rpn.eval().to(device)          
            
        
        self.target_generator_rpn = TargetGeneratorRPN(rpn_configs.receptive_field,
                                                       rpn_configs.scales,
                                                       rpn_configs.ratios,
                                                       rpn_configs.pos_iou_thresh, 
                                                       rpn_configs.neg_iou_thresh
                                     )
        self.rpn_loss = RPNLoss(rpn_configs.samples_per_min_batch,
                                rpn_configs.pos_ratio)
        self.init_running_losses()
        self.step = 0        
        self.log_step = 10
        
        self.display = True
        self.writer = writer
        
        
    def forward(self, img, bboxes, classes):
        self.step+=1
        features = self.fe(img)
        rpn_cls_scores, rpn_op_reg = self.rpn(features)

        rpn_cls_scores = rpn_cls_scores.permute(0,2,3,1).contiguous().view(1,-1,2).squeeze()
        rpn_op_reg = rpn_op_reg.permute(0,2,3,1).contiguous().view(1,-1,4).squeeze()      
        
        rpn_gt_labels, rpn_gt_reg = self.target_generator_rpn(img.shape[2:4], 
                                                              bboxes
                                                          )
        rpn_cls_loss_t, rpn_reg_loss_t = self.rpn_loss(rpn_op_reg, 
                                               rpn_gt_reg, 
                                               rpn_cls_scores, 
                                               rpn_gt_labels
                                           )


        if self.step % self.log_step == 0:
            self.loss_logger(rpn_cls_loss_t, rpn_reg_loss_t)
            
        loss = rpn_cls_loss_t[0]/rpn_cls_loss_t[1] + rpn_reg_loss_t[0]/rpn_reg_loss_t[1]
            
        return loss
                
        
    def init_running_losses(self):
        self.running_rpn_cls_loss = 0
        self.running_rpn_cls_cnt = 0
        self.running_rpn_reg_loss = 0
        self.running_rpn_reg_cnt = 0
        
    def loss_logger(self, rpn_cls_loss_t, rpn_reg_loss_t):
        # running statistics                
        self.running_rpn_cls_loss += rpn_cls_loss_t[0].item()   
        self.running_rpn_cls_cnt += rpn_cls_loss_t[1]   
        rpn_cls = self.running_rpn_cls_loss/self.running_rpn_cls_cnt
        if self.writer!=None:
            self.writer.add_scalar('RPN_cls_loss', rpn_cls, self.step)
        
        self.running_rpn_reg_loss += rpn_reg_loss_t[0].item()   
        self.running_rpn_reg_cnt += rpn_reg_loss_t[1]     
        rpn_reg = self.running_rpn_reg_loss/self.running_rpn_reg_cnt
        if self.writer!=None:
            self.writer.add_scalar('RPN_reg_loss', rpn_reg, self.step)
        
        if self.display:
            print('Step: {}. RPN_cls: {:.4f}, RPN_reg: {:.4f}'
                  .format(self.step, rpn_cls, rpn_reg))
        
        
        