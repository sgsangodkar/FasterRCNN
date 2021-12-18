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
from model import RPN, FastRCNN
from utils import gen_anchors, target_gen_rpn, gen_rois, target_gen_fast_rcnn
from configs import config
from loss import faster_rcnn_loss
from torchnet.meter import AverageValueMeter
import torch.optim as optim
from torch.optim import lr_scheduler

class FasterRCNNTrainer(nn.Module):
    def __init__(self, device, writer=None):
        super().__init__()
        vgg = models.vgg16(pretrained=True, progress=False)
        self.fe = vgg.features[:-1].to(device)   
        
        self.rpn = RPN(config.in_chan,
                       config.out_chan,
                       config.anchors_per_location
                   ).to(device)
        
        self.fast_rcnn = FastRCNN(vgg.classifier,
                                  config.num_classes,
                                  config.roi_pool_size,
                                  config.receptive_field
                              ).to(device)
        
        
        self.meters = {'rpn_cls': AverageValueMeter(),
                       'rpn_reg': AverageValueMeter(),
                       'fast_rcnn_cls' : AverageValueMeter(),
                       'fast_rcnn_reg' : AverageValueMeter()
                      }
        self.writer = writer
        self.optimizer = self.get_optimizer()
        #Per layer learning rate??
        #VGG conv3_1 and above make learnable
        
        self.scheduler = lr_scheduler.StepLR(self.optimizer, 
                                             step_size=20000, 
                                             gamma=0.1
                                         )

        self.step = 0

    def get_optimizer(self):
       
        lr = config.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': 0.0005}]
        
        self.optimizer = optim.SGD(params, 
                                   momentum=0.9
                                )
        return self.optimizer
    # optimizer can only optimize Tensors, but one of the params 
    #is Module.parameters     
    
    def forward(self, img, bboxes_gt, classes_gt):
        #print("inside forward")
        features = self.fe(img)

        _,_,W,H = img.shape
        img_size = (W,H)
        
        anchors = gen_anchors(
                img_size, 
                receptive_field=16, 
                scales=[8,16,32], 
                ratios=[0.5,1,2]
            )  
        
        rpn_cls_gt, rpn_reg_gt = target_gen_rpn(anchors, bboxes_gt, img_size)
        #print("Target RPN Success")
        
        rpn_cls_op, rpn_reg_op = self.rpn(features)
        #print("RPN Success")
        rpn_cls_op = rpn_cls_op.permute(0,2,3,1).contiguous().view(1,-1,2).squeeze()
        rpn_reg_op = rpn_reg_op.permute(0,2,3,1).contiguous().view(1,-1,4).squeeze()      

        rpn_gt = (rpn_cls_gt, rpn_reg_gt)
        rpn_op = (rpn_cls_op, rpn_reg_op)
        
        #print(rpn_cls_op.shape, rpn_cls_gt.shape, rpn_reg_op.shape, rpn_reg_gt.shape)

        #print(rpn_cls_loss, rpn_reg_loss)
        #print(rpn_cls_gt.shape, rpn_reg_gt.shape)
        rois = gen_rois(rpn_cls_op.detach(), 
                        rpn_reg_op.detach(), 
                        anchors, 
                        img_size
                    )
        
        rois, fast_rcnn_cls_gt, fast_rcnn_reg_gt = target_gen_fast_rcnn(rois, 
                                                                        bboxes_gt, 
                                                                        classes_gt
                                                                    )
        #print("Target FastRCNN Success")
        #print(rois.shape, roi_cls_gt.shape, roi_reg_gt.shape)        
        fast_rcnn_cls_op, fast_rcnn_reg_op = self.fast_rcnn(features, rois)
        #print("Fast RCNN Success")
        
        fast_rcnn_gt = (fast_rcnn_cls_gt, fast_rcnn_reg_gt)
        fast_rcnn_op = (fast_rcnn_cls_op, fast_rcnn_reg_op)
        #print(roi_cls_op.shape, roi_reg_op.shape)
        # 128x21, 128X(20*4)
        
        return rpn_gt, rpn_op, fast_rcnn_gt, fast_rcnn_op
    
        
    def train_step(self, img, bboxes_gt, classes_gt):
        #print("inside train step")
        self.fe.train()
        self.rpn.train()
        self.fast_rcnn.train()
        
        self.step += 1
        self.optimizer.zero_grad()    
        rpn_gt, rpn_op, fast_rcnn_gt, fast_rcnn_op = self.forward(img, 
                                                                  bboxes_gt, 
                                                                  classes_gt
                                                              )   
        #print("Computing Loss")
        loss_total, loss_dict = faster_rcnn_loss(rpn_gt, 
                                                 rpn_op, 
                                                 fast_rcnn_gt, 
                                                 fast_rcnn_op
                                             )
        #print("Loss Computed")
        loss_total.backward()
        
        for key, value in loss_dict.items():
            self.meters[key].add(value.item())
                    
        self.optimizer.step()
        self.scheduler.step()   
        
        if self.writer is not None:      
             self.writer.add_scalar('RPN_cls', loss_dict['rpn_cls'], self.step)      
             self.writer.add_scalar('RPN_reg', loss_dict['rpn_reg'], self.step)      
             self.writer.add_scalar('FastRCNN_cls', loss_dict['fast_rcnn_cls'], self.step)      
             self.writer.add_scalar('FastRCNN_reg', loss_dict['fast_rcnn_reg'], self.step)      

    #def val_step(self, img, bboxes_gt, classes_gt):
    """
    TO DO
    Study difference between and and &
    detach, item usage, separate tensor?
    using detach for loss dictionary is good?
    
    weighting of loss components
    optimiser: conv3_1 and upper learnable
    target generation FastRCNN: normalisation
    using train val and test sets
    
    eval method
    
    """
                
    """    
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
            self.writer.add_scalar('RPN_cls_loss', rpn_cls, self.step/self.log_step)
        
        self.running_rpn_reg_loss += rpn_reg_loss_t[0].item()   
        self.running_rpn_reg_cnt += rpn_reg_loss_t[1]     
        rpn_reg = self.running_rpn_reg_loss/self.running_rpn_reg_cnt
        if self.writer!=None:
            self.writer.add_scalar('RPN_reg_loss', rpn_reg, self.step/self.log_step)
        
        if self.display:
            print('Step: {}. RPN_cls: {:.4f}, RPN_reg: {:.4f}'
                  .format(self.step, rpn_cls, rpn_reg))




        roi_cls_loss, roi_reg_loss = fast_rcnn_loss(
                                            roi_cls_op, 
                                            roi_cls_gt, 
                                            roi_reg_op, 
                                            roi_reg_gt
                                        )
        print(roi_cls_loss, roi_reg_loss)

        #if self.step % self.log_step == 0:
            #self.loss_logger(rpn_cls_loss_t, rpn_reg_loss_t)
            
        #loss = rpn_cls_loss_t[0]/rpn_cls_loss_t[1] + rpn_reg_loss_t[0]/rpn_reg_loss_t[1]
            
        #return loss

        rpn_cls_loss, rpn_reg_loss = rpn_loss(
                                            rpn_cls_op, 
                                            rpn_cls_gt, 
                                            rpn_reg_op, 
                                            rpn_reg_gt
                                        )        
    """    
        