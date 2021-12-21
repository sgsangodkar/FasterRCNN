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
from utils import reg2bbox, visualize_bboxes
from configs import config
from loss import faster_rcnn_loss
from torchnet.meter import AverageValueMeter
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import cv2

class FasterRCNNTrainer(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg16(pretrained=True, progress=False)
        self.fe = vgg.features[:-1].to(device)   
        for layer in self.fe[:10]:
            for p in layer.parameters():
                p.requires_grad = False
            
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
        self.optimizer = self.get_optimizer()
        #print(self.optimizer)
        
        self.scheduler = lr_scheduler.StepLR(self.optimizer, 
                                             step_size=30000, 
                                             gamma=0.1
                                         )

        self.step = 0
    """
    TODO:
        check if roi size <7x7:
            what is the pool output?
            Is there an error?
    """
    def get_optimizer(self):
        lr = config.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                #if 'fe' in key:
                #    params += [{'params': [value], 'lr': lr}]
                
                #if 'regressor' in key or 'classifier' in key:
                    #if 'bias' in key:
                    #    params += [{'params': [value], 'lr': lr*2}]
                    #else:
                    #    params += [{'params': [value], 'lr': lr}]
               
                        
                #elif 'rpn.regressor' in key:
                #    if 'bias' in key:
                #        params += [{'params': [value], 'lr': lr*2}]
                #    else:
                #        params += [{'params': [value], 'lr': lr}]  
                        
                #else:    
                params += [{'params': [value], 'lr': lr}]
        
        self.optimizer = optim.SGD(params, 
                                   momentum=0.9,
                                   weight_decay=0.0005
                                )
        #self.optimizer = optim.Adam(params)
        return self.optimizer
 
    
    def forward(self, img, bboxes_gt, classes_gt):
        #print("inside forward")
        features = self.fe(img)

        _,_,H,W = img.shape
        img_size = (H,W)
        
        anchors = gen_anchors(
                img_size, 
                receptive_field=16, 
                scales=[4,8,16], 
                ratios=[0.5,1,2]
            )  
        #print(anchors.shape, anchors.dtype)
        #print(anchors[0], bboxes_gt, img_size)
        rpn_cls_gt, rpn_reg_gt = target_gen_rpn(anchors, bboxes_gt, img_size)
        #print(rpn_reg_gt.mean(axis=0))
        #print("Target RPN Success")
        #print(img.shape)
        #if False:
        #    rpn_bboxes = reg2bbox(anchors[rpn_cls_gt==1], 
        #                          rpn_reg_gt[rpn_cls_gt==1]
        #                      )
        #    img_np = visualize_bboxes(img, rpn_bboxes[torch.randperm(len(rpn_bboxes))[:25]])
        #    plt.imshow(img_np) 
        #    plt.title("RPN GT Boxes")
        #    plt.show()
        
        rpn_cls_op, rpn_reg_op = self.rpn(features)
        #print(features.dtype, rpn_cls_op.dtype, rpn_reg_op.dtype, "Here")
        #print("RPN Success")
        rpn_cls_op = rpn_cls_op.permute(0,2,3,1).contiguous().view(1,-1,2).squeeze()
        rpn_reg_op = rpn_reg_op.permute(0,2,3,1).contiguous().view(1,-1,4).squeeze()  
        #print(rpn_reg_gt.shape, rpn_reg_op.shape, rpn_reg_op.view(-1,4).shape)
        #print(rpn_reg_op.mean(axis=0))

        #if False:
        #    classes = torch.argmax(rpn_cls_op, axis=1)
        #    mask = classes>0
        #    if torch.sum(mask):
        #        bboxes2 = rpn_reg_op[mask]
        #        
        #        anchors2 = anchors[mask]
        #        bboxes2 = reg2bbox(anchors2, bboxes2).detach()
        #        #bboxes = bboxes.view(len(bboxes), -1, 4)
        #        #print(fast_rcnn_reg_op.shape, bboxes.shape, mask.shape)
        #        #bboxes = bboxes[torch.arange(len(bboxes)), classes[mask]-1]
        #        #print(bboxes.shape)
               
        #        img_np = visualize_bboxes(img, bboxes2[torch.randperm(len(bboxes2))[:25]])
        #        plt.imshow(img_np) 
        #        plt.title("RPN Output Boxes")
        #        plt.show()
            
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


        #if False:
        #    #print(torch.sum(fast_rcnn_cls_gt==0), torch.sum(fast_rcnn_cls_gt>0))
        #    img_np = visualize_bboxes(img, rois[torch.randperm(len(rois))[:25]])
        #    plt.imshow(img_np) 
        #    plt.title("RPN Output Boxes")
        #    plt.show()

        #if False:
            #print(torch.sum(fast_rcnn_cls_gt==0), torch.sum(fast_rcnn_cls_gt>0))
        #    img_np = visualize_bboxes(img, bboxes_gt)
        #    plt.imshow(img_np) 
        #    plt.title("FastRCNN GT Boxes")
        #    plt.show()
            
        rois, fast_rcnn_cls_gt, fast_rcnn_reg_gt = target_gen_fast_rcnn(rois, 
                                                                        bboxes_gt, 
                                                                        classes_gt
                                                                    )
        #print(fast_rcnn_reg_gt.mean(axis=0))
        #print(rois.shape, "in trainer")

        #if False:
            #print(torch.sum(fast_rcnn_cls_gt==0), torch.sum(fast_rcnn_cls_gt>0))
        #    img_np = visualize_bboxes(img, rois[torch.randperm(len(rois))[:10]])
        #    plt.imshow(img_np) 
        #    plt.title("FastRCNN GT +ve Boxes")
        #    plt.show()
            
        #if False:
            ## This is wrong
            #print(torch.sum(fast_rcnn_cls_gt==0), torch.sum(fast_rcnn_cls_gt>0))
        #    img_np = visualize_bboxes(img,anchors, rois[fast_rcnn_cls_gt>0])
        #    plt.imshow(img_np) 
        #    plt.title("FastRCNN GT +ve Boxes")
        #    plt.show()
            
        #print("Target FastRCNN Success")
        #print(rois.shape, fast_rcnn_cls_gt.shape, fast_rcnn_reg_gt.shape)  
        if len(rois):
            fast_rcnn_cls_op, fast_rcnn_reg_op = self.fast_rcnn(features, rois)
        else:
            print("here")
            fast_rcnn_cls_op = torch.zeros((1, config.num_classes+1))
            fast_rcnn_reg_op = torch.zeros((1, config.num_classes*4))
            fast_rcnn_cls_gt  = torch.zeros(1, dtype=torch.long)
            fast_rcnn_reg_gt = torch.zeros((1, 4))
        #print(fast_rcnn_reg_op[:,0:4].mean(axis=0))

        
            #print(torch.sum(fast_rcnn_cls_gt==0), torch.sum(fast_rcnn_cls_gt>0))
            #print(torch.argmax(fast_rcnn_cls_op, axis=1).shape)
        #classes = torch.argmax(fast_rcnn_cls_op, axis=1)
        #print(fast_rcnn_cls_op.shape, classes.shape)
        #mask = classes>0
        #print("Hi")
        #print(torch.sum(mask))
        #print(classes)
        #print(fast_rcnn_cls_gt)
        #print(fast_rcnn_reg_op, fast_rcnn_reg_gt)
        #if torch.sum(mask):
        #bboxes = fast_rcnn_reg_op
        #rois = rois
        #bboxes = bboxes.view(len(bboxes), -1, 4)
        #print(fast_rcnn_reg_op.shape, bboxes.shape, mask.shape)
        #print(len(bboxes))
        #print(bboxes)
        #bboxes = bboxes[torch.arange(len(bboxes)), 1-1]
        #bboxes = reg2bbox(rois, bboxes)

        #print(bboxes.shape)
        #img_np = visualize_bboxes(img, bboxes.detach())
        #plt.imshow(img_np) 
        #plt.title("Output Boxes")
        #plt.show()
       
        #print(fast_rcnn_cls_op.shape, fast_rcnn_reg_op.shape)
        #print("Fast RCNN Success")
        
        fast_rcnn_gt = (fast_rcnn_cls_gt, fast_rcnn_reg_gt)
        fast_rcnn_op = (fast_rcnn_cls_op, fast_rcnn_reg_op)
        #print(roi_cls_op.shape, roi_reg_op.shape)
        # 128x21, 128X(20*4)
        
        return rpn_gt, rpn_op, fast_rcnn_gt, fast_rcnn_op
    
        
    def train_step(self, img, bboxes_gt, classes_gt, step):
        #print("inside train step")
        self.fe.train()
        self.rpn.train()
        self.fast_rcnn.train()
        
        self.optimizer.zero_grad()    
        rpn_gt, rpn_op, fast_rcnn_gt, fast_rcnn_op = self.forward(img, 
                                                                  bboxes_gt, 
                                                                  classes_gt
                                                              )   
        #print("Computing Loss")
        total_loss, loss_dict = faster_rcnn_loss(rpn_gt, 
                                                 rpn_op, 
                                                 fast_rcnn_gt, 
                                                 fast_rcnn_op
                                             )
        #print("Loss Computed")
        total_loss.backward()
        #if step>100:
        #    fast_rcnn_l.backward()
        
        
        for key, value in loss_dict.items():
            self.meters[key].add(value.item())
                    
        self.optimizer.step()
        self.scheduler.step()   
 
    #def val_step(self, img, bboxes_gt, classes_gt):
    """
    TO DO
    detach, item usage, separate tensor?
    
    using train val and test sets
    why rpn reg loss is lower compared to fastrcnn reg loss
    
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
        