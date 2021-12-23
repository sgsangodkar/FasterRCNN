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
from loss import faster_rcnn_loss, get_rpn_loss, get_fast_rcnn_loss
from torchnet.meter import AverageValueMeter
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision.ops import RoIPool


class FasterRCNNTrainer(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        vgg = models.vgg16(pretrained=True, progress=False)
        self.fe = vgg.features[:-1].to(self.device)   
        for layer in self.fe[:10]:
            for p in layer.parameters():
                p.requires_grad = False
            
        self.rpn = RPN(config.in_chan,
                       config.out_chan,
                       config.anchors_per_location
                   ).to(self.device)
        
        self.fast_rcnn = FastRCNN(vgg.classifier,
                                  config.num_classes
                              ).to(self.device)
        
        
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
                
                if 'fast_rcnn' in key:
                    if 'bias' in key:
                        params += [{'params': [value], 'lr': lr*2}]
                    else:
                        params += [{'params': [value], 'lr': lr}]
               
                        
                #elif 'rpn.regressor' in key:
                #    if 'bias' in key:
                #        params += [{'params': [value], 'lr': lr*2}]
                #    else:
                #        params += [{'params': [value], 'lr': lr}]  
                        
                else:    
                    params += [{'params': [value], 'lr': lr}]
        
        self.optimizer = optim.SGD(params, 
                                   momentum=0.9,
                                   weight_decay=0.0005
                                )
        #self.optimizer = optim.Adam(params)
        return self.optimizer
 
    
    #def forward(self, img, bboxes_gt, classes_gt):
        #print("inside forward")
        #features = self.fe(img)

        #_,_,H,W = img.shape
        #img_size = (H,W)
        
        #anchors = gen_anchors(
        #        img_size, 
        #        receptive_field=16, 
        #        scales=[4,8,16], 
        #        ratios=[0.5,1,2]
        #    )  
        #print(anchors.shape, anchors.dtype)
        #print(anchors[0], bboxes_gt, img_size)
        #rpn_cls_gt, rpn_reg_gt = target_gen_rpn(anchors, bboxes_gt, img_size)
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
        
        #rpn_cls_op, rpn_reg_op = self.rpn(features)
        #print(features.dtype, rpn_cls_op.dtype, rpn_reg_op.dtype, "Here")
        #print("RPN Success")
        #rpn_cls_op = rpn_cls_op.permute(0,2,3,1).contiguous().view(1,-1,2).squeeze()
        #rpn_reg_op = rpn_reg_op.permute(0,2,3,1).contiguous().view(1,-1,4).squeeze()  
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
            
        #rpn_gt = (rpn_cls_gt, rpn_reg_gt)
        #rpn_op = (rpn_cls_op, rpn_reg_op)
        
        #print(rpn_cls_op.shape, rpn_cls_gt.shape, rpn_reg_op.shape, rpn_reg_gt.shape)

        #print(rpn_cls_loss, rpn_reg_loss)
        #print(rpn_cls_gt.shape, rpn_reg_gt.shape)
        """
        rois = gen_rois(rpn_cls_op.detach(), 
                        rpn_reg_op.detach(), 
                        anchors, 
                        img_size
                    )
        """

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
        """    
        rois, fast_rcnn_cls_gt, fast_rcnn_reg_gt = target_gen_fast_rcnn(rois, 
                                                                        bboxes_gt, 
                                                                        classes_gt
                                                                    )
        """
        #print(fast_rcnn_reg_gt.mean(axis=0))
        #print(rois.shape, "in trainer")

        #if False:        trainer.train_step(img, bboxes, classes)

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
        #if len(rois):
        #fast_rcnn_cls_op, fast_rcnn_reg_op = self.fast_rcnn(features, rois)
        #else:
        #    print("here")
        #    fast_rcnn_cls_op = torch.zeros((1, config.num_classes+1))
        #    fast_rcnn_reg_op = torch.zeros((1, config.num_classes*4))
        #    fast_rcnn_cls_gt  = torch.zeros(1, dtype=torch.long)
        #    fast_rcnn_reg_gt = torch.zeros((1, 4))
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
        #bboxes = bboxes[torch.arange(len(bboxes)), classes]
        #bboxes = reg2bbox(rois, bboxes)

        #print(bboxes.shape)
        #img_np = visualize_bboxes(img, bboxes.detach())
        #plt.imshow(img_np)
        #plt.show()
        #plt.imshow(img_np) 
        #plt.title("Output Boxes")
        #plt.show()
       
        #print(fast_rcnn_cls_op.shape, fast_rcnn_reg_op.shape)
        #print("Fast RCNN Success")
        
        #fast_rcnn_gt = (fast_rcnn_cls_gt, fast_rcnn_reg_gt)
        #fast_rcnn_op = (fast_rcnn_cls_op, fast_rcnn_reg_op)
        #print(roi_cls_op.shape, roi_reg_op.shape)
        # 128x21, 128X(20*4)
        
        #return rpn_gt, rpn_op, fast_rcnn_gt, fast_rcnn_op
    
    def rpn_train_step(self, features_l, img_size_l, bboxes_gt_l):
        #print("inside rpn_train_step")
        rois_l = []
        self.rpn.train()
        for data in zip(features_l, img_size_l, bboxes_gt_l):
            features = data[0]
            img_size = data[1]
            bboxes_gt = data[2].to(self.device)
            
            anchors = gen_anchors(
                    img_size, 
                    receptive_field=16, 
                    scales=[4,8,16], 
                    ratios=[0.5,1,2]
                )   
            cls_gt, reg_gt = target_gen_rpn(anchors, bboxes_gt, img_size)
        
            cls_op, reg_op = self.rpn(features)
            #print(cls_op.shape, reg_op.shape)
            cls_op = cls_op.permute(0,2,3,1).contiguous().view(1,-1,2).squeeze()
            """
            Check the permutation using sample example
            """
            reg_op = reg_op.permute(0,2,3,1).contiguous().view(1,-1,4).squeeze()
            cls_loss, reg_loss = get_rpn_loss(cls_op, 
                                              cls_gt, 
                                              reg_op, 
                                              reg_gt
                                          )
            
            rpn_loss = cls_loss + 10*reg_loss 
            rpn_loss = rpn_loss/config.batch_size # Loss normalisation
            rpn_loss.backward(retain_graph=True)
            # Graph retained so as to use for fast-rcnn backward pass

            self.meters['rpn_cls'].add(cls_loss.item())
            self.meters['rpn_reg'].add(reg_loss.item()*10)
            
            rois = gen_rois(cls_op.detach(), 
                            reg_op.detach(), 
                            anchors, 
                            img_size
                        )
            rois_l.append(rois) # x1, y1, x2, y2
            
        return rois_l  

    def fast_rcnn_train_step(self, img_l, features_l, rois_l, bboxes_gt_l, classes_gt_l):
        #print("inside fast_rcnn_train_step")
        self.fast_rcnn.train()
        
        output_size = (config.roi_pool_size, config.roi_pool_size)
        spatial_scale = 1/config.receptive_field
        roi_layer = RoIPool(output_size, spatial_scale)
        
        features_b = []
        cls_gt_b = []
        reg_gt_b = []
        #rois_b = []
        #print(features_l)
        for data in zip(features_l, rois_l, bboxes_gt_l, classes_gt_l, img_l):
            features = data[0]
            rois = data[1]
            bboxes_gt = data[2].to(self.device)
            classes_gt = data[3].to(self.device)
            #img = data[4].to(self.device)
            
            #print(rois.shape)
            #im1 = visualize_bboxes(img, rois)
            #print(im1.shape)
            #plt.imshow(im1)
            #plt.show()
            #print(classes_gt)
            rois, cls_gt, reg_gt = target_gen_fast_rcnn(rois, 
                                                        bboxes_gt, 
                                                        classes_gt
                                                    )    
            #im1 = visualize_bboxes(img, rois)
            #print(im1.shape)
            #plt.imshow(im1)
            #plt.show()

            #im1 = visualize_bboxes(img, bboxes_gt)
            #print(im1.shape)
            #plt.imshow(im1)
            #plt.show()
            
            if len(rois)>0:
                #print("Here")
                pool = roi_layer(features, [rois])
                #print(len(rois))
                #print(pool.shape)
                pool = pool.view(pool.size(0), -1)
                features_b.append(pool)
                cls_gt_b.append(cls_gt)
                #print(cls_gt)
                reg_gt_b.append(reg_gt)
            #rois_b.append(rois)
        
        features_b = torch.vstack(features_b)
        cls_gt_b = torch.hstack(cls_gt_b)
        reg_gt_b = torch.vstack(reg_gt_b)
        #rois_b = torch.vstack(rois_b)
        #print(features_b.shape)
        cls_op, reg_op = self.fast_rcnn(features_b)
        #print(torch.argmax(cls_op, dim=1))
        #print(cls_gt_b)
        #print(cls_op.shape, cls_gt_b.shape, reg_op.shape, reg_gt_b.shape)
        cls_loss, reg_loss = get_fast_rcnn_loss(cls_op, cls_gt_b, reg_op, reg_gt_b)

        fast_rcnn_loss = cls_loss + 10*reg_loss 
        fast_rcnn_loss.backward()
        # Graph retained so as to use for fast-rcnn backward pass

        self.meters['fast_rcnn_cls'].add(cls_loss.item())
        self.meters['fast_rcnn_reg'].add(reg_loss.item()*10)
        
        if True:
            classes = torch.argmax(cls_op, axis=1)
        #    print(cls_op.shape, classes.shape)
            reg_op = reg_op.view(len(reg_op), -1, 4)
            reg_op = reg_op[torch.arange(len(reg_op)), classes]
            bboxes = reg2bbox(reg_gt_b, reg_op)
    
            print(bboxes[0:10])
            img_np = visualize_bboxes(img_l[0].unsqueeze(0), bboxes[0:10].detach())
            plt.imshow(img_np)
         
        
    def train_step(self, img_l, bboxes_gt_l, classes_gt_l):
        #print("inside train step")
        self.fe.train()
        features_l = []
        img_size_l = []
        for img in img_l:
            img = img.unsqueeze(0).to(self.device)
            features_l.append(self.fe(img))
            _,_,H,W = img.shape
            img_size_l.append((H,W))

        self.optimizer.zero_grad()                
        rois_l = self.rpn_train_step(features_l, img_size_l, bboxes_gt_l)
        self.fast_rcnn_train_step(img_l, features_l, rois_l, bboxes_gt_l, classes_gt_l)
            
        #print("Computing Loss")
        #total_loss, loss_dict = faster_rcnn_loss(rpn_gt, 
        #                                         rpn_op, 
        #                                         fast_rcnn_gt, 
        #                                         fast_rcnn_op
        #                                     )
        #print("Loss Computed")
        #total_loss.backward()
        #if step>100:
        #    fast_rcnn_l.backward()
        
        
        #for key, value in loss_dict.items():
        #    self.meters[key].add(value.item())
                    
        self.optimizer.step()
        self.scheduler.step()   
 
    def val_step(self, img_l, bboxes_gt_l, classes_gt_l):
        self.fe.eval()
        self.rpn.eval()
        self.fast_rcnn.eval()
        output_size = (config.roi_pool_size, config.roi_pool_size)
        spatial_scale = 1/config.receptive_field
        roi_layer = RoIPool(output_size, spatial_scale)
        
        for img, bboxes_gt in zip(img_l, bboxes_gt_l):
            img = img.unsqueeze(0).to(self.device)
            features = self.fe(img)
            _,_,H,W = img.shape
            img_size = (H,W)

            anchors = gen_anchors(
                    img_size, 
                    receptive_field=16, 
                    scales=[4,8,16], 
                    ratios=[0.5,1,2]
                )           
            cls_op, reg_op = self.rpn(features)
            cls_op = cls_op.permute(0,2,3,1).contiguous().view(1,-1,2).squeeze()
            reg_op = reg_op.permute(0,2,3,1).contiguous().view(1,-1,4).squeeze()
            
            rois = gen_rois(cls_op.detach(), 
                            reg_op.detach(), 
                            anchors, 
                            img_size
                        ) # x1, y1, x2, y2
            
            im1 = visualize_bboxes(img, bboxes_gt)
            plt.imshow(im1)
            plt.show()
            
            if len(rois)>0:
                pool = roi_layer(features, [rois])
                pool_feats = pool.view(pool.size(0), -1)
            
            cls_op, reg_op = self.fast_rcnn(pool_feats)

            
            if True:
                classes = torch.argmax(cls_op, axis=1)
                reg_op = reg_op.view(len(reg_op), -1, 4)
                reg_op = reg_op[torch.arange(len(reg_op)), classes-1]
                bboxes = reg2bbox(rois, reg_op)
                #indices = nms(bboxes, fg_scores, nms_thresh)
                img_np = visualize_bboxes(img, bboxes[classes>0].detach())
                plt.imshow(img_np)
                plt.show()
             
 
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
        