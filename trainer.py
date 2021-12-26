#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 09:14:17 2021

@author: sagar
"""
import torch
import torch.nn as nn
from torchvision import models
from model import RPN, FastRCNN
from utils import gen_anchors, target_gen_rpn, gen_rois, target_gen_fast_rcnn
from utils import reg2bbox, visualize_bboxes
from configs import config
from loss import get_rpn_loss, get_fast_rcnn_loss
from torchnet.meter import AverageValueMeter
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from torchvision.ops import RoIPool
from torchvision.ops import nms


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
        
        self.scheduler = lr_scheduler.StepLR(self.optimizer, 
                                             step_size=30000, 
                                             gamma=0.1
                                         )

        self.step = 0

    def get_optimizer(self):
        lr = config.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'fast_rcnn' in key:
                    if 'bias' in key:
                        params += [{'params': [value], 'lr': lr*2}]
                    else:
                        params += [{'params': [value], 'lr': lr}]

                else:    
                    params += [{'params': [value], 'lr': lr}]
        
        self.optimizer = optim.SGD(params, 
                                   momentum=0.9,
                                   weight_decay=0.0005
                                )
        return self.optimizer
 
    
    def rpn_train_step(self, features_l, img_size_l, bboxes_gt_l):
        rois_l = []
        self.rpn.train()
        for data in zip(features_l, img_size_l, bboxes_gt_l):
            features = data[0]
            img_size = data[1]
            bboxes_gt = data[2].to(self.device)
            
            anchors = gen_anchors(
                    img_size, 
                    receptive_field=16, 
                    scales=[8,16,32], 
                    ratios=[0.5,1,2]
                ).to(self.device) 
            
            cls_op, reg_op = self.rpn(features)
            cls_op = cls_op.permute(0,2,3,1).contiguous().view(1,-1,2).squeeze()

            reg_op = reg_op.permute(0,2,3,1).contiguous().view(1,-1,4).squeeze()
            
            rois = gen_rois(cls_op.detach(), 
                            reg_op.detach(), 
                            anchors, 
                            img_size
                        )
            rois_l.append(rois) # x1, y1, x2, y2

            
            cls_gt, reg_gt = target_gen_rpn(anchors, bboxes_gt, img_size)

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
            
            
        return rois_l  

    def fast_rcnn_train_step(self, features_l, rois_l, bboxes_gt_l, classes_gt_l):
        self.fast_rcnn.train()
        
        output_size = (config.roi_pool_size, config.roi_pool_size)
        spatial_scale = 1/config.receptive_field
        roi_layer = RoIPool(output_size, spatial_scale)
        
        features_b = []
        cls_gt_b = []
        reg_gt_b = []

        for data in zip(features_l, rois_l, bboxes_gt_l, classes_gt_l):
            features = data[0]
            rois = data[1]
            bboxes_gt = data[2].to(self.device)
            classes_gt = data[3].to(self.device)
            
            
            rois, cls_gt, reg_gt = target_gen_fast_rcnn(rois, 
                                                        bboxes_gt, 
                                                        classes_gt
                                                    )    
            pool = roi_layer(features, [rois])
            pool = pool.view(pool.size(0), -1)
            features_b.append(pool)
            cls_gt_b.append(cls_gt)
            reg_gt_b.append(reg_gt)
                
        features_b = torch.vstack(features_b)
        cls_gt_b = torch.hstack(cls_gt_b)
        reg_gt_b = torch.vstack(reg_gt_b)
        cls_op, reg_op = self.fast_rcnn(features_b)
      
        cls_loss, reg_loss = get_fast_rcnn_loss(cls_op, cls_gt_b, reg_op, reg_gt_b)

        fast_rcnn_loss = cls_loss + 10*reg_loss 
        fast_rcnn_loss.backward()

        self.meters['fast_rcnn_cls'].add(cls_loss.item())
        self.meters['fast_rcnn_reg'].add(reg_loss.item()*10)
        
    def train_step(self, img_l, bboxes_gt_l, classes_gt_l):
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
        
        self.fast_rcnn_train_step(features_l, rois_l, bboxes_gt_l, classes_gt_l)
            
                    
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
                    scales=[8,16,32], 
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
                reg_op = reg_op[classes>0]
                reg_op = reg_op[torch.arange(len(reg_op)), classes[classes>0]-1]
                rois = rois[classes>0]
                bboxes = reg2bbox(rois, reg_op)
                cls_op = cls_op[classes>0]
                fg_scores = cls_op[torch.arange(len(cls_op)),classes[classes>0]-1]
                indices = nms(bboxes, fg_scores, 0.7)
                bboxes = bboxes[indices[:10]]
                classes = classes[classes>0]
                print(classes[indices[:10]])
                img_np = visualize_bboxes(img, bboxes.detach())
                plt.imshow(img_np)
                plt.show()
             
