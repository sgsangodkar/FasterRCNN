#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 09:14:17 2021

@author: sagar
"""
import os
import torch
import torch.nn as nn
from model import FasterRCNN
from utils import gen_anchors, target_gen_rpn, gen_rois, target_gen_fast_rcnn
from loss import get_rpn_loss, get_fast_rcnn_loss
from torchnet.meter import AverageValueMeter
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class FasterRCNNTrainer(nn.Module):
    def __init__(self, num_classes, lr=0.001, log=True):
        super().__init__()
            
        self.model = FasterRCNN(num_classes).to(device)
        self.lr = lr
        self.log = log

        for layer in self.model.fe[:10]:
            for p in layer.parameters():
                p.requires_grad = False
         
        self.it = 0
        
        self.meters = {'rpn_cls': AverageValueMeter(),
                       'rpn_reg': AverageValueMeter(),
                       'fast_rcnn_cls' : AverageValueMeter(),
                       'fast_rcnn_reg' : AverageValueMeter()
                      }
        self.optimizer = self.get_optimizer()
        
        self.scheduler = lr_scheduler.StepLR(self.optimizer, 
                                             step_size=2, 
                                             gamma=0.1
                                         )
        self.writer = SummaryWriter()
          
        
    def get_optimizer(self):
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'fast_rcnn' in key:
                    if 'bias' in key:
                        params += [{'params': [value], 'lr': self.lr*2}]
                    else:
                        params += [{'params': [value], 'lr': self.lr}]

                else:    
                    params += [{'params': [value], 'lr': self.lr}]
        
        self.optimizer = optim.SGD(params, 
                                   momentum=0.9,
                                   weight_decay=0.0005
                                )
        return self.optimizer
 
    
    def rpn_train_step(self, features_l, img_size_l, bboxes_gt_l):
        rois_l = []
        batch_size = len(features_l)
        
        for data in zip(features_l, img_size_l, bboxes_gt_l):
            features = data[0]
            img_size = data[1]
            bboxes_gt = data[2].to(device)
            
            anchors = gen_anchors(
                    img_size, 
                    receptive_field=16, 
                    scales=[8,16,32], 
                    ratios=[0.5,1,2]
                ).to(device) 
            
            cls_op, reg_op = self.model.rpn(features)
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
                                              cls_gt.to(device), 
                                              reg_op, 
                                              reg_gt.to(device)
                                          )                              
            
            rpn_loss = cls_loss + 10*reg_loss 
            rpn_loss = rpn_loss/batch_size # Loss normalisation
            
            rpn_loss.backward(retain_graph=True)
            # Graph retained so as to use for fast-rcnn backward pass

            self.meters['rpn_cls'].add(cls_loss.item())
            self.meters['rpn_reg'].add(reg_loss.item()*10)
            
            
        return rois_l  

    def fast_rcnn_train_step(self, features_l, rois_l, bboxes_gt_l, classes_gt_l):        
        
        features_b = []
        cls_gt_b = []
        reg_gt_b = []
        
        batch_size = len(features_l)
        
        for data in zip(features_l, rois_l, bboxes_gt_l, classes_gt_l):     
            features = data[0]
            rois = data[1]
            bboxes_gt = data[2].to(device)
            classes_gt = data[3].to(device)
            
            rois, cls_gt, reg_gt = target_gen_fast_rcnn(
                                        rois, 
                                        bboxes_gt, 
                                        classes_gt, 
                                        n_targets = int(128/batch_size)
                                    )
                                                      
            pool = self.model.roi_layer(features, [rois])
            pool = pool.view(pool.size(0), -1)
            features_b.append(pool)
            cls_gt_b.append(cls_gt)
            reg_gt_b.append(reg_gt)
                
        features_b = torch.vstack(features_b)
        cls_gt_b = torch.hstack(cls_gt_b)
        reg_gt_b = torch.vstack(reg_gt_b)
        cls_op, reg_op = self.model.fast_rcnn(features_b)
      
        cls_loss, reg_loss = get_fast_rcnn_loss(cls_op, cls_gt_b, reg_op, reg_gt_b)

        fast_rcnn_loss = cls_loss + 10*reg_loss 
        fast_rcnn_loss.backward()

        self.meters['fast_rcnn_cls'].add(cls_loss.item())
        self.meters['fast_rcnn_reg'].add(reg_loss.item()*10)
        
    def train_step(self, img_l, bboxes_gt_l, classes_gt_l):
        self.model.train()
        features_l = []
        img_size_l = []
        for img in img_l:
            img = img.unsqueeze(0).to(device)
            features_l.append(self.model.fe(img))
            _,_,H,W = img.shape
            img_size_l.append((H,W))

        self.optimizer.zero_grad()   
        rois_l = self.rpn_train_step(features_l, img_size_l, bboxes_gt_l)
        
        self.fast_rcnn_train_step(features_l, rois_l, bboxes_gt_l, classes_gt_l)
            
                    
        self.optimizer.step()
        self.scheduler.step()   

        if self.log:
             self.writer.add_scalar('RPN_cls', self.meters['rpn_cls'].mean, self.it)      
             self.writer.add_scalar('RPN_reg', self.meters['rpn_reg'].mean, self.it)      
             self.writer.add_scalar('FastRCNN_cls', self.meters['fast_rcnn_cls'].mean, self.it)      
             self.writer.add_scalar('FastRCNN_reg', self.meters['fast_rcnn_reg'].mean, self.it)      

        self.it += 1
        
    def save_model(self, path=None):
        model_params = self.model.state_dict()
      
        if path == None:
            path = 'checkpoints'
        
        if not os.path.exists(path):
            os.makedirs(path)
                       
        model_filename = os.path.join(path,'model_params.pt')
            
        torch.save(model_params, model_filename)
        
        state_params = dict()
        state_params['it'] = self.it
        state_params['meters'] = self.meters  
        state_params['optimizer'] = self.optimizer.state_dict()
        state_params['scheduler'] = self.scheduler.state_dict()         

        state_filename = os.path.join(path, 'state_params.pt')
        
        torch.save(state_params, state_filename)

    def load_model(self, path=None):
        model_filename = os.path.join(path,'model_params.pt')
        model_state_dict = torch.load(model_filename)
        self.model.load_state_dict(model_state_dict)
        
        state_filename = os.path.join(path,'state_params.pt')
        state_params = torch.load(state_filename)
        self.it = state_params['it']  
        self.meters = state_params['meters'] 
        self.optimizer.load_state_dict(state_params['optimizer'])   
        self.scheduler.load_state_dict(state_params['scheduler'])   
        print('Model and trainer-state loaded')
 
        
