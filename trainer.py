#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 09:14:17 2021

@author: sagar
"""
import torch
import torch.nn as nn
from model import FasterRCNN
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
            
        self.model = FasterRCNN(config).to(self.device)

        for layer in self.model.fe[:10]:
            for p in layer.parameters():
                p.requires_grad = False
                
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
        cls_op, reg_op = self.model.fast_rcnn(features_b)
      
        cls_loss, reg_loss = get_fast_rcnn_loss(cls_op, cls_gt_b, reg_op, reg_gt_b)

        fast_rcnn_loss = cls_loss + 10*reg_loss 
        fast_rcnn_loss.backward()

        self.meters['fast_rcnn_cls'].add(cls_loss.item())
        self.meters['fast_rcnn_reg'].add(reg_loss.item()*10)
        
    def train_step(self, it, img_l, bboxes_gt_l, classes_gt_l):
        self.model.train()
        self.it = it
        features_l = []
        img_size_l = []
        for img in img_l:
            img = img.unsqueeze(0).to(self.device)
            features_l.append(self.model.fe(img))
            _,_,H,W = img.shape
            img_size_l.append((H,W))

        self.optimizer.zero_grad()   
        rois_l = self.rpn_train_step(features_l, img_size_l, bboxes_gt_l)
        
        self.fast_rcnn_train_step(features_l, rois_l, bboxes_gt_l, classes_gt_l)
            
                    
        self.optimizer.step()
        self.scheduler.step()   
        
    def save_model(self, prefix=None, save_train_state=False):
        model_params = self.model.state_dict()
      
        if prefix == None:
            model_filename = 'checkpoint_'+str(self.it)+'_model.pt'
        else:
            model_filename = prefix+'_model.pt'
            
        torch.save(model_params, model_filename)
        
        if save_train_state:
            state_params = dict()
            state_params['optimizer'] = self.optimizer.state_dict()
            state_params['scheduler'] = self.scheduler.state_dict()
            state_params['it'] = self.it
            state_params['loss_meters'] = self.meters                  

            if prefix == None:
                state_filename = 'checkpoint_'+str(self.it)+'_state.pt'
            else:
                state_filename = prefix+'_state.pt'

            torch.save(state_params, state_filename)

    def load_model(self, prefix=None, load_train_state=False):
        model_filename = prefix+'_model.pt'
        model_state_dict = torch.load(model_filename)
        self.model.load_state_dict(model_state_dict)
        
        if load_train_state == True:
            state_filename = prefix+'_state.pt'
            state_params = torch.load(state_filename)
            self.optimizer.load_state_dict(state_params['optimizer'])   
            self.scheduler.load_state_dict(state_params['scheduler'])   
            self.it = state_params['it']  
            self.loss_meters = state_params['loss_meters']  
        
    def val_step(self, img_l, bboxes_gt_l, classes_gt_l):
        self.model.eval()
        """
        TODO:
            if accuracy > best_acc:
                call savemodel function
        """
        output_size = (config.roi_pool_size, config.roi_pool_size)
        spatial_scale = 1/config.receptive_field
        roi_layer = RoIPool(output_size, spatial_scale)
        
        for img, bboxes_gt in zip(img_l, bboxes_gt_l):
            img = img.unsqueeze(0).to(self.device)
            features = self.model.fe(img)
            _,_,H,W = img.shape
            img_size = (H,W)

            anchors = gen_anchors(
                    img_size, 
                    receptive_field=16, 
                    scales=[8,16,32], 
                    ratios=[0.5,1,2]
                )           
            cls_op, reg_op = self.model.rpn(features)
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
            
            cls_op, reg_op = self.model.fast_rcnn(pool_feats)
            
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
             
