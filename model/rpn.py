#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 12:10:32 2021

@author: sagar
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import generate_anchors, bbox2reg, unmap, obtain_iou_matrix

class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, anchors_per_location=9):
        super().__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Conv2d(out_channels, 2*anchors_per_location, kernel_size=1)
        self.regressor = nn.Conv2d(out_channels, 4*anchors_per_location, kernel_size=1)
        self.init_layers()
        
    def init_layers(self):
        self.conv3x3.weight.data.normal_(0, 0.01)
        self.conv3x3.bias.data.zero_()
        self.classifier.weight.data.normal_(0, 0.01)
        self.classifier.bias.data.zero_()
        self.regressor.weight.data.normal_(0, 0.01)
        self.regressor.bias.data.zero_()
        
    def forward(self, x):
        x = F.relu(self.conv3x3(x))
        cls_scores = self.classifier(x)
        reg_op = self.regressor(x)
        
        return cls_scores, reg_op
    
class TargetGeneratorRPN(object):
    def __init__(self, 
                 receptive_field=16,
                 scales = [8,16,32],
                 ratios = [0.5,1,2],
                 pos_iou_thresh=0.7, 
                 neg_iou_thresh=0.3
        ):
        self.receptive_field = receptive_field
        self.scales = scales
        self.ratios = ratios
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        
    def __call__(self, img_size, bboxes):
        anchors = generate_anchors(img_size, 
                                   self.receptive_field, 
                                   self.scales, 
                                   self.ratios)
        num_anchors = anchors.shape[0]
       
        
        indx_inside = np.where(
                        (anchors[:, 0] >= 0) &
                        (anchors[:, 1] >= 0) &
                        (anchors[:, 2] < img_size[1]) &  # width
                        (anchors[:, 3] < img_size[0])    # height
                        )[0]
        
        anchors, labels, bboxes = self.get_labels_bboxes(anchors, bboxes, indx_inside)
        
        reg_targets = bbox2reg(bboxes, anchors)
        
        labels = unmap(labels, num_anchors, indx_inside)
        reg_targets = unmap(reg_targets, num_anchors, indx_inside)
            
        return torch.tensor(labels, dtype=torch.long), \
            torch.tensor(reg_targets, dtype=torch.float32)

    def get_labels_bboxes(self, anchors, gt_bboxes, indx_inside):  
        num_bboxes = gt_bboxes.shape[0]
        iou_matrix = obtain_iou_matrix(anchors[indx_inside], gt_bboxes)
        labels = np.empty(len(indx_inside), dtype=np.int32)
        labels.fill(-1)
    
        max_iou_per_anchor = np.max(iou_matrix, axis=1)
        min_iou_per_anchor = np.min(iou_matrix, axis=1)
        labels[max_iou_per_anchor>0.7] = 1
        labels[min_iou_per_anchor<0.3] = 0
        
        max_iou = np.max(iou_matrix)
        for i in range(num_bboxes):
            labels[iou_matrix[:,i]==max_iou] = 1
            
        return anchors[indx_inside], labels, gt_bboxes[np.argmax(iou_matrix, axis=1)]

