#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 10:40:56 2021

@author: sagar
"""
import os
import cv2
from torchvision import datasets
from torch.utils.data import Dataset
from utils import generate_anchors, rm_cross_boundary_anchors, obtain_positive_anchors

class VOCDataset(Dataset):
    def __init__(self, transform, gt_parser, anchor_params):
        super().__init__()
        self.transform = transform
        self.gt_parser = gt_parser
        self.anchor_params = anchor_params
           
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        gt_data = self.gt_parser.get_gt_data(idx) 
        img = cv2.imread(gt_data['img_path'])
        if self.transform is not None:
            img = self.transform(img)
        anchors = generate_anchors(img.shape[1:3], 
                                   self.anchor_params['receptive_field'], 
                                   self.anchor_params['scales'], 
                                   self.anchor_params['ratios']
                  )
        #classes = gt_data['classes']
        bboxes = gt_data['bboxes']
        #difficult = gt_data['bboxes']
        
        positives = obtain_positive_anchors(anchors, bboxes)
        return img, positives
               
        
