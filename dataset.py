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
from utils import generate_anchors, rm_cross_boundary_anchors

class VOCDataset(Dataset):
    def __init__(self, data_path, data_type, transform, gt_parser, anchors):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.gt_parser = gt_parser
        self.anchors = anchors
           
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_name, gt_data = self.gt_parser.get_gt_class_and_bbox(idx) 
        img = cv2.imread(os.path.join(self.data_path, 'JPEGImages', img_name))
        if self.transform is not None:
            img = self.transform(img)
        
        classes = gt_data['classes']
        bboxes = gt_data['bboxes']
        difficult = gt_data['bboxes']
        
        return bboxes
               
        
