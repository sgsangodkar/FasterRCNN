#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 10:40:56 2021

@author: sagar
"""
import os
import cv2
import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import Dataset
from utils import generate_anchors, scale_bboxes, assign_label_and_gt_bbox, get_t_parameters

class VOCDataset(Dataset):
    def __init__(self, transform, gt_parser, anchor_params):
        super().__init__()
        self.transform = transform
        self.gt_parser = gt_parser
        self.anchor_params = anchor_params
           
    def __len__(self):
        return len(self.gt_parser.img_ids)
    
    def __getitem__(self, idx):
        gt_data = self.gt_parser.get_gt_data(idx) 
        bboxes = gt_data['bboxes']
        img_np = cv2.imread(gt_data['img_path'])
        if self.transform is not None:
            img = self.transform(img_np)
        s_x = img.shape[1]/img_np.shape[0]
        s_y = img.shape[2]/img_np.shape[1]
        bboxes = scale_bboxes(bboxes, s_x, s_y)
        anchors = generate_anchors(img.shape[1:3], 
                                   self.anchor_params['receptive_field'], 
                                   self.anchor_params['scales'], 
                                   self.anchor_params['ratios']
                  )

        anchor_labels, gt_bboxes_id = assign_label_and_gt_bbox(anchors, bboxes)

        anchors = get_t_parameters(anchors, bboxes, gt_bboxes_id)

        return img, torch.tensor(anchor_labels, dtype=torch.long), torch.tensor(anchors)
               
