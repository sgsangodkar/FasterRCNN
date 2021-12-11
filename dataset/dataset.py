#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 08:20:11 2021

@author: sagar
"""
import os
import cv2
import torch
import copy
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset

from dataset.xml_parser import ParseGTxmls


class VOCDataset(Dataset):
    def __init__(self, configs):
        super().__init__()
        self.gt_parser = ParseGTxmls(configs.data_path, configs.data_type)
        self.transform = Transformer(configs.min_size, configs.random_flips)
           
    def __len__(self):
        return len(self.gt_parser.img_ids)
    
    def __getitem__(self, idx):
        gt_data = self.gt_parser.get_gt_data(idx) 
        img = cv2.imread(gt_data['img_path'])
        bboxes = gt_data['bboxes']
        classes = gt_data['classes']
        
        img, bboxes = self.transform(img, bboxes)
        return img, bboxes, classes
               
class Transformer(object):
    def __init__(self, min_size, random_flips=True):
        self.min_size = min_size
        self.random_flips = random_flips
        self.img_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.min_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), 
                                     (0.229, 0.224, 0.225))
            ])        
    def __call__(self, img, bboxes):
        ht, wt, _ = img.shape
        img = self.img_transform(img)
        _, ht_new, wt_new = img.shape
        
        sx, sy = wt_new/wt, ht_new/ht
        bboxes = self.scale_bboxes(bboxes, sx, sy)
        return img, bboxes
 
    def scale_bboxes(self, bboxes, sx, sy):
        bboxes_scaled = np.empty((len(bboxes),4))   
        for i, bbox in enumerate(bboxes):
            bboxes_scaled[i] = [int(bbox[0]*sx), 
                                int(bbox[1]*sy), 
                                int(bbox[2]*sx), 
                                int(bbox[3]*sy)
                               ]
        return torch.tensor(bboxes_scaled)       

if __name__ == '__main__':
    data_configs = dict(
            data_path='../../voc_data/VOCdevkit/VOC2012',
            data_type='trainval',
            min_size=600,
            random_flips=True
            )
    dataset = VOCDataset(data_configs)
    
    img, bboxes, classes = dataset[0]