#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 08:20:11 2021

@author: sagar
"""
import cv2
import torch
import random
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from dataset.xml_parser import ParseGTxmls
from utils import hflip_bboxes, hflip_img

class VOCDataset(Dataset):
    def __init__(self, data_path, data_type, min_size, random_flips):
        super().__init__()
        self.gt_parser = ParseGTxmls(data_path, data_type)
        self.transform = Transformer(min_size, random_flips)
           
    def __len__(self):
        return len(self.gt_parser.img_ids)
    
    def __getitem__(self, idx):
        gt_data = self.gt_parser.get_gt_data(idx) 
        img = cv2.imread(gt_data['img_path'])
        bboxes = gt_data['bboxes']
        classes = gt_data['classes']
        difficult = gt_data['difficult']
        
        return self.transform(img, bboxes, classes, difficult)
               
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
                
    def __call__(self, img, bboxes, classes, difficult):
        ht, wt, _ = img.shape
        img = self.img_transform(img)
        _, ht_new, wt_new = img.shape
        
        sx, sy = wt_new/wt, ht_new/ht
        bboxes = self.scale_bboxes(bboxes, sx, sy)

        classes = torch.tensor(classes)
        difficult = torch.tensor(difficult)
        
        if self.random_flips:
            if random.random()>0.5:
                img = hflip_img(img)
                bboxes = hflip_bboxes(bboxes, (ht_new, wt_new))
       
        return img, bboxes, classes, difficult
 
    def scale_bboxes(self, bboxes, sx, sy):
        bboxes_scaled = np.zeros((len(bboxes),4))   
        for i, bbox in enumerate(bboxes):
            bboxes_scaled[i] = [int(bbox[0]*sx), 
                                int(bbox[1]*sy), 
                                int(bbox[2]*sx), 
                                int(bbox[3]*sy)
                               ]
        return torch.tensor(bboxes_scaled, dtype=torch.float32)       

if __name__ == '__main__': 
            
    dataset = VOCDataset(data_path='../../voc_data/VOCdevkit/VOC2012',
            data_type='trainval',
            min_size=600,
            random_flips=True)
    
    img, bboxes, classes, difficult = dataset[6]
