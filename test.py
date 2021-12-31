#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 19:40:36 2021

@author: sagar
"""
import time
import torch
import numpy as np
from tqdm import tqdm
from dataset import VOCDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from configs import config
from model import FasterRCNN
from utils import gen_anchors, gen_rois, reg2bbox, visualize_bboxes
from torchvision.ops import nms
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = FasterRCNN(config.num_classes).to(device)

model_filename = 'final/model_params.pt'
model_state_dict = torch.load(model_filename, map_location=torch.device('cpu'))
model.load_state_dict(model_state_dict)
model.eval()


voc_dataset = VOCDataset(config.data_path, 
                         data_type='test',
                         min_size=600, 
                         random_flips=False
                     )
  
voc_dataloader = DataLoader(voc_dataset, 
                            batch_size=1, 
                            shuffle=True,
                            pin_memory=True
                        )
 
        

for data in voc_dataloader:
    img = data[0].to(device)
    bboxes = data[1]
    classes = data[2]
      
    with torch.no_grad():
        features = model.fe(img)
        
    _,_,H,W = img.shape
    img_size = (H,W)

    anchors = gen_anchors(
            img_size, 
            receptive_field=16, 
            scales=[8,16,32], 
            ratios=[0.5,1,2]
        )  
         
    with torch.no_grad():
        cls_op, reg_op = model.rpn(features)
        
    cls_op = cls_op.permute(0,2,3,1).contiguous().view(1,-1,2).squeeze()
    reg_op = reg_op.permute(0,2,3,1).contiguous().view(1,-1,4).squeeze()
    
    rois = gen_rois(cls_op.detach(), 
                    reg_op.detach(), 
                    anchors, 
                    img_size
                ) # x1, y1, x2, y2
    
    
    pool_feats = model.roi_layer(features, [rois])
    pool_feats = pool_feats.view(pool_feats.size(0), -1)
    
    with torch.no_grad():
        cls_op, reg_op = model.fast_rcnn(pool_feats)
    
    classes = torch.argmax(cls_op, axis=1)
    reg_op = reg_op.view(len(reg_op), -1, 4)


    cls_op = F.softmax(cls_op, dim=1)
    
    final_bboxes = []
    final_classes = []
    
    for cid in range(config.num_classes):
        fg_scores = cls_op[:, cid+1]
        mask = fg_scores>0.9
        fg_scores = fg_scores[mask]
        bboxes = reg2bbox(rois[mask], reg_op[mask, cid])
        
        bboxes[:, 0::2] = (bboxes[:, 0::2]).clamp(min=0, max=W)
        bboxes[:, 1::2] = (bboxes[:, 1::2]).clamp(min=0, max=H)

        indices = nms(bboxes, fg_scores, 0.2)
        bboxes = bboxes[indices]
       
        if len(bboxes):
            final_bboxes.append(bboxes.numpy())
            for _ in range(len(bboxes)):
                final_classes.append(cid)
     
    if len(final_bboxes):
        final_bboxes = np.vstack(final_bboxes)
        img_np = visualize_bboxes(img, final_bboxes, final_classes)
        plt.imshow(img_np)
        plt.show()
    else:
        print('No Object Detected')
     
