#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 19:40:36 2021

@author: sagar
"""
import time
import torch
from tqdm import tqdm
from dataset import VOCDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from configs import config
from model import FasterRCNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = FasterRCNN(config.num_classes).to(device)
model_filename = 'trained_model.pt'
model_state_dict = torch.load(model_filename)
model.load_state_dict(model_state_dict)
model.eval()


voc_dataset = VOCDataset(config.data_path, 
                         data_type='test',
                         min_size=600, 
                         random_flips=False
                     )
  
voc_dataloader = DataLoader(voc_dataset, 
                            batch_size=1, 
                            shuffle=False,
                            pin_memory=True
                        )

        pool_size = 7
        output_size = (pool_size, pool_size)
        spatial_scale = 1/self.receptive_field
        self.roi_layer = RoIPool(output_size, spatial_scale)  
        
for epoch in range(config.epochs):
    print("Epoch: {}".format(epoch+1))
    for data in tqdm(voc_dataloader):
        img = data[0]
        bboxes = data[1]
        classes = data[2]
  
   
        img = img.unsqueeze(0).to(device)
        features = model.fe(img)
        _,_,H,W = img.shape
        img_size = (H,W)

        anchors = gen_anchors(
                img_size, 
                receptive_field=16, 
                scales=[8,16,32], 
                ratios=[0.5,1,2]
            )           
        cls_op, reg_op = model.rpn(features)
        cls_op = cls_op.permute(0,2,3,1).contiguous().view(1,-1,2).squeeze()
        reg_op = reg_op.permute(0,2,3,1).contiguous().view(1,-1,4).squeeze()
        
        rois = gen_rois(cls_op.detach(), 
                        reg_op.detach(), 
                        anchors, 
                        img_size
                    ) # x1, y1, x2, y2
        
        
        if len(rois)>0:
            pool = roi_layer(features, [rois])
            pool_feats = pool.view(pool.size(0), -1)
        
        cls_op, reg_op = model.fast_rcnn(pool_feats)
        
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
         
