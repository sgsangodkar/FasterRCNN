#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 12:33:39 2021

@author: sagar
"""

import time
import torch
import copy
from tqdm import tqdm
from dataset import VOCDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from configs import config
from trainer import FasterRCNNTrainer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_loss = 1e5

since = time.time()
writer = SummaryWriter()
 
trainer = FasterRCNNTrainer(device)


dataset = VOCDataset(config.data_path, 
                     config.data_type,
                     config.min_size, 
                     config.random_flips
                 )
   

def custom_collate(batch):
    imgs = []
    gt_bboxes = []
    gt_classes = []
    gt_difficults = []
    for img, bboxes, classes, difficult in batch:
        imgs.append(img)
        gt_bboxes.append(bboxes)
        gt_classes.append(classes)
        gt_difficults.append(difficult)
    
    return [imgs, gt_bboxes, gt_classes, gt_difficults]
  
dataloader = DataLoader(dataset, 
                        batch_size=config.batch_size, 
                        shuffle=True, 
                        collate_fn = custom_collate,
                        pin_memory=True
                   )


     
log_step=0   
for epoch in range(config.epochs):
    print("Epoch: {}".format(epoch+1))
    for data in dataloader:
        img = data[0]
        bboxes = data[1]
        classes = data[2]
        #print(log_step)
        #since=time.time()
        trainer.train_step(img, bboxes, classes)
        #trainer.val_step(img, bboxes, classes)
        #print(time.time()-since)

        if config.log:
             writer.add_scalar('RPN_cls', trainer.meters['rpn_cls'].mean, log_step)      
             writer.add_scalar('RPN_reg', trainer.meters['rpn_reg'].mean, log_step)      
             writer.add_scalar('FastRCNN_cls', trainer.meters['fast_rcnn_cls'].mean, log_step)      
             writer.add_scalar('FastRCNN_reg', trainer.meters['fast_rcnn_reg'].mean, log_step)      
             log_step+=1
       

    filename = str(epoch+1)+'_checkpoint.pt'    
    torch.save(trainer.state_dict(), filename)

