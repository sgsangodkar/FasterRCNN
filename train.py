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
   

dataloader = DataLoader(dataset, 
                        batch_size=1, 
                        shuffle=True, 
                        pin_memory=True
                   )


     
log_step=0   
for epoch in range(config.epochs):
    print(epoch+1)
    for it, data in enumerate(1, dataloader):
        img = data[0].to(device)
        bboxes = data[1].squeeze(0).to(device)
        classes = data[2].squeeze(0).to(device)
        #print(bboxes.shape, img.shape, classes.shape)
    
        trainer.train_step(img, bboxes, classes, it)
        
        if config.log and (it%100)==0:
             writer.add_scalar('RPN_cls', trainer.meters['rpn_cls'].mean, log_step)      
             writer.add_scalar('RPN_reg', trainer.meters['rpn_reg'].mean, log_step)      
             writer.add_scalar('FastRCNN_cls', trainer.meters['fast_rcnn_cls'].mean, log_step)      
             writer.add_scalar('FastRCNN_reg', trainer.meters['fast_rcnn_reg'].mean, log_step)      
             log_step+=1
       
#a = torch.tensor([-0.0075,  0.1660, -0.1233,  0.3146])
#b = torch.tensor([-0.4346, -0.0029, -0.0311, -0.0042])  
#F.smooth_l1_loss(a,b)     
    
"""        
torch.save(faster_rcnn.state_dict(), 'checkpoint.pt')

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
"""  
