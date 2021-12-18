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
 
trainer = FasterRCNNTrainer(device, writer)


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


     
        
for epoch in range(config.epochs):
    for it, data in enumerate(dataloader):
        print(it)
        img = data[0].to(device)
        bboxes = data[1].squeeze(0).to(device)
        classes = data[2].squeeze(0).to(device)
        #print(bboxes.shape, img.shape, classes.shape)
    
        trainer.train_step(img, bboxes, classes)
        
     
      
    
"""        
torch.save(faster_rcnn.state_dict(), 'checkpoint.pt')

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
"""  
