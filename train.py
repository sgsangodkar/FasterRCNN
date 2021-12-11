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
import torch.optim as optim
from torch.optim import lr_scheduler
from configs import train_configs, data_configs
from model import FasterRCNN
from torch.utils.tensorboard import SummaryWriter


device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_loss = 1e5

since = time.time()
writer = SummaryWriter()
 
faster_rcnn = FasterRCNN(train_configs.phase, device, writer)



dataset = VOCDataset(data_configs)
   

dataloader = DataLoader(dataset, 
                        batch_size=1, 
                        shuffle=True, 
                        pin_memory=True
                   )

#print([param[0] for param in faster_rcnn.named_parameters()])

optimizer = optim.SGD(faster_rcnn.parameters(), 
                      lr=train_configs.lr, 
                      weight_decay=0.0005, 
                      momentum=0.9
                )

# Decay LR by a factor of 0.1 every 10 epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=20000, gamma=0.1)

for epoch in range(train_configs.epochs):
    for it, data in enumerate(dataloader):
        img = data[0].to(device)
        bboxes = data[1].squeeze(0).to(device)
        classes = data[2]
    
        with torch.set_grad_enabled(True):
            loss = faster_rcnn(img, bboxes, classes)
          
         
        optimizer.zero_grad()                        
        loss.backward()
        optimizer.step()
        scheduler.step()
    
         
torch.save(faster_rcnn.state_dict(), 'checkpoint.pt')

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
  