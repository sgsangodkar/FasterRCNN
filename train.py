#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 12:33:39 2021

@author: sagar
"""

import time
import torch
from tqdm import tqdm
from dataset import VOCDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from configs import config
from trainer import FasterRCNNTrainer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_loss = 1e5

since = time.time()

 
trainer = FasterRCNNTrainer(device)

if config.resume_prefix is not None:
    trainer.load_model(config.resume_prefix, load_train_state=True)
  

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

voc_dataset = VOCDataset(config.data_path, 
                         data_type='trainval',
                         min_size=600, 
                         random_flips=True
                     )
  
voc_dataloader = DataLoader(voc_dataset, 
                            batch_size=config.batch_size, 
                            shuffle=True, 
                            collate_fn = custom_collate,
                            pin_memory=True
                        )

    
for epoch in range(config.epochs):
    print("Epoch: {}".format(epoch+1))
    for data in tqdm(voc_dataloader):
        img = data[0]
        bboxes = data[1]
        classes = data[2]
        trainer.train_step(img, bboxes, classes)
  

    prefix = 'trained_final'    
    trainer.save_model(prefix, save_train_state=True)

