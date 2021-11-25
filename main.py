#/usr/bin/env python3
"""
Created on Sat Nov 20 14:16:06 2021

@author: sagar
"""

# https://github.com/ruotianluo/pytorch-faster-rcnn

# https://fractaldle.medium.com/guide-to-build-faster-rcnn-in-pytorch-95b10c273439

# https://github.com/sorg20/RPN

import torch
from torchvision import models
import torchvision.transforms as transforms
from xml_parser import ParseGTxmls
from model_definations import RPN, FeatureExtractor
from dataset import VOCDataset
import cv2
from loss import rpn_loss
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from train import train_model
import torch.optim as optim
from torch.optim import lr_scheduler

device = 'cuda' if torch.cuda.is_available() else 'cpu'


fe = FeatureExtractor('vgg16').to(device)
rpn = RPN(512, 512, 9).to(device)
models_dict = dict(fe=fe, rpn=rpn)

#dummy_image = torch.zeros((1, 3, 600, 600)).float()

#bbox = torch.FloatTensor([[20, 30, 400, 500], [300, 400, 500, 600]]) # [y1, x1, y2, x2] format
#labels = torch.LongTensor([6, 8]) # 0 represents background


#print(fe(dummy_image.to(device)).shape)

data_path = '/home/sagar/Desktop/voc_data/VOCdevkit/VOC2012'
data_type = 'trainval'

transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(800),
                transforms.ToTensor()]
        )
gt_parser = ParseGTxmls(data_path, data_type)
anchor_params = dict(receptive_field=16,
                     scales = [8,16,32],
                     ratios = [0.5,1,2]
                )

dataset = VOCDataset(transform, gt_parser, anchor_params)
lengths = [int(0.9*len(dataset)), int(0.1*len(dataset))]
trainset, valset = random_split(dataset, lengths)

train_dataloader = DataLoader(trainset, 
                              batch_size=1, 
                              shuffle=True, 
                              pin_memory=True
                   )


val_dataloader = DataLoader(valset, 
                            batch_size=1, 
                            shuffle=False, 
                            pin_memory=True
                 )
dataloaders = dict(train = train_dataloader, val = val_dataloader)

optimiser = optim.SGD([
                {'params': models_dict['fe'].fe[10:].parameters(), 'lr': 0.001},
                {'params': models_dict['rpn'].parameters()} 
            ], lr=0.01, weight_decay=0.0005)

# Decay LR by a factor of 0.1 every 10 epochs
scheduler = lr_scheduler.StepLR(optimiser, step_size=5, gamma=0.1)

trained_model = train_model(models_dict, dataloaders, optimiser, scheduler, num_epochs=25)           
 
"""          
img = np.zeros(a.shape[1:3])
for idx in range(len(b)):
    start = (int(b[idx][0]), int(b[idx][1]))
    end = (int(b[idx][2]), int(b[idx][3]))
    img = cv2.rectangle(img, start, end, (255, 255, 255), 1)  
plt.imshow(img, 'gray')
"""