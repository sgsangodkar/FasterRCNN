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

rpn = RPN(in_channels=512, 
          out_channels=512, 
          num_proposals=9).to(device)

models_dict = dict(fe=fe, rpn=rpn)


#fe.load_state_dict(torch.load('checkpoint_fe.pt' ,map_location=torch.device('cpu')))
#rpn.load_state_dict(torch.load('checkpoint_rpn.pt', map_location=torch.device('cpu')))
                
#dummy_image = torch.zeros((1, 3, 600, 600)).float()

#bbox = torch.FloatTensor([[20, 30, 400, 500], [300, 400, 500, 600]]) # [y1, x1, y2, x2] format
#labels = torch.LongTensor([6, 8]) # 0 represents background


#print(fe(dummy_image.to(device)).shape)

data_path = '../voc_data/VOCdevkit/VOC2012'
data_type = 'trainval'

data_configs = dict(
        data_path='../../voc_data/VOCdevkit/VOC2012',
        data_type='trainval',
        min_size=600,
        random_flips=True
        )
dataset = VOCDataset(data_configs)
   
lengths = [int(0.8*len(dataset)), int(0.2*len(dataset))]
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

             

optimizer = optim.SGD([
                {'params': models_dict['fe'].fe[10:].parameters()},
                {'params': models_dict['rpn'].parameters()} 
            ], lr=0.001, weight_decay=0.0005, momentum=0.9)

# Decay LR by a factor of 0.1 every 10 epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

faster_rcnn_configs = dict()

faster_rcnn = FasterRCNN(faster_rcnn_configs)
trainer = TrainerFasterRCNN(faster_rcnn, train_configs)
trainer.train()

trained_model = train_model(models_dict, dataloaders, optimizer, scheduler, num_epochs=7)           
 
"""          
img = np.zeros(a.shape[1:3])
for idx in range(len(b)):
    start = (int(b[idx][0]), int(b[idx][1]))
    end = (int(b[idx][2]), int(b[idx][3]))
    img = cv2.rectangle(img, start, end, (255, 255, 255), 1)  
plt.imshow(img, 'gray')
"""