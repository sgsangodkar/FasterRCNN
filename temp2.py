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
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'


fe = FeatureExtractor('vgg16').to(device)

rpn = RPN(512, 512, 9)


dummy_image = torch.zeros((1, 3, 600, 600)).float()

bbox = torch.FloatTensor([[20, 30, 400, 500], [300, 400, 500, 600]]) # [y1, x1, y2, x2] format
labels = torch.LongTensor([6, 8]) # 0 represents background


print(fe(dummy_image.to(device)).shape)

data_path = '/home/sagar/Desktop/voc_data/VOCdevkit/VOC2012'
data_type = 'trainval'

transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(600),
                transforms.ToTensor()]
        )
gt_parser = ParseGTxmls(data_path, data_type)
anchor_params = dict(receptive_field=16,
                     scales = [8,16,32],
                     ratios = [0.5,1,2]
                )

trainset = VOCDataset(transform, gt_parser, anchor_params)
a,b,c = trainset[7]


img = np.zeros(a.shape[1:3])
for idx in range(len(b)):
    start = (int(b[idx][0]), int(b[idx][1]))
    end = (int(b[idx][2]), int(b[idx][3]))
    img = cv2.rectangle(img, start, end, (255, 255, 255), 1)  
plt.imshow(img, 'gray')


#################
a = torch.arange(24)
a = a.reshape(1,6,2,2)
a_new = a.permute(0,2,3,1).contiguous()
a_new = a_new.view(1,-1,2)
print(a, a_new)



a, cls_gt, reg_gt = dataset[11]
features = fe(a.unsqueeze(0))
cls_op, reg_op = rpn(features)
cls_op = cls_op.permute(0,2,3,1).contiguous().view(1,-1,2).squeeze()
reg_op = reg_op.permute(0,2,3,1).contiguous().view(1,-1,4).squeeze()
loss = rpn_loss(cls_op, reg_op, cls_gt.squeeze(), reg_gt.squeeze())