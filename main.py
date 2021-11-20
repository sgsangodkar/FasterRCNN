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

from model_definations import RPN, FeatureExtractor


device = 'cuda' if torch.cuda.is_available() else 'cpu'


fe = FeatureExtractor('vgg16').to(device)

rpn = RPN(512, 512, 9)


dummy_image = torch.zeros((1, 3, 600, 600)).float()

bbox = torch.FloatTensor([[20, 30, 400, 500], [300, 400, 500, 600]]) # [y1, x1, y2, x2] format
labels = torch.LongTensor([6, 8]) # 0 represents background


print(fe(dummy_image.to(device)).shape)

aspect_ratios = [0.5,1,2]
anchor_scales = [128,256,512]
