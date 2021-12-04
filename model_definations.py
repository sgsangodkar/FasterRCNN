#/usr/bin/env python3
"""
Created on Sat Nov 20 14:16:06 2021

@author: sagar
"""


import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        if model_name == 'vgg16':
            model = models.vgg16(pretrained=True, progress=False)
            self.fe = model.features[:-1]   
        else:
            print('Invalid model name')
            
            
    def forward(self, x):
        features = self.fe(x)
        return features
        
class RPN(nn.Module):
    def __init__(self, in_channels, out_channels, num_proposals):
        super().__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Conv2d(out_channels, 2*num_proposals, kernel_size=1)
        self.regressor = nn.Conv2d(out_channels, 4*num_proposals, kernel_size=1)
        self.init_layers()
        
    def init_layers(self):
        self.conv3x3.weight.data.normal_(0, 0.01)
        self.conv3x3.bias.data.zero_()
        self.classifier.weight.data.normal_(0, 0.01)
        self.classifier.bias.data.zero_()
        self.regressor.weight.data.normal_(0, 0.01)
        self.regressor.bias.data.zero_()
        
    def forward(self, x):
        x = F.relu(self.conv3x3(x))
        cls_scores = self.classifier(x)
        reg_coor = self.regressor(x)
        
        return cls_scores, reg_coor
    
