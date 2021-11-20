#/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sat Nov 20 14:16:06 2021

@author: sagar
"""
import torch
import torch.nn as nn
from torchvision import models

class FeatureExtractor(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        if model_name == 'vgg16':
            model = models.vgg16(pretrained=True)
            self.fe = model.features[:-1]   
        else:
            print('Invalid model name')
            
            
    def forward(self, x):
        features = self.fe(x)
        return features