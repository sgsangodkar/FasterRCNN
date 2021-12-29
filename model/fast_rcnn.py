#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 12:09:54 2021

@author: sagar
"""

import torch.nn as nn

class FastRCNN(nn.Module):
    def __init__(self, 
                 fc_layers, 
                 num_classes,
                 pool_size=7, 
                 receptive_field=16
             ):
        super().__init__()
        self.fc_layers = fc_layers[:-1]
        self.regressor = nn.Linear(4096, num_classes * 4)
        self.classifier = nn.Linear(4096, num_classes+1)

        self.num_classes = num_classes
 
        self.init_layers()
    
        
    def init_layers(self):
        self.classifier.weight.data.normal_(0, 0.01)
        self.classifier.bias.data.zero_()
        self.regressor.weight.data.normal_(0, 0.001)
        self.regressor.bias.data.zero_()

        
    def forward(self, features):
        x = self.fc_layers(features)
        cls_op = self.classifier(x)
        reg_op = self.regressor(x)
        
        return cls_op, reg_op
 