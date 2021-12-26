#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 12:10:32 2021

@author: sagar
"""
import torch.nn as nn
import torch.nn.functional as F

class RPN(nn.Module):
    def __init__(self, in_chan, out_chan, anchors_per_loc):
        super().__init__()
        self.conv3x3 = nn.Conv2d(
                        in_chan, 
                        out_chan, 
                        kernel_size=3, 
                        stride=1, 
                        padding=1
                    )
        self.classifier = nn.Conv2d(
                            out_chan, 
                            2*anchors_per_loc, 
                            kernel_size=1
                        )
        self.regressor = nn.Conv2d(
                            out_chan, 
                            4*anchors_per_loc, 
                            kernel_size=1
                        )
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
        cls_op = self.classifier(x)
        reg_op = self.regressor(x)

        return cls_op, reg_op
 