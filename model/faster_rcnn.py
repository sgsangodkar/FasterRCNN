# -*- coding: utf-8 -*-

import torch.nn as nn
from torchvision import models
from model.rpn import RPN
from model.fast_rcnn import FastRCNN

class FasterRCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        vgg = models.vgg16(pretrained=True, progress=False)
        self.fe = vgg.features[:-1]   
                    
        self.rpn = RPN(config.in_chan,
                       config.out_chan,
                       config.anchors_per_location
                   )
        
        self.fast_rcnn = FastRCNN(vgg.classifier,
                                  config.num_classes
                              )

   