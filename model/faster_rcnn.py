# -*- coding: utf-8 -*-

import torch.nn as nn
from torchvision import models
from model.rpn import RPN
from model.fast_rcnn import FastRCNN

class FasterRCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        vgg = models.vgg16(pretrained=True, progress=False)
        self.fe = vgg.features[:-1]   
                    
        self.rpn = RPN(in_chan=512,
                       out_chan=512,
                       anchors_per_loc=9
                   )
        
        self.fast_rcnn = FastRCNN(vgg.classifier,
                                  num_classes
                              )

   