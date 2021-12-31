# -*- coding: utf-8 -*-

import torch.nn as nn
from torchvision import models
from model.rpn import RPN
from model.fast_rcnn import FastRCNN
from torchvision.ops import RoIPool

class FasterRCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        vgg = models.vgg16(pretrained=True, progress=False)
        self.fe = vgg.features[:-1]   
                    
        self.rpn = RPN(in_chan=512,
                       out_chan=512,
                       anchors_per_loc=9
                   )
     
        """
        RoIPool
            def __init__(output_size, spatial_scale):
            output_size (int or Tuple[int, int]): 
                The size of the output after pooling
            spatial_scale (float): 
                A scaling factor that maps the input coordinates to
            the box coordinates. Default: 1.0
        """           
        self.roi_layer = RoIPool(output_size=(7,7), 
                                 spatial_scale=1/16
                             )  
        
        self.fast_rcnn = FastRCNN(fc_layers=vgg.classifier,
                                  num_classes=num_classes
                              )

   