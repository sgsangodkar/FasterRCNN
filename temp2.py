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


#################################
import cv2
import matplotlib.pyplot as plt
path = '/home/sagar/Desktop/voc_data/VOCdevkit/VOC2012/JPEGImages/2008_000003.jpg'
img_np = cv2.imread(path)
img = transform(img_np)

plt.imshow(img_np)
plt.show()
print(img_np.shape, img.shape)
# shape gives (ht, wt)
# ==> x:vertical, y:horizontal

##################################
gt_data = gt_parser.get_gt_data(1)
bboxes = gt_data['bboxes']
from utils import scale_bboxes
s_x = img.shape[1]/img_np.shape[0] #vertical
s_y = img.shape[2]/img_np.shape[1] #horizontal
bboxes2 = scale_bboxes(bboxes, s_x, s_y)
print(bboxes, bboxes2)

#####################

def func(a):
    a = a+2
    return a

a = 4
b = func(a)

import copy
def func(a):
    b=copy.deepcopy(a)
    #b=a
    for i, x in enumerate(a):
        b[i] = [x[0]*2, x[1]*2, x[2]*3, x[3]*3]        
    return b
#https://docs.python.org/3/library/copy.html
a = [[4, 2, 3, 1]]
b=func(a)
print(a,b)

def func(a):
    #b=copy.deepcopy(a)
    b=a
    for i, x in enumerate(a):
        b[i] = x*2        
    return b
#https://docs.python.org/3/library/copy.html
a = [4, 2, 3, 1]
b=func(a)
print(a,b)


##################
import numpy as np
a = np.array([1,2,3,4,5,6])
b = np.array([2,3,5])
c = set(a) ^ set(b)

print(c, np.array(list(c)))


###############################
from tqdm import tqdm
import time
for i in tqdm(a, desc = 'tqdm() Progress Bar'):
    time.sleep(0.5)
    
###################################

models_dict['fe'].eval()
models_dict['rpn'].eval()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#for image, cls_gt, reg_gt in tqdm(train_dataloader):
image, cls_gt, reg_gt = next(iter(val_dataloader))
image = image.to(device)
cls_gt = cls_gt.squeeze().to(device)
reg_gt = reg_gt.squeeze().to(device)

with torch.set_grad_enabled(False):
    features = models_dict['fe'](image)
    cls_op, reg_op = models_dict['rpn'](features)  

cls_op = cls_op.permute(0,2,3,1).contiguous().view(1,-1,2).squeeze()
reg_op = reg_op.permute(0,2,3,1).contiguous().view(1,-1,4).squeeze()                    

print(reg_op[cls_gt==1])
print(reg_gt[cls_gt==1])

print(rpn_loss(cls_op, reg_op, cls_gt, reg_gt))

####################
(pred_logits, pred_reg, gt_cls, gt_reg) = (cls_op, reg_op, cls_gt, reg_gt)
