#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 07:45:18 2021

@author: sagar
"""
import os
import cv2
import torchvision.transforms as transforms
from utils import generate_anchors, calculate_iou, obtain_iou_matrix
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

data_path = '/home/sagar/Desktop/voc_data/VOCdevkit/VOC2012'
data_type = 'trainval'

ids_file = os.path.join(data_path,'ImageSets/Main', data_type+'.txt')  
with open(ids_file, 'r') as f:
    img_ids = [x.strip() for x in f.readlines()]    
    
bboxes = []
classes = []
difficult = []
idx = 12
xml_path = os.path.join(data_path, 
                        'Annotations', 
                        img_ids[idx]+'.xml')
tree = ET.parse(xml_path)
objects = tree.findall('object')    

img_path = os.path.join(data_path, 'JPEGImages', img_ids[idx]+'.jpg')     

for obj in objects:
    bndbox = obj.find('bndbox')
    xmin = int(bndbox.find('xmin').text) - 1
    ymin = int(bndbox.find('ymin').text) - 1
    xmax = int(bndbox.find('xmax').text) - 1
    ymax = int(bndbox.find('ymax').text) - 1
    bboxes.append((xmin,ymin,xmax,ymax))
    classes.append(obj.find('name').text)
    difficult.append(obj.find('difficult').text)
    
gt_data = dict(img_path = img_path,
             classes = classes,
             bboxes = bboxes,
             difficult = difficult
       )

img_np = cv2.imread(gt_data['img_path'])
transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(800),
                transforms.ToTensor()]
        )
img = transform(img_np)
s_x = img.shape[1]/img_np.shape[0]
s_y = img.shape[2]/img_np.shape[1]

anchor_params = dict(receptive_field=16,
                     scales = [8,16,32],
                     ratios = [0.5,1,2]
                )

anchors = generate_anchors(img.shape[1:3], 
                           anchor_params['receptive_field'], 
                           anchor_params['scales'], 
                           anchor_params['ratios']
          )
#classes = gt_data['classes']
bboxes = gt_data['bboxes']
def scale_bboxes(bboxes, sx, sy):
    for i, bbox in enumerate(bboxes):
        bboxes[i] = [bbox[0]*sx, bbox[1]*sy, bbox[2]*sx, bbox[3]*sy]
        
    return bboxes
bboxes = scale_bboxes(bboxes, s_x, s_y)

def label_anchors(anchors, bboxes):
    anchors = np.array(anchors)
    anchor_labels = np.empty(len(anchors), dtype=np.int32)
    anchor_labels.fill(-1)
    
    iou_matrix = obtain_iou_matrix(anchors, bboxes)
    max_iou_per_bbox = np.max(iou_matrix, axis=0)
    max_iou_per_anchor = np.max(iou_matrix, axis=1)
    min_iou_per_anchor = np.min(iou_matrix, axis=1)
    anchor_labels[max_iou_per_anchor>0.7] = 1
    anchor_labels[min_iou_per_anchor<0.3] = 0
    for i in range(len(bboxes)):
        anchor_labels[iou_matrix[:,i]==max_iou_per_bbox[i]] = 1

    return anchor_labels

anchor_labels = label_anchors(anchors, bboxes)
print(np.unique(anchor_labels))
print(np.sum(anchor_labels==1))
print(np.sum(anchor_labels==0))
print(np.sum(anchor_labels==-1))

im = np.ascontiguousarray(img.transpose(0,2).transpose(0,1))
'''
for cx in bboxes:
    start = (int(cx[0]), int(cx[1]))
    end = (int(cx[2]), int(cx[3]))
    im = cv2.rectangle(im, start, end, (255, 255, 255), 2)   
'''
anchors = np.array(anchors)
a = anchors[anchor_labels==1,:]  

for cx in a:
    start = (int(cx[0]), int(cx[1]))
    end = (int(cx[2]), int(cx[3]))
    cim = cv2.rectangle(im, start, end, (255, 255, 255), 7)    
    
plt.imshow(im)


pos_markers = np.where(anchor_labels==1)[0]
neg_markers = np.where(anchor_labels==0)[0]
anchors_final = np.empty((256,4))
labels_final = np.empty((256))
num = min(128, len(pos_markers))
selection = np.random.choice(pos_markers, num)
anchors_final[0:num,:] = anchors[selection]
labels_final[0:num] = 1
if num < 128:
    neg = 128 + (128-num)
else:
    neg = 128
selection = np.random.choice(neg_markers, neg)
anchors_final[num:,:] = anchors[selection]
labels_final[num:] = 0

    