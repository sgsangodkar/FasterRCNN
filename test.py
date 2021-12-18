#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 05:44:51 2021

@author: sagar
"""
import torch
import time

bboxes_gt = torch.randn(10,4)
classes_gt=torch.tensor([5,1,0,2,3,4,4,5,6,2]) # GT classes of 10 GT bounding boxes
gt_id = torch.tensor([1,2,3,1,5,9]) # Overlapping GT id for 6 ROIs
indx_v = torch.tensor([1,2,4,5]) # Valid indices for ROIs
n_neg = 2

since=time.time()
cls_gt_v = torch.tensor(list(map(lambda x:classes_gt[x]+1, gt_id[indx_v])))
cls_gt_v[n_neg:]=0
bboxes_v = torch.vstack(list(map(lambda x:bboxes_gt[x], gt_id[indx_v])))
print(time.time()-since)

since=time.time()
cls_gt_v = classes_gt[gt_id[indx_v]]+1
cls_gt_v[n_neg:]=0
bboxes_v = bboxes_gt[gt_id[indx_v]]
print(time.time()-since)
#0.0007402896881103516
#0.0006067752838134766
########################

bboxes_gt = torch.randn(10,4)
classes_gt=torch.tensor([5,1,0,2,3,4,4,5,6,2]) # GT classes of 10 GT bounding boxes
gt_id = torch.randint(10,(100,)) # Overlapping GT id for 6 ROIs
indx_v = torch.randint(100,(50,)) # Valid indices for ROIs
n_neg = 50

since=time.time()
cls_gt_v = torch.tensor(list(map(lambda x:classes_gt[x]+1, gt_id[indx_v])))
cls_gt_v[n_neg:]=0
bboxes_v = torch.vstack(list(map(lambda x:bboxes_gt[x], gt_id[indx_v])))
print(time.time()-since)

since=time.time()
cls_gt_v = classes_gt[gt_id[indx_v]]+1
cls_gt_v[n_neg:]=0
bboxes_v = bboxes_gt[gt_id[indx_v]]
print(time.time()-since)
#0.0015075206756591797
#0.0008707046508789062
#########################################
import torch

a = torch.tensor([1.0,2.0,3.0], requires_grad=True)

b = torch.tensor([1.0], requires_grad=True)

c = a+b

print(a,b,c)

op=c.sum()

#d = c.detach()
d = a.data
op.backward()


d[0]=0

op.backward()

c[2]=6

op.backward()

a.grad


bboxes_gt = torch.randn((10,4))
class_gt = torch.arange(1,11)
mask = torch.where(class_gt>8)[0]
