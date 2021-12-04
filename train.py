#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 12:33:39 2021

@author: sagar
"""

import time
import torch
import copy
from loss import rpn_loss
#from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'


<<<<<<< HEAD
def train_model(models, dataloaders, optimiser, scheduler, num_epochs):
=======
def train_model(models_dict, dataloaders, optimiser, scheduler, num_epochs):
>>>>>>> c5533ac035c3bb5239907a9bece3201eba6aa673
    best_loss = 1e5
    since = time.time()
    #Writer will output to ./runs/ directory by default
    #writer = SummaryWriter()

    for epoch in range(num_epochs):
        print('Epoch {}'.format(epoch))
        for phase in ['train', 'val']:

            epoch_loss = 0
            epoch_cls_loss = 0
            epoch_reg_loss = 0
            
            if phase == 'train':
                models_dict['fe'].train()
                models_dict['rpn'].train()
            else:
                models_dict['fe'].eval()
                models_dict['rpn'].eval()
                
            num_images = 0
            for image, cls_gt, reg_gt in dataloaders[phase]:
                image = image.to(device)
                cls_gt = cls_gt.squeeze().to(device)
                reg_gt = reg_gt.squeeze().to(device)
                
                optimiser.zero_grad()
                
                with torch.set_grad_enabled(phase=='train'):
                    features = models_dict['fe'](image)
                    cls_op, reg_op = models_dict['rpn'](features)  
                    
                cls_op = cls_op.permute(0,2,3,1).contiguous().view(1,-1,2).squeeze()
                reg_op = reg_op.permute(0,2,3,1).contiguous().view(1,-1,4).squeeze()                    
<<<<<<< HEAD
                loss = rpn_loss(cls_op, reg_op, cls_gt, reg_gt)
=======
                loss, cls_loss, reg_loss = rpn_loss(cls_op, reg_op, cls_gt, reg_gt)
>>>>>>> c5533ac035c3bb5239907a9bece3201eba6aa673
            
                if phase == 'train':
                    loss.backward()
                    optimiser.step()

                # epoch statistics
                num_images += 1
                epoch_loss += loss.item()       
                epoch_cls_loss += cls_loss.item()       
                epoch_reg_loss += reg_loss.item() 
                
                if num_images%10==0:
                    print('Cls, Reg Loss: {:.2f}, {:.2f}'.format(epoch_cls_loss/num_images, epoch_reg_loss/num_images))
                            
            if phase == 'train':
                scheduler.step()

            epoch_loss /= num_images

            print('{}. Loss: {:.4f}'.format(phase, epoch_loss))

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                fe_best_weights = copy.deepcopy(models_dict['fe'].state_dict())
                rpn_best_weights = copy.deepcopy(models_dict['rpn'].state_dict())
                torch.save(models_dict['fe'].state_dict(), 'checkpoint_fe.pt')
                torch.save(models_dict['rpn'].state_dict(), 'checkpoint_rpn.pt')

            #if phase == 'train':
                #writer.add_scalar('TrainLoss', epoch_loss, epoch)
                #writer.add_scalar('TrainAcc', epoch_acc, epoch)
            #if phase == 'val':
                #writer.add_scalar('ValLoss', epoch_loss, epoch)
                #writer.add_scalar('ValAcc', epoch_acc, epoch)
                
            
            
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
<<<<<<< HEAD
    models['fe'].load_state_dict(fe_best_weights)
    models['rpn'].load_state_dict(rpn_best_weights)
    return models
=======
    models_dict['fe'].load_state_dict(fe_best_weights)
    models_dict['rpn'].load_state_dict(rpn_best_weights)
    return models_dict
>>>>>>> c5533ac035c3bb5239907a9bece3201eba6aa673
