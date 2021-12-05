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
from tqdm import tqdm

#from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_model(models, dataloaders, optimizer, scheduler, num_epochs):
    best_loss = 1e5
    best_avg_running_loss = 1e5

    since = time.time()
    #Writer will output to ./runs/ directory by default
    #writer = SummaryWriter()

    for epoch in range(num_epochs):
        print('Epoch {}'.format(epoch))
        for phase in ['train', 'val']:

            epoch_loss = 0
            running_loss = 0
            running_cls_loss = 0
            running_reg_loss = 0
            
            if phase == 'train':
                models['fe'].train()
                models['rpn'].train()
            else:
                models['fe'].eval()
                models['rpn'].eval()
            
            acc_steps = 16
            disp_steps = 16
            num_images = 0
            
            optimizer.zero_grad()
            
            for image, cls_gt, reg_gt in (dataloaders[phase]):
                num_images+=1
                image = image.to(device)
                cls_gt = cls_gt.squeeze().to(device)
                reg_gt = reg_gt.squeeze().to(device)
                
                
                
                with torch.set_grad_enabled(phase=='train'):
                    features = models['fe'](image)
                    cls_op, reg_op = models['rpn'](features)  
                
                cls_op = cls_op.permute(0,2,3,1).contiguous().view(1,-1,2).squeeze()
                reg_op = reg_op.permute(0,2,3,1).contiguous().view(1,-1,4).squeeze()                    
                
                #print(reg_op[cls_gt==1], reg_gt[cls_gt==1])
                cls_loss, reg_loss = rpn_loss(cls_op, reg_op, cls_gt, reg_gt)
                loss = cls_loss + 10*reg_loss
                
                # epoch statistics
                epoch_loss += loss.item()  
                running_loss += loss.item()                
                running_cls_loss += cls_loss.item()     
                if reg_loss!=0:
                    running_reg_loss += 10*reg_loss.item()                
                
            
                if phase == 'train':
                    loss = loss / acc_steps 
                    loss.backward()
                    if num_images%acc_steps == 0: # Wait for several backward steps
                            optimizer.step()          # Now we can do an optimizer step
                            optimizer.zero_grad()
        

                
                if num_images%disp_steps == 0:
                    avg_running_loss = running_loss/disp_steps
                    avg_running_cls_loss = running_cls_loss/disp_steps
                    avg_running_reg_loss = running_reg_loss/disp_steps
                    print('Iter: {}. Loss: {:.4f}. CLS Loss: {:.4f}. REG Loss: {:.4f}.'
                          .format(num_images*(epoch+1), 
                                  avg_running_loss, 
                                  avg_running_cls_loss,
                                  avg_running_reg_loss
                    ))
                    
                    if phase == 'train' and avg_running_loss < best_avg_running_loss:
                        best_avg_running_loss = avg_running_loss
                        fe_best_weights = copy.deepcopy(models['fe'].state_dict())
                        rpn_best_weights = copy.deepcopy(models['rpn'].state_dict())
                        torch.save(models['fe'].state_dict(), 't_checkpoint_fe.pt')
                        torch.save(models['rpn'].state_dict(), 't_checkpoint_rpn.pt')
                    running_loss = 0
                    running_cls_loss = 0
                    running_reg_loss = 0
                
                
            if phase == 'train':
                scheduler.step()

            epoch_loss /= num_images

            print('{}. Loss: {:.4f}'.format(phase, epoch_loss))

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                fe_best_weights = copy.deepcopy(models['fe'].state_dict())
                rpn_best_weights = copy.deepcopy(models['rpn'].state_dict())
                torch.save(models['fe'].state_dict(), 'checkpoint_fe.pt')
                torch.save(models['rpn'].state_dict(), 'checkpoint_rpn.pt')

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
    models['fe'].load_state_dict(fe_best_weights)
    models['rpn'].load_state_dict(rpn_best_weights)
    return models
