# -*- coding: utf-8 -*-
from loss.rpn_loss import rpn_loss
from loss.fast_rcnn_loss import fast_rcnn_loss

def faster_rcnn_loss(rpn_gt, rpn_op, fast_rcnn_gt, fast_rcnn_op):   
    rpn_cls_gt, rpn_reg_gt = rpn_gt
    rpn_cls_op, rpn_reg_op = rpn_op
    #print(rpn_cls_gt.dtype, rpn_reg_gt.dtype)
    #print(rpn_cls_op.dtype, rpn_reg_op.dtype)
    rpn_cls_loss, rpn_reg_loss = rpn_loss(
                                        rpn_cls_op, 
                                        rpn_cls_gt, 
                                        rpn_reg_op, 
                                        rpn_reg_gt
                                    )  
    #print(rpn_cls_loss, rpn_reg_loss)
    fast_rcnn_cls_gt, fast_rcnn_reg_gt = fast_rcnn_gt
    fast_rcnn_cls_op, fast_rcnn_reg_op = fast_rcnn_op   
    #print(fast_rcnn_cls_gt.dtype, fast_rcnn_reg_gt.dtype)    
    #print(fast_rcnn_cls_op.dtype, fast_rcnn_reg_op.dtype)    
    fast_rcnn_cls_loss, fast_rcnn_reg_loss = fast_rcnn_loss(
                                        fast_rcnn_cls_op, 
                                        fast_rcnn_cls_gt, 
                                        fast_rcnn_reg_op, 
                                        fast_rcnn_reg_gt
                                    )
    #print(fast_rcnn_cls_loss, fast_rcnn_reg_loss)
    
    total_loss = rpn_cls_loss + 50*rpn_reg_loss + \
                 fast_rcnn_cls_loss + 10*fast_rcnn_reg_loss
                 
    loss_dict = {'rpn_cls': rpn_cls_loss,
                 'rpn_reg': 50*rpn_reg_loss,
                 'fast_rcnn_cls' : fast_rcnn_cls_loss,
                 'fast_rcnn_reg' : 10*fast_rcnn_reg_loss
             }
    return total_loss, loss_dict
