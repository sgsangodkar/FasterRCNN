#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 07:26:17 2021

@author: sagar
"""
import os
import numpy as np
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET

# https://docs.python.org/3/library/xml.etree.elementtree.html

class ParseGTxmls(object):
    def __init__(self, data_path, data_type):
        super().__init__()
        self.data_path = data_path
        ids_file = os.path.join(data_path,'ImageSets/Main', data_type+'.txt')  
        with open(ids_file, 'r') as f:
            self.img_ids = [x.strip() for x in f.readlines()]    
            
    def get_gt_class_and_bbox(self, idx):
        bboxes = []
        classes = []
        difficult = []
        xml_path = os.path.join(self.data_path, 
                                'Annotations', 
                                self.img_ids[idx]+'.xml')
        tree = ET.parse(xml_path)
        objects = tree.findall('object')    
        for obj in objects:
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text) - 1
            ymin = int(bndbox.find('ymin').text) - 1
            xmax = int(bndbox.find('xmax').text) - 1
            ymax = int(bndbox.find('ymax').text) - 1
            bboxes.append((xmin,ymin,xmax,ymax))
            classes.append(obj.find('name').text)
            difficult.append(obj.find('difficult').text)
                
        return (self.img_ids[idx], 
               dict(classes = classes,
                    bboxes = bboxes,
                    difficult = difficult
               )
        )
    
        
     