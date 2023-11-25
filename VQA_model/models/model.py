#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 09:55:48 2018

@author: sylvain
"""

from torchvision import models as torchmodels
import torch.nn as nn
import models.seq2vec
import torch.nn.functional as F
import torch
from block import fusions

VISUAL_OUT = 2048
QUESTION_OUT = 2400
FUSION_IN = 1200
FUSION_HIDDEN = 256
DROPOUT_V = 0.5
DROPOUT_Q = 0.5
DROPOUT_F = 0.5

class VQAModel(nn.Module):
    def __init__(self, 
                 input_size = 512):
        
        super(VQAModel, self).__init__()
        
        self.num_classes = 95

        # Initialize CompactBilinearPooling layer
        # Adjust the input1_size, input2_size, and output_size as needed
        self.fusion = fusions.Mutan([FUSION_IN, FUSION_IN], FUSION_IN)
        
        self.dropoutV = torch.nn.Dropout(DROPOUT_V)
        self.dropoutQ = torch.nn.Dropout(DROPOUT_Q)
        self.dropoutF = torch.nn.Dropout(DROPOUT_F)

        self.linear_q = nn.Linear(QUESTION_OUT, FUSION_IN)
        
        output_size = (input_size / 32)**2
        self.visual = torch.nn.Conv2d(2048,int(2048/output_size), 1)
        self.linear_v = nn.Linear(VISUAL_OUT, FUSION_IN)
        
        self.linear_classif1 = nn.Linear(FUSION_IN, FUSION_HIDDEN)
        self.linear_classif2 = nn.Linear(FUSION_HIDDEN, self.num_classes)
        
    def forward(self, input_v, input_q):
        #input_v = torch.squeeze(input_v, 1)
        x_v = self.visual(input_v).view(-1, VISUAL_OUT)
        x_v = self.dropoutV(x_v)
        x_v = self.linear_v(x_v)
        x_v = nn.Tanh()(x_v)

        x_q = self.dropoutV(input_q)
        x_q = self.linear_q(x_q)
        x_q = nn.Tanh()(x_q)
        
        # x = torch.mul(x_v, x_q)
        # x = torch.squeeze(x, 1)
        x = self.fusion([x_v, x_q])
        x = nn.Tanh()(x)
        x = self.dropoutF(x)
        x = self.linear_classif1(x)
        x = nn.Tanh()(x)
        x = self.dropoutF(x)
        x = self.linear_classif2(x)
        return x
        