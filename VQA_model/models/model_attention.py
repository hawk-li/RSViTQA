#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 09:55:48 2018

@author: sylvain
"""

import torch.nn as nn
from torch.nn import MultiheadAttention
import torch.nn.functional as F
import torch

VISUAL_OUT = 2048
QUESTION_OUT = 2400
FUSION_IN = 1200
FUSION_HIDDEN = 256
DROPOUT_V = 0.5
DROPOUT_Q = 0.5
DROPOUT_F = 0.5

class VQAModel(nn.Module):
    def __init__(self, input_size=512):
        super(VQAModel, self).__init__()

        self.num_classes = 95
        
        # Dropout
        self.dropoutV = nn.Dropout(DROPOUT_V)
        self.dropoutQ = nn.Dropout(DROPOUT_Q)
        self.dropoutF = nn.Dropout(DROPOUT_F)
        
        # Visual pathway
        output_size = (input_size / 32)**2
        self.visual = nn.Conv2d(2048, int(2048 / output_size), 1)

        self.linear_v = nn.Linear(VISUAL_OUT, FUSION_IN)
        self.ln_linear_v = nn.LayerNorm(FUSION_IN)  # Add BatchNorm after linear layer

        # Question pathway
        self.linear_q = nn.Linear(QUESTION_OUT, FUSION_IN)
        self.ln_linear_q = nn.LayerNorm(FUSION_IN)  # Add BatchNorm after linear layer

        # Multihead attention
        self.multihead_attn = nn.MultiheadAttention(embed_dim=FUSION_IN, num_heads=1, dropout=0.5, batch_first=True)
        self.ln_multihead_attn = nn.LayerNorm(FUSION_IN)  # Layer Normalization after Multihead Attention


        # Classification layers
        self.linear_classif1 = nn.Linear(FUSION_IN, FUSION_HIDDEN)
        self.ln_linear_classif1 = nn.LayerNorm(FUSION_HIDDEN)  # Add BatchNorm after linear layer

        self.linear_classif2 = nn.Linear(FUSION_HIDDEN, self.num_classes)

    def forward(self, input_v, input_q):
        x_v = self.visual(input_v)
        x_v = x_v.view(-1, VISUAL_OUT)
        x_v = self.dropoutV(x_v)
        x_v = self.linear_v(x_v)
        x_v = self.ln_linear_v(x_v)  # Apply BatchNorm
        x_v = nn.Tanh()(x_v)

        x_q = self.dropoutQ(input_q)
        x_q = self.linear_q(x_q)
        x_q = self.ln_linear_q(x_q)  # Apply BatchNorm
        x_q = nn.Tanh()(x_q)

        x_v_new, _ = self.multihead_attn(query=x_q, key=x_v, value=x_v)
        x_v_new = self.ln_multihead_attn(x_v_new)  # Apply Layer Normalization
        # x = torch.mul(x_v_new, x_q)
        # x = torch.squeeze(x, 1)
        # x = nn.Tanh()(x)
        x = x_v_new
        x = self.dropoutF(x)
        x = self.linear_classif1(x)
        x = self.ln_linear_classif1(x)  # Apply BatchNorm
        x = nn.Tanh()(x)
        x = self.dropoutF(x)
        x = self.linear_classif2(x)

        return x