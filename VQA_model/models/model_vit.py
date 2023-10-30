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

VISUAL_OUT = 2048
VISUAL_OUT_VIT = 768
QUESTION_OUT = 2400
FUSION_IN = 1200
FUSION_HIDDEN = 256
DROPOUT_V = 0.5
DROPOUT_Q = 0.5
DROPOUT_F = 0.5

class VQAModel(nn.Module):
    def __init__(self, 
                 # vocab_questions, 
                 # vocab_answers, 
                 input_size = 512):
        
        super(VQAModel, self).__init__()
        
        # self.vocab_questions = vocab_questions
        # self.vocab_answers = vocab_answers
        # self.num_classes = len(self.vocab_answers)
        self.num_classes = 95
        
        self.dropoutV = torch.nn.Dropout(DROPOUT_V)
        self.dropoutQ = torch.nn.Dropout(DROPOUT_Q)
        self.dropoutF = torch.nn.Dropout(DROPOUT_F)
        
        # self.seq2vec = models.seq2vec.factory(self.vocab_questions, {'arch': 'skipthoughts', 'dir_st': 'data/skip-thoughts', 'type': 'BayesianUniSkip', 'dropout': 0.25, 'fixed_emb': False})
        # for param in self.seq2vec.parameters():
        #     param.requires_grad = False
        
        self.linear_q = nn.Linear(QUESTION_OUT, FUSION_IN)
        
        # self.visual = torchmodels.resnet152(pretrained=True)
        # extracted_layers = list(self.visual.children())
        # extracted_layers = extracted_layers[0:8] #Remove the last fc and avg pool
        # self.visual = torch.nn.Sequential(*(list(extracted_layers)))
        # for param in self.visual.parameters():
        #     param.requires_grad = False
        
        #output_size = (input_size / 32)**2
        #self.visual = torch.nn.Conv2d(VISUAL_OUT,int(REDUCED_VISUAL_OUT/output_size), 1)
        #self.bottleneck = nn.Linear(VISUAL_OUT, REDUCED_VISUAL_OUT)  # Bottleneck layer

        self.linear_v = nn.Linear(VISUAL_OUT_VIT, FUSION_IN)
        
        self.linear_classif1 = nn.Linear(FUSION_IN, FUSION_HIDDEN)
        self.linear_classif2 = nn.Linear(FUSION_HIDDEN, self.num_classes)
        
    def forward(self, input_v, input_q):
        #input_v = torch.squeeze(input_v, 1)
        #x_v = torch.mean(input_v, dim=1)
        #x_v = self.bottleneck(x_v) 
        #x_v = self.visual(input_v).view(-1, VISUAL_OUT)
        x_v = self.linear_v(input_v)
        #x_v = self.dropoutV(x_v)
        #x_v = self.linear_v(x_v)
        x_v = nn.Tanh()(x_v)
        #x_v = torch.unsqueeze(x_v, 1)
        #print(input_q.shape)
        # print datatype of input_q
        #input_q = input_q.float()
        #print(input_q.dtype)
        # x_q = self.seq2vec(input_q)
        x_q = self.dropoutV(input_q)
        x_q = self.linear_q(x_q)
        x_q = nn.Tanh()(x_q)
        
        x = torch.mul(x_v, x_q)
        x = torch.squeeze(x, 1)
        x = nn.Tanh()(x)
        x = self.dropoutF(x)
        x = self.linear_classif1(x)
        x = nn.Tanh()(x)
        x = self.dropoutF(x)
        x = self.linear_classif2(x)
        # print("Shape of x_v:", x_v.shape)
        # print("Shape of x_q:", x_q.shape)
        # print("Shape of x after mul:", x.shape)
        # print("Shape of output:", x.shape)
        return x
        