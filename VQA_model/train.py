#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sylvain
"""

# Script principal d'apprentissage

import matplotlib
matplotlib.use('Agg')

from models import model
import VQALoader
import VocabEncoder
import torchvision.transforms as T
import torch
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data

import pickle
import os
import datetime
from shutil import copyfile


def train(model, train_dataset, validate_dataset, batch_size, num_epochs, learning_rate, modeltype, Dataset='HR'):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,RSVQA.parameters()), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()#weight=weights)
        
    trainLoss = []
    valLoss = []

    accPerQuestionType = {'area': [], 'presence': [], 'count': [], 'comp': []}

    OA = []
    AA = []
    for epoch in range(num_epochs):
        RSVQA.train()
        runningLoss = 0
        print('start training')
        for i, data in enumerate(train_loader, 0):
            if i % (len(train_loader)//10) == (len(train_loader)//10 - 1):
                print('Training progress: %d %%' % (100*i/len(train_loader)))
            question, answer, image, _ = data
            #question = torch.squeeze(question, 1).to("cuda")
            question = question.to("cuda")
            print(answer.shape)
            answer = answer.to("cuda").resize_(question.shape[0])
            print(question.shape, answer.shape)
            image = torch.squeeze(image, 1).to("cuda")

            pred = RSVQA(image,question)
            #print(pred.shape, answer.shape)
            loss = criterion(pred, answer)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            runningLoss += loss.cpu().item() * question.shape[0]
        
            
        trainLoss.append(runningLoss / len(train_dataset))
        print('epoch #%d loss: %.3f' % (epoch, trainLoss[epoch]))
                
        torch.save(RSVQA.state_dict(), 'RSVQA_model_epoch_' + str(epoch) + '.pth')

        with torch.no_grad():
            RSVQA.eval()
            runningLoss = 0
 
            countQuestionType = {'presence': 0, 'count': 0, 'comp': 0, 'area': 0}
            rightAnswerByQuestionType = {'presence': 0, 'count': 0, 'comp': 0, 'area': 0}

            count_q = 0
            for i, data in enumerate(validate_loader, 0):
                if i % 1000 == 999:
                    print(i/len(validate_loader))
                question, answer, image, type_str, image_original = data
                question = Variable(question.long()).cuda()
                answer = Variable(answer.long()).cuda().resize_(question.shape[0])
                image = Variable(image.float()).cuda()
                if modeltype == 'MCB':
                    pred, att_map = RSVQA(image,question)
                else:
                    pred = RSVQA(image,question)
                loss = criterion(pred, answer)
                runningLoss += loss.cpu().item() * question.shape[0]
                
                answer = answer.cpu().numpy()
                pred = np.argmax(pred.cpu().detach().numpy(), axis=1)
                for j in range(answer.shape[0]):
                    countQuestionType[type_str[j]] += 1
                    if answer[j] == pred[j]:
                        rightAnswerByQuestionType[type_str[j]] += 1
                        
            valLoss.append(runningLoss / len(validate_dataset))
            print('epoch #%d val loss: %.3f' % (epoch, valLoss[epoch]))
        
            numQuestions = 0
            numRightQuestions = 0
            currentAA = 0
            for type_str in countQuestionType.keys():
                if countQuestionType[type_str] > 0:
                    accPerQuestionType[type_str].append(rightAnswerByQuestionType[type_str] * 1.0 / countQuestionType[type_str])
                numQuestions += countQuestionType[type_str]
                numRightQuestions += rightAnswerByQuestionType[type_str]
                currentAA += accPerQuestionType[type_str][epoch]
                
            OA.append(numRightQuestions *1.0 / numQuestions)
            AA.append(currentAA * 1.0 / 4)


if __name__ == '__main__':
    disable_log = False
    batch_size = 200
    num_epochs = 35
    learning_rate = 0.00001
    ratio_images_to_use = 1
    modeltype = 'Simple'
    Dataset = 'HR'

    work_dir = os.getcwd()
    data_path = work_dir + '/data'
    images_path = data_path + '/image_representations'
    questions_path = data_path + '/text_representations'
    questions_train_path = questions_path + '/train'
    questions_val_path = questions_path + '/val'

    patch_size = 512   

    train_dataset = VQALoader.VQADataset(questions_train_path, images_path)
    validate_dataset = VQALoader.VQADataset(questions_val_path, images_path)
    
    
    RSVQA = model.VQAModel(input_size = patch_size).cuda()
    RSVQA = train(RSVQA, train_dataset, validate_dataset, batch_size, num_epochs, learning_rate, modeltype, Dataset)
    
    
