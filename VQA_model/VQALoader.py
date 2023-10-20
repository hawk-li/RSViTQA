#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sylvain
"""

# Classe définissant le jeu de donnée VQA au format pytorch


import json
import random

import numpy as np
# from skimage import io

from torch.utils.data import Dataset
# import torchvision.transforms as T

import glob
import os
import torch

RANDOM_SEED = 42


class VQADataset(Dataset):
    
    def __init__(self, 
                 # imgFolder, 
                 # images_file, 
                 # questions_file, 
                 # answers_file, 
                 # encoder_questions, 
                 # encoder_answers, 
                 # train=True, 
                 # ratio_images_to_use = 1, 
                 # transform=None, 
                 # patch_size=512,
                 textual_path,
                 visual_path
                  ):
        
        # self.transform = transform
        # self.encoder_questions = encoder_questions
        # self.encoder_answers = encoder_answers
        # self.train = train
        
        # vocab = self.encoder_questions.words
        # self.relationalWords = [vocab['top'], vocab['bottom'], vocab['right'], vocab['left']]
        
        # with open(questions_file) as json_data:
        #     self.questionsJSON = json.load(json_data)
            
        # with open(answers_file) as json_data:
        #     self.answersJSON = json.load(json_data)
            
        # with open(images_file) as json_data:
        #     self.imagesJSON = json.load(json_data)
        
        # images = [img['id'] for img in self.imagesJSON['images'] if img['active']]
        # images = images[:int(len(images)*ratio_images_to_use)]
        # self.images = np.empty((len(images), patch_size, patch_size, 3))
        
        # self.len = 0
        # for image in images:
        # self.len += len(self.imagesJSON['images'][image]['questions_ids'])
        # self.images_questions_answers = [[None] * 4] * self.len
        
        # index = 0
        # for i, image in enumerate(images):
        #     img = io.imread(os.path.join(imgFolder, str(image)+'.tif'))
        #     self.images[i, :, :, :] = img
        #     for questionid in self.imagesJSON['images'][image]['questions_ids']:
        #         question = self.questionsJSON['questions'][questionid]
            
        #         question_str = question["question"]
        #         type_str = question["type"]
        #         answer_str = self.answersJSON['answers'][question["answers_ids"][0]]['answer']
            
        #         self.images_questions_answers[index] = [self.encoder_questions.encode(question_str), self.encoder_answers.encode(answer_str), i, type_str]
        #         index += 1

        self.textual_path = textual_path
        self.visual_path = visual_path

        files = glob.glob(os.path.join(textual_path, "*.pt"))
        self.textual_ids = [int(os.path.splitext(os.path.basename(file))[0]) for file in files]
        self.textual_ids.sort()
    
        self.len = len(self.textual_ids)

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        # question = self.images_questions_answers[idx]
        # img = self.images[question[2],:,:,:]
        # if self.train and not self.relationalWords[0] in question[0] and not self.relationalWords[1] in question[0] and not self.relationalWords[2] in question[0] and not self.relationalWords[3] in question[0]:
        #     if random.random() < .5:
        #         img = np.flip(img, axis = 0)
        #     if random.random() < .5:
        #         img = np.flip(img, axis = 1)
        #     if random.random() < .5:
        #         img = np.rot90(img, k=1)
        #     if random.random() < .5:
        #         img = np.rot90(img, k=3)
        # if self.transform:
        #     imgT = self.transform(img.copy())
        # if self.train:
        #     return np.array(question[0], dtype='int16'), np.array(question[1], dtype='int16'), imgT, question[3]
        # else:
        #     return np.array(question[0], dtype='int16'), np.array(question[1], dtype='int16'), imgT, question[3], T.ToTensor()(img / 255)   

        id = self.textual_ids[idx]
        text = torch.load(os.path.join(self.textual_path, str(id) + ".pt"))
        question = text["question"]
        answer = text["answer"]
        question_type = text["question_type"]
        image_id = int(text["image_id"])
        
        image = torch.load(os.path.join(self.visual_path, str(image_id) + ".pt"))

        return question, answer, image, question_type