#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sylvain
"""

# Classe définissant le jeu de donnée VQA au format pytorch

import os.path
import json
import random

import numpy as np
from skimage import io

from torch.utils.data import Dataset
import torchvision.transforms as T

RANDOM_SEED = 42

class VQALoader(Dataset):
    def __init__(self, imgFolder, images_file, questions_file, answers_file, train=True, ratio_images_to_use = 1, transform=None, patch_size=512):
        self.transform = transform
        self.train = train

        self.words = {
            "no": 0,
            "yes": 1,
            "0m2": 2,
            "between 0m2 and 10m2": 3,
            "between 10m2 and 100m2": 4,
            "between 100m2 and 1000m2": 5,
            "more than 1000m2": 6,
            "0": 7,
            "1": 8,
            "2": 9,
            "3": 10,
            "4": 11,
            "5": 12,
            "6": 13,
            "7": 14,
            "8": 15,
            "9": 16,
            "10": 17,
            "11": 18,
            "12": 19,
            "13": 20,
            "14": 21,
            "15": 22,
            "16": 23,
            "17": 24,
            "18": 25,
            "19": 26,
            "20": 27,
            "21": 28,
            "22": 29,
            "23": 30,
            "24": 31,
            "25": 32,
            "26": 33,
            "27": 34,
            "28": 35,
            "29": 36,
            "30": 37,
            "31": 38,
            "32": 39,
            "33": 40,
            "34": 41,
            "35": 42,
            "36": 43,
            "37": 44,
            "38": 45,
            "39": 46,
            "40": 47,
            "41": 48,
            "42": 49,
            "43": 50,
            "44": 51,
            "45": 52,
            "46": 53,
            "47": 54,
            "48": 55,
            "49": 56,
            "50": 57,
            "51": 58,
            "52": 59,
            "53": 60,
            "54": 61,
            "55": 62,
            "56": 63,
            "57": 64,
            "58": 65,
            "59": 66,
            "60": 67,
            "61": 68,
            "62": 69,
            "63": 70,
            "64": 71,
            "65": 72,
            "66": 73,
            "67": 74,
            "68": 75,
            "69": 76,
            "70": 77,
            "71": 78,
            "72": 79,
            "73": 80,
            "74": 81,
            "75": 82,
            "76": 83,
            "77": 84,
            "78": 85,
            "79": 86,
            "80": 87,
            "81": 88,
            "82": 89,
            "83": 90,
            "84": 91,
            "85": 92,
            "86": 93,
            "89": 94,
        }
        
        with open(questions_file) as json_data:
            self.questionsJSON = json.load(json_data)
            
        with open(answers_file) as json_data:
            self.answersJSON = json.load(json_data)
            
        with open(images_file) as json_data:
            self.imagesJSON = json.load(json_data)
        
        images = [img['id'] for img in self.imagesJSON['images'] if img['active']]
        
        self.len = 0
        for image in images:
            self.len += len(self.imagesJSON['images'][image]['questions_ids'])
        self.images_questions_answers = [[None] * 4] * self.len
        
        index = 0
        for i, image in enumerate(images):
            for questionid in self.imagesJSON['images'][image]['questions_ids']:
                question = self.questionsJSON['questions'][questionid]
            
                question_str = question["question"]
                type_str = question["type"]
                answer_str = self.answersJSON['answers'][question["answers_ids"][0]]['answer']
                token = answer_str#.split()

                if token[-2:] == 'm2':
                        num = int(token[:-2])
                        if num > 0 and num <= 10:
                            answer_str = "between 0m2 and 10m2"
                        if num > 10 and num <= 100:
                            answer_str = "between 10m2 and 100m2"
                        if num > 100 and num <= 1000:
                            answer_str = "between 100m2 and 1000m2"
                        if num > 1000:
                            answer_str = "more than 1000m2"
                        else:
                            answer_str = "0m2"
            
                self.images_questions_answers[index] = [question_str, self.words[answer_str], i, type_str]
                index += 1
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        question_type_to_idx = {
            "count": 3,
            "presence": 0,
            "area": 2,
            "comp": 1,
        }
        question = self.images_questions_answers[idx]
        work_dir = os.getcwd()
        data_path = work_dir + '/data'
        images_path = os.path.join(data_path + '/images/')
        img = io.imread(os.path.join(images_path, str(question[2])+'.tif'))
        if self.transform:
            imgT = self.transform(img.copy())

        return question[0], question[1], imgT, question_type_to_idx[question[3]], question[3]
