#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sylvain
"""

# Calcul des statistiques sur un jeu de test

from functools import partial
import textwrap
import VocabEncoder as VocabEncoder
import VQADataset_FE as VQADataset
from models import multitask_attention as model
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torchvision.transforms as T
from torch.autograd import Variable
from skimage import io
import numpy as np
import pickle
import os
from tqdm import tqdm
import torch.utils.data.dataloader as dataloader
from transformers import AutoTokenizer

def vqa_collate_fn(batch, tokenizer):
    # Separate the list of tuples into individual lists
    questions, answers, images, question_types, question_types_str = zip(*batch)

    # Tokenize each question
    questions_encoded = [tokenizer.encode_plus(question, pad_to_multiple_of=60, add_special_tokens=True, return_attention_mask=True, padding=True, return_tensors="pt") for question in questions]

    # Convert list of tokenized outputs to a single dictionary
    input_ids = torch.cat([qe["input_ids"] for qe in questions_encoded], dim=0)
    attention_masks = torch.cat([qe["attention_mask"] for qe in questions_encoded], dim=0)

    # Token type IDs if necessary
    if 'token_type_ids' in questions_encoded[0]:
        token_type_ids = torch.cat([qe["token_type_ids"] for qe in questions_encoded], dim=0)
    else:
        token_type_ids = None

    # Combine into a single dictionary
    questions_batch = {
        'input_ids': input_ids,
        'attention_mask': attention_masks
    }
    if token_type_ids is not None:
        questions_batch['token_type_ids'] = token_type_ids

    # Convert answers, images, and question types to tensors
    answers_batch = torch.stack([torch.tensor(answer) for answer in answers]) 
    images_batch = torch.stack(images)  
    question_types_batch = torch.tensor(question_types)

    return questions_batch, answers_batch, images_batch, question_types_batch, question_types_str

def do_confusion_matrix(all_mat, old_vocab, new_vocab, dataset):
    print(new_vocab)
    new_mat = np.zeros((len(new_vocab), len(new_vocab)))
    for i in range(1,all_mat.shape[0]):
        answer = old_vocab[i]
        new_i = new_vocab.index(answer)
        for j in range(1,all_mat.shape[1]):
            answer = old_vocab[j]
            new_j = new_vocab.index(answer)
            new_mat[new_i, new_j] = all_mat[i, j]

    if len(old_vocab) > 20:#HR
        new_mat = new_mat[0:18,0:18]
        new_vocab = new_vocab[0:18]
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    cax = ax.matshow(np.log(new_mat+1), cmap="YlGn")
    #plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + new_vocab)
    ax.set_yticklabels([''] + new_vocab)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    fig.savefig('confusion_matrix_' + dataset + '.svg')
    #plt.close()

        

def get_vocab():
    work_dir = os.getcwd()
    data_path = work_dir + '/data/text'
    allanswersJSON = os.path.join(data_path, 'USGSanswers.json')
    encoder_answers = VocabEncoder.VocabEncoder(allanswersJSON, questions=False, range_numbers = False)
        
    return encoder_answers.getVocab()

def load_model(experiment, epoch, patch_size=512):
    weight_file = experiment + '_' + str(epoch) + '.pth'
    work_dir = os.getcwd()
    path = os.path.join(work_dir, 'outputs', weight_file)
    network = model.MultiTaskVQAModel().cuda()#input_size = patch_size).cuda()
    state = network.state_dict()
    state.update(torch.load(path))
    network.load_state_dict(state, strict=False)
    network.eval().cuda()
    return network

def get_image(image_id):
    work_dir = os.getcwd()
    images_path = os.path.join(work_dir + "/data/images", str(int(image_id)) + '.png')
    image = io.imread(images_path)
    return image

def run(network, experiment, num_batches=-1):
    work_dir = os.getcwd()
    data_path = work_dir + '/data'
    images_path = os.path.join(data_path + '/images/')
    questionsvalJSON = os.path.join(data_path + '/text/USGS_split_test_questions.json')

    answersvalJSON = os.path.join(data_path + '/text/USGS_split_test_answers.json')

    imagesvalJSON = os.path.join(data_path + '/text/USGS_split_test_images.json')

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),            
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
      ])
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    test_dataset = VQADataset.VQALoader(images_path, imagesvalJSON, questionsvalJSON, answersvalJSON, train=False, ratio_images_to_use=1, transform=transform, patch_size = 1)
    test_loader = dataloader.DataLoader(test_dataset, batch_size=70, shuffle=False, persistent_workers=False, pin_memory=True, num_workers=8, collate_fn=partial(vqa_collate_fn, tokenizer=tokenizer))
    batch_size = 100
    patch_size = 512
    
    
    print ('---' + experiment + '---')
    countQuestionType = {'area': 0, 'presence': 0, 'count': 0, 'comp': 0}
    rightAnswerByQuestionType = {'area': 0, 'presence': 0, 'count': 0, 'comp': 0}
    encoder_answers = get_vocab()
    confusionMatrix = np.zeros((len(encoder_answers), len(encoder_answers)))
    progress_bar = tqdm(enumerate(test_loader, 0), total=len(test_loader))
    for i, data in progress_bar:
        if num_batches == 0:
            break
        num_batches -= 1
        question, answer, image, type_idx, type_str  = data
        question = {key: value.to("cuda") for key, value in question.items()}
        answer = answer.to("cuda")
        image = image.to("cuda")
        type_idx = type_idx.to("cuda")

        pred = network(image,question, type_idx)
        
        answer = answer.cpu().numpy()
        pred = np.argmax(pred.cpu().detach().numpy(), axis=1)


        for j in range(answer.shape[0]):
            countQuestionType[type_str[j]] += 1
            if answer[j] == pred[j]:
                rightAnswerByQuestionType[type_str[j]] += 1
            confusionMatrix[answer[j], pred[j]] += 1
    
    Accuracies = {'AA': 0}
    for type_str in countQuestionType.keys():
        Accuracies[type_str] = rightAnswerByQuestionType[type_str] * 1.0 / countQuestionType[type_str]
        Accuracies['AA'] += Accuracies[type_str] / len(countQuestionType.keys())
    Accuracies['OA'] = np.trace(confusionMatrix)/np.sum(confusionMatrix)
    
    print('- Accuracies')
    for type_str in countQuestionType.keys():
        print (' - ' + type_str + ': ' + str(Accuracies[type_str]))
    print('- AA: ' + str(Accuracies['AA']))
    print('- OA: ' + str(Accuracies['OA']))
    
    return Accuracies, confusionMatrix

if __name__ == '__main__':
    # relative path to the model folder
    expes = {
            'HR': ['finetune/RSVQA_model_epoch'],
    }
    work_dir = os.getcwd()
    data_path = work_dir + '/data'

    for dataset in expes.keys():
        acc = []
        mat = []
        for experiment_name in expes[dataset]:
            model_att = load_model(experiment_name, 3)
            tmp_acc, tmp_mat = run(model_att, experiment_name)

            acc.append(tmp_acc)
            mat.append(tmp_mat)
            
        print('--- Total (' + dataset + ') ---')
        print('- Accuracies')
        for type_str in tmp_acc.keys():
            all_acc = []
            for tmp_acc in acc:
                all_acc.append(tmp_acc[type_str])
            print(' - ' + type_str + ': ' + str(np.mean(all_acc)) + ' ( stddev = ' + str(np.std(all_acc)) + ')')