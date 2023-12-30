import torch
import torch.nn as nn
from models.attention import SelfAttentionQuestion, CrossAttention, SelfAttentionImage
from torchvision import models

## Import HuggingFace libraries
from transformers import BertTokenizer, BertModel

VISUAL_OUT = 768
QUESTION_OUT = 768
HIDDEN_DIMENSION_ATTENTION = 512
HIDDEN_DIMENSION_CROSS = 5000
FUSION_IN = 800
FUSION_HIDDEN = 256
FUSION_HIDDEN_2 = 64
DROPOUT_V = 0.5
DROPOUT_Q = 0.5
DROPOUT_F = 0.5

from block import fusions

# used in dataset, only for reference
# question_type_to_idx = {
#     "presence": 0,
#     "comp": 0,
#     "area": 1,
#     "count": 2,
# }


# question_type_to_indices = {
#             "presence": [0, 1],
#             "area": list(range(2, 6)),
#             "count": list(range(6, 95))
#         }

class CustomFusionModule(nn.Module):
    def __init__(self, fusion_in, fusion_hidden, num_answers):
        super(CustomFusionModule, self).__init__()

        self.dropoutV = nn.Dropout(DROPOUT_V)
        self.dropoutQ = nn.Dropout(DROPOUT_Q)
        self.dropoutF = nn.Dropout(DROPOUT_F)

        ## Attention Modules
        self.selfattention_q = SelfAttentionQuestion(QUESTION_OUT, HIDDEN_DIMENSION_ATTENTION, 1)
        self.crossattention = CrossAttention(HIDDEN_DIMENSION_CROSS, QUESTION_OUT, VISUAL_OUT)
        self.selfattention_v = SelfAttentionImage(HIDDEN_DIMENSION_CROSS, HIDDEN_DIMENSION_ATTENTION, 1)
        
        self.linear_q = nn.Linear(QUESTION_OUT, FUSION_IN)
        self.linear_v = nn.Linear(VISUAL_OUT, FUSION_IN)

        self.dropout = nn.Dropout(DROPOUT_F)

        self.fusion = fusions.Mutan([FUSION_IN, FUSION_IN], FUSION_IN)
        
        self.linear1 = nn.Linear(fusion_in, fusion_hidden)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(fusion_hidden, FUSION_HIDDEN_2)
        self.linear3 = nn.Linear(FUSION_HIDDEN_2, num_answers)

    def forward(self, input_v, input_q):

        ## Dropouts
        input_q = self.dropoutQ(input_q)
        input_v = self.dropoutV(input_v)
        
        ## Self-Attention for Question
        q = self.selfattention_q(input_q)
        c = self.crossattention(q, input_v)
        v = self.selfattention_v(c, input_v)
    
        ## Prepare fusion
        q = self.linear_q(q)
        q = nn.Tanh()(q)
        v = self.linear_v(v)
        v = nn.Tanh()(v)

        ## Fusion & Classification 
        # when using multiplication        
        #x = torch.mul(v, q)
        #x = nn.Tanh()(x)
        # when using mutan
        x = self.fusion([v, q])
        
        # when using concatenation
        #x = torch.cat((v, q), dim=1)
        #x = torch.squeeze(x, 1)
        
        x = self.dropoutF(x)
        x = self.linear1(x)
        x = nn.Tanh()(x)
        x = self.dropoutF(x)
        x = self.linear2(x)
        x = nn.Tanh()(x)
        x = self.dropoutF(x)
        x = self.linear3(x)
        return x


class MultiTaskVQAModel(nn.Module):
    def __init__(self):
        super(MultiTaskVQAModel, self).__init__()

        # pretrained vit-b
        self.vit = models.vit_b_16(weights="DEFAULT")

        # pretrained bert
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Mapping question types to number of unique answers
        question_type_to_num_answers = {
            0: 2,
            1: 2,
            2: 5,
            3: 88
        }

        #self.fusion = fusions.Mutan([FUSION_IN, FUSION_IN], FUSION_IN)

        self.question_type_to_num_answers = question_type_to_num_answers
        self.total_num_classes = 95

        # self.selfattention = attention.SelfAttention(FUSION_IN)
        # self.crossattention = attention.CrossAttention(FUSION_IN)

        self.classifiers = nn.ModuleList([
            CustomFusionModule(FUSION_IN, FUSION_HIDDEN, num_answers) 
            for num_answers in question_type_to_num_answers.values()
        ])

    def shared_parameters(self):
        # Return all parameters that are not part of the classifier heads
        for name, param in self.named_parameters():
            if not any(name.startswith(f'classifiers.{i}') for i in range(len(self.classifiers))):
                yield param

    def forward(self, input_v, input_q, question_type):

        input_v = self.vit._process_input(input_v)
        n = input_v.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        input_v = torch.cat([batch_class_token, input_v], dim=1)

        input_v = self.vit.encoder(input_v)

        ## Question Features
        #input_q = self.tokenizer.encode_plus(input_q, pad_to_multiple_of=35, add_special_tokens=True, return_attention_mask=True, padding=True, return_tensors="pt")
        input_q = self.bert(**input_q)
        input_q = input_q.last_hidden_state.squeeze(0)

        # Initialize a tensor to hold the final predictions for the entire batch
        batch_size = input_v.size(0)
        final_output = torch.zeros(batch_size, self.total_num_classes, device=input_q.device)

        for qt, classifier in zip(self.question_type_to_num_answers.keys(), self.classifiers):
            mask = (question_type == qt)
            v_masked = input_v[mask]
            q_masked = input_q[mask]
            
            if v_masked.size(0) > 0:  # Check if there are any items of this question type
                classifier_output = classifier(v_masked, q_masked)
                output_masked = self.get_final_prediction(pred=classifier_output, question_type=qt, num_classes=self.total_num_classes)
                
                # Place the result back in the correct positions of the final_output tensor
                final_output[mask] = output_masked

        return final_output
    
    def get_final_prediction(self, pred, question_type, num_classes):
        question_type_to_indices = {
            0: [0, 1],
            1: [0, 1],
            2: list(range(2, 7)),
            3: list(range(7, 95))
        }
        final_pred = torch.zeros((pred.shape[0], num_classes), device=pred.device)
        indices = question_type_to_indices[question_type]
        
        # Assign the predictions to the relevant indices.
        final_pred[:, indices] = pred
        return final_pred
