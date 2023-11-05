import torch
import torch.nn as nn

VISUAL_OUT_VIT = 1280
QUESTION_OUT = 2400
FUSION_IN = 1200
FUSION_HIDDEN = 256
DROPOUT_V = 0.5
DROPOUT_Q = 0.5
DROPOUT_F = 0.5

question_type_to_idx = {
    "presence": 0,
    "comp": 1,
    "area": 2,
    "cound": 3,
}

# Mapping question types to number of unique answers
question_type_to_num_answers = {
    0: 2,
    1: 2,
    2: 4,
    3: 89
}

class MultiTaskVQAModel(nn.Module):
    def __init__(self, question_type_to_num_answers):
        super(MultiTaskVQAModel, self).__init__()

        self.dropoutV = nn.Dropout(DROPOUT_V)
        self.dropoutQ = nn.Dropout(DROPOUT_Q)
        self.dropoutF = nn.Dropout(DROPOUT_F)
        
        self.linear_q = nn.Linear(QUESTION_OUT, FUSION_IN)
        self.linear_v = nn.Linear(VISUAL_OUT_VIT, FUSION_IN)

        self.tanh = nn.Tanh()

        self.question_type_to_num_answers = question_type_to_num_answers
        self.total_num_classes = max(question_type_to_num_answers.values())

        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(FUSION_IN, FUSION_HIDDEN),
                self.tanh,
                nn.Dropout(DROPOUT_F),
                nn.Linear(FUSION_HIDDEN, num_answers)
            ) for num_answers in question_type_to_num_answers.values()
        ])

    def forward(self, input_v, input_q, question_type):
        x_v = self.linear_v(input_v)
        x_v = self.tanh(x_v)

        x_q = self.dropoutQ(input_q)
        x_q = self.linear_q(x_q)
        x_q = self.tanh(x_q)
        
        x = x_v * x_q
        x = self.tanh(x)
        x = self.dropoutF(x)

        # Choose the right classifier based on question type
        x = self.classifiers[question_type](x)

        # Projection to uniform output size
        output = torch.zeros(x.size(0), self.total_num_classes).to(x.device)
        num_answers = self.question_type_to_num_answers[question_type]
        output[:, :num_answers] = x

        return output
