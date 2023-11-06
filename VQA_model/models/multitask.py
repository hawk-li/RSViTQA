import torch
import torch.nn as nn

VISUAL_OUT_VIT = 768
QUESTION_OUT = 2400
FUSION_IN = 1200
FUSION_HIDDEN = 256
DROPOUT_V = 0.5
DROPOUT_Q = 0.5
DROPOUT_F = 0.5

# used in dataset, only for reference
        # question_type_to_idx = {
        #     "presence": 0,
        #     "comp": 1,
        #     "area": 2,
        #     "count": 3,
        # }


# question_type_to_indices = {
#             "presence": [0, 1],
#             "comp": [0, 1],
#             "area": list(range(2, 6)),
#             "count": list(range(6, 95))
#         }


class MultiTaskVQAModel(nn.Module):
    def __init__(self):
        super(MultiTaskVQAModel, self).__init__()

        # Mapping question types to number of unique answers
        question_type_to_num_answers = {
            0: 2,
            1: 2,
            2: 4,
            3: 89
        }

        self.dropoutV = nn.Dropout(DROPOUT_V)
        self.dropoutQ = nn.Dropout(DROPOUT_Q)
        self.dropoutF = nn.Dropout(DROPOUT_F)
        
        self.linear_q = nn.Linear(QUESTION_OUT, FUSION_IN)
        self.linear_v = nn.Linear(VISUAL_OUT_VIT, FUSION_IN)

        self.tanh = nn.Tanh()

        self.question_type_to_num_answers = question_type_to_num_answers
        self.total_num_classes = 95

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
        # Handle batch of question types
        output = torch.zeros((question_type.shape[0], self.total_num_classes), device=question_type.device)

        for qt in self.question_type_to_num_answers.keys():
            mask = question_type == qt
            x_masked = x[mask]
            classifier_output = self.classifiers[qt](x_masked)
            #print(f"qt: {qt}, classifier_output: {classifier_output.shape}")
            output[mask] = self.get_final_prediction(classifier_output, qt, self.total_num_classes)
        #print(f"output: {output.shape}")
        return output
    
    def get_final_prediction(self, pred, question_type, num_classes):
        question_type_to_indices = {
            0: [0, 1],
            1: [0, 1],
            2: list(range(2, 6)),
            3: list(range(6, 95))
        }
        final_pred = torch.zeros((pred.shape[0], num_classes), device=pred.device)
        indices = question_type_to_indices[question_type]
        
        # Assign the predictions to the relevant indices.
        final_pred[:, indices] = pred
        return final_pred
