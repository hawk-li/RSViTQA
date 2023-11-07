import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, feature_size):
        super(SelfAttention, self).__init__()
        # Initialize weight matrices for Q, K, and V
        self.W_q = nn.Linear(feature_size, feature_size, bias=False)
        self.W_k = nn.Linear(feature_size, feature_size, bias=False)
        self.W_v = nn.Linear(feature_size, feature_size, bias=False)
        
        # Save the feature size to scale the dot product attention scores
        self.d_k = feature_size

    def forward(self, x):
        # x is assumed to be of shape [batch_size, seq_len, feature_size]
        
        # Compute Q, K, V matrices
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Calculate attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))

        # Scale scores
        scaled_attention_scores = attention_scores / torch.sqrt(torch.tensor(self.d_k).float())

        # Apply softmax on the last dimension to obtain the attention probabilities
        attention_weights = F.softmax(scaled_attention_scores, dim=-1)

        # Weight the values by the attention weights
        weighted_values = torch.matmul(attention_weights, V)

        return weighted_values, attention_weights
    
class CrossAttention(nn.Module):
    def __init__(self, feature_size):
        super(CrossAttention, self).__init__()
        # Initialize weight matrices for Q, K, and V
        self.W_q = nn.Linear(feature_size, feature_size, bias=False)
        self.W_k = nn.Linear(feature_size, feature_size, bias=False)
        self.W_v = nn.Linear(feature_size, feature_size, bias=False)

        # Save the feature size to scale the dot product attention scores
        self.d_k = feature_size

    def forward(self, x, y):
        # x is assumed to be of shape [batch_size, seq_len_x, feature_size]
        # y is assumed to be of shape [batch_size, seq_len_y, feature_size]
        
        # Compute Q, K, V matrices
        Q = self.W_q(x)
        K = self.W_k(y)
        V = self.W_v(y)

        # Calculate attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))

        # Scale scores
        scaled_attention_scores = attention_scores / torch.sqrt(torch.tensor(self.d_k).float())

        # Apply softmax on the last dimension to obtain the attention probabilities
        attention_weights = F.softmax(scaled_attention_scores, dim=-1)

        # Weight the values by the attention weights
        weighted_values = torch.matmul(attention_weights, V)

        return weighted_values, attention_weights