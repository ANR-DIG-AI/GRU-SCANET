import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(x.size(0))
        # print(len(x))
        # print(len(self.pe[:x.size(0), :]))
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)