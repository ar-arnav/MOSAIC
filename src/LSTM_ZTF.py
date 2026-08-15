import os
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence


class LSTM_ZTF(nn.Module):
    def __init__(self, 
                 input_size: int = 4, 
                 hidden_size: int = 64, 
                 num_layers: int = 2, 
                 output_size: int = 1, 
                 dropout: float = 0.2
                 ):

        super().__init__()

        self.lstm = nn.LSTM(input_size = input_size, 
                            hidden_size = hidden_size, 
                            num_layers = num_layers, 
                            batch_first=True, 
                            dropout=dropout if num_layers>1 else 0
                            )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        out, (h_n, c_n) = self.lstm(packed_x)
        last_hidden = h_n[-1]

        logits = self.fc(last_hidden)
        return logits.squeeze(-1)
