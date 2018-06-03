import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from tqdm import tqdm

class Decoder(nn.Module):
    def __init__(self,
                 hidden_size=100,
                 output_size=20000,
                 device='cpu'):
        super(Decoder, self).__init__()
        # self.embedding_size = hidden_size # embedding_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.device = device

    def forward(self, input, hidden=None, batch_size=1):
        output = self.embed(input).view(batch_size, 1, -1)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output[:, 0]))
        return output, hidden

class TreeDecoder(nn.Module):
    def __init__(self,
                 hidden_size=100,
                 output_size=20000,
                 device='cpu'):
        super(TreeDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(output_size, hidden_size).cuda(device)
        # double the size of the LSTM (we append the parent <n> hidden state to input)
        self.lstm = nn.LSTM(hidden_size*2, hidden_size, batch_first=True).cuda(device)
        self.out = nn.Linear(hidden_size, output_size).cuda(device)
        self.softmax = nn.LogSoftmax(dim=1)
        self.device = device

    def forward(self, input, hidden=None, parent=None):
        output = self.embed(input).view(1, 1, -1)
        output = torch.cat([output, parent], dim=2)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output[:, 0]))
        return output, hidden
