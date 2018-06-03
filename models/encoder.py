import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from tqdm import tqdm

class Encoder(nn.Module):
    def __init__(self,
                 input_size=20000,
                 hidden_size=100,
                 device='cpu'):
        super(Encoder, self).__init__()
        self.embedding_size = hidden_size # embedding_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.device = device

    def forward(self, text, hidden=None, batch_size=1):
        # sent = torch.tensor(sent, dtype=torch.long, device=self.device)
        inputs = self.embed(text).view(batch_size, len(text[0]), -1)
        output, hidden = self.lstm(inputs, hidden)
        return output, hidden

class TreeLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True):
        super(TreeLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(input_size, hidden_size)
        self.iou_x = nn.Linear(self.input_size, 3*self.hidden_size)
        self.iou_h = nn.Linear(self.hidden_size, 3*self.hidden_size)
        self.f_x = nn.Linear(self.input_size, self.hidden_size)
        self.f_h = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, input, hidden):
        children_c, children_h = hidden

        iou = self.iou_x(input) + self.iou_h(torch.sum(children_h, dim=0))
        i, o, u = torch.split(iou, self.hidden_size)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        # initialized to zero in case no children
        f = []
        for c_k, h_k in zip(children_c, children_h):
            f_k = F.sigmoid(self.f_x(input) + self.f_h(h_k))
            f.append(torch.mul(c_k, f_k))
        f = torch.stack(f)

        c = torch.mul(i, u) + torch.sum(f, dim=0)
        h = torch.mul(o, F.tanh(c))
        return o, (c, h)

class TreeEncoder(nn.Module):
    def __init__(self,
                 input_size=20000,
                 hidden_size=20,
                 device='cpu'):
        super(TreeEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(input_size, hidden_size)
        self.lstm = TreeLSTMCell(hidden_size, hidden_size, batch_first=True)
        self.device = device

    def forward(self, tree, batch_size=1):
        children_c = []
        children_h = []

        if len(tree.children) == 0:
            # base case:
            children_c.append(torch.zeros(self.hidden_size,
                                          device=self.device))
            children_h.append(torch.zeros(self.hidden_size,
                                          device=self.device))
        else:
            # recursive case
            for child in tree.children:
                _, (child_c, child_h) = self.forward(child)
                # child_c, child_h = child_hidden
                children_c.append(child_c.view(-1))
                children_h.append(child_h.view(-1))

        output = self.embed(tree.input)
        hidden = torch.stack(children_c), torch.stack(children_h)
        output, hidden = self.lstm(output, hidden=hidden)
        return output.view(1, 1, -1), (hidden[0].view(1, 1, -1), hidden[1].view(1, 1, -1))
