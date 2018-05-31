import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Tree2Tree:
    def __init__(self, input_size=None, hidden_size=None, output_size=None,
                 optimizer=optim.Adam, criterion=nn.NLLLoss, lr=0.0001,
                 src_vocab=None, tar_vocab=None,
                 sess='', device='cpu'):
        raise NotImplementedError()
