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

class ChildSumTreeLSTM(nn.Module):
    def __init__(self):
        super(ChildSumTreeLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.iou_x = nn.Linear(self.input_size, 3*self.hidden_size)
        self.iou_h = nn.Linear(self.input_size, 3*self.hidden_size)
        self.f_x = nn.Linear(self.input_size, self.hidden_size)
        self.f_h = nn.Linear(self.hidden_size, self.hidden_size)

    def node_forward(self, inputs, child_c, child_h):
        # Eq 2: h_hat_j = \sum(h_k)
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        # Eq 3: i_j = sigmoid(W x_j + U h_hat_j + b)
        # Eq 5: o_j = sigmoid(W x_j + U h_hat_j + b)
        # Eq 6: u_j =    tanh(W x_j + U h_hat_j + b)
        iou = self.iou_x(inputs) + self.iou_h(child_h_sum)
        i, o, u = torch.split(iou, int(iou.size(1) / 3), dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        # Eq 4: f_jk = sigmoid(W x_j + U h_k + b)
        f = F.sigmoid(self.f_h(child_h) + self.f_x(inputs).repeat(len(child_h), 1))

        # Subeq 7: (f_jk x c_k)
        f_c = torch.mul(f, child_c)

        # Eq 7: c_j = i_j x u_j + \sum(f_jk x c_k)
        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)

        # Eq 8: h_j = o_j x tanh(c_j)
        h = torch.mul(o, F.tanh(c))
        return c, h

    def forward(self, tree, inputs):
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs)

        if tree.num_children == 0:
            child_c = inputs[0].detach().new(1, self.hidden_size).fill_(0.).requires_grad_()
            child_h = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
        else:
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)
        return tree.state

class TreeEncoder(nn.Module):
    def __init__(self, vocab_size=0, input_size=0, mem_size=0, output_size=0,
                 optimizer=optim.Adam, criterion=nn.NLLLoss, lr=0.0001,
                 sess='', device='cpu'):
        super(TreeEncoder, self).__init__()
        self.tree_module = ChildSumTreeLSTM(input_size=input_size,
                                            mem_size=mem_size,
                                            criterion=criterion,
                                            device=device)
        self.embed = nn.Embedding(input_size, hidden_size)
        self.childsumtree = ChildSumTreeLSTM(input_size, hidden)

    def forward(self, tree, inputs, training = False):
        """
        TreeLSTMSentiment forward function
        :param tree:
        :param inputs: (sentence_length, 1, 300)
        :param training:
        :return:
        """
        tree_state, loss = self.tree_module(tree, inputs, training)
        output = tree.output
        return output, loss
