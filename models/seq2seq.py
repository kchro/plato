"""
overview:
implementation of the seq2seq model:
L-layer LSTM Encoder
L-layer LSTM Decoder
language => logical form
"""
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from utils import NLVocab, FOLVocab
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=100):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class Seq2Seq:
    def __init__(self, input_size=0, hidden_size=100, output_size=0, max_length=0, device='cpu', lr=0.01, criterion=criterion):
        self.encoder = EncoderRNN(input_size=input_size,
                                  hidden_size=hidden_size).to(device)
        self.decoder = AttnDecoderRNN(hidden_size=hidden_size,
                                      output_size=output_size,
                                      max_length=max_length).to(device)

        self.encoder_opt = optim.SGD(self.encoder.parameters(), lr=lr)
        self.decoder_opt = optim.SGD(self.decoder.parameters(), lr=lr)

        self.max_length = max_length
        self.device = device
        self.lr = lr
        self.criterion = criterion

    def _train(self, X, y):
        encoder_hidden = self.encoder.initHidden()

        self.encoder_opt.zero_grad()
        self.decoder_opt.zero_grad()

        input_length = X.size(0)
        target_length = y.size(0)

        encoder_outputs = torch.zeros(self.max_length, encoder.hidden_size, self.device=device)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == EOS_token:
                    break

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.item() / target_length

    def train(self, X_train, y_train,
              n_iters=10,
              batch_size=1,
              loss_update=10):

        total_loss = 0

        # TODO: make the optimizer a parameter

        n = 0
        for iter in tqdm(range(n_iters)):
            while n < len(X_train[:100]):
                # X_batch = torch.stack(X_train[n:n+batch_size])
                # y_batch = torch.stack(y_train[n:n+batch_size])
                X_batch = X_train[n]
                y_batch = y_train[n]

                loss = self._train(X_batch, y_batch)

                total_loss += loss

                if iter % loss_update == 0:
                    avg_loss = total_loss / float(loss_update)
                    print 'avg loss @ iteration %d: %0.2f' % (iter, avg_loss)
                    total_loss = 0

                n += batch_size
