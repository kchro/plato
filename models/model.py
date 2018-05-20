import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from tqdm import tqdm

class Encoder(nn.Module):
    def __init__(self,
                 input_size=20000,
                 # embedding_size=100,
                 hidden_size=100,
                 device='cpu'):
        super(Encoder, self).__init__()
        self.embedding_size = hidden_size # embedding_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.device = device

    def forward(self, sent, hidden=None):
        # sent = torch.tensor(sent, dtype=torch.long, device=self.device)
        inputs = self.embed(sent).view(len(sent), 1, -1)
        output, hidden = self.lstm(inputs, hidden)
        return output, hidden

class Decoder(nn.Module):
    def __init__(self,
                 # vocab_size=20000,
                 # embedding_size=100,
                 hidden_size=100,
                 output_size=20000,
                 device='cpu'):
        super(Decoder, self).__init__()
        # self.embedding_size = hidden_size # embedding_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.device = device

    def forward(self, input, hidden=None):
        output = self.embed(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

class Seq2Seq:
    def __init__(self, input_size=0, hidden_size=0, output_size=0,
                 optimizer=optim.Adam, criterion=nn.NLLLoss, lr=0.0001,
                 sess='', device='cpu'):
        self.encoder = Encoder(input_size, hidden_size, device)
        self.decoder = Decoder(hidden_size, output_size, device)

        self.encoder_opt = optimizer(self.encoder.parameters(), lr=lr)
        self.decoder_opt = optimizer(self.decoder.parameters(), lr=lr)
        self.criterion = criterion()
        self.sess = sess
        self.device = device

    def save(self, filename):
        torch.save(self.encoder.state_dict(), 'logs/sessions/enc_%s' % filename)
        torch.save(self.decoder.state_dict(), 'logs/sessions/dec_%s' % filename)

    def load(self, filename):
        self.encoder.load_state_dict(torch.load('logs/sessions/enc_%s' % filename))
        self.decoder.load_state_dict(torch.load('logs/sessions/dec_%s' % filename))

    def set_vocab(self, src_vocab, tar_vocab):
        self.src_vocab = src_vocab
        self.tar_vocab = tar_vocab

    def run_epoch(self, src_input, tar_output):
        self.encoder_opt.zero_grad()
        self.decoder_opt.zero_grad()

        # encode the source input
        encoder_output, encoder_hidden = self.encoder(src_input)

        SOS_token = self.tar_vocab.word_to_index['<S>']
        decoder_input = torch.LongTensor([[SOS_token]], device=self.device)
        decoder_hidden = encoder_hidden

        loss = 0

        # Teacher forcing: Feed the target as the next input
        tar_len = self.tar_vocab.max_len
        for di in range(tar_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                          decoder_hidden)
            loss += self.criterion(decoder_output, tar_output[di:di+1])
            decoder_input = tar_output[di] # Teacher forcing

        loss.backward()
        self.encoder_opt.step()
        self.decoder_opt.step()

        return loss.item() / tar_len

    def train(self, X_train, y_train, epochs=10, loss_update=10):
        cum_loss = 0
        history = {}
        losses = []

        def get_progress(num, den, length):
            if num == den-1:
                return '='*length
            arrow = int(float(num)/den*length)
            return '='*(arrow-1)+'>'+'.'*(20-arrow)

        for epoch in range(epochs):
            epoch_loss = 0

            print 'Epoch %d/%d' % (epoch, epochs)

            for i in range(len(X_train)):
                loss = self.run_epoch(X_train[i], y_train[i])
                cum_loss += loss
                epoch_loss += loss

                progress = get_progress(i, len(X_train), length=20)
                out = '%d/%d [%s] loss: %f' % (i, len(X_train), progress, epoch_loss)
                sys.stdout.write('{0}\r'.format(out))
                sys.stdout.flush()
            print

            losses.append(epoch_loss)
            if epoch % loss_update == 0:
                self.save('%s_epoch_%d.json' % (self.sess, epoch))
        history['losses'] = losses
        return history
