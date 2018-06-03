import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from tqdm import tqdm

from encoder import Encoder
from decoder import Decoder

class Seq2Seq:
    def __init__(self, input_size=None, hidden_size=None, output_size=None,
                 optimizer=optim.Adam, criterion=nn.NLLLoss, lr=0.0001,
                 src_vocab=None, tar_vocab=None,
                 sess='', device='cpu'):
        self.encoder = Encoder(input_size, hidden_size, device)
        self.decoder = Decoder(hidden_size, output_size, device)

        self.encoder_opt = optimizer(self.encoder.parameters(), lr=lr)
        self.decoder_opt = optimizer(self.decoder.parameters(), lr=lr)
        self.criterion = criterion()

        self.src_vocab = src_vocab
        self.tar_vocab = tar_vocab

        self.sess = sess
        self.device = device

    def save(self, filename):
        """
        save the model parameters to an enc & dec file
        """
        torch.save(self.encoder.state_dict(), 'logs/sessions/enc_%s' % filename)
        torch.save(self.decoder.state_dict(), 'logs/sessions/dec_%s' % filename)

    def load(self, filename):
        """
        load the model parameters
        """
        self.encoder.load_state_dict(torch.load('logs/sessions/enc_%s' % filename))
        self.decoder.load_state_dict(torch.load('logs/sessions/dec_%s' % filename))

    def run_epoch(self, src_input, tar_output, batch_size=20):
        """
        run one epoch
        """
        self.encoder_opt.zero_grad()
        self.decoder_opt.zero_grad()

        # encode the source input
        encoder_output, encoder_hidden = self.encoder(src_input, batch_size=batch_size)

        SOS_token = self.tar_vocab.word_to_index['<S>']
        decoder_input = torch.tensor([SOS_token]*batch_size,
                                     dtype=torch.long,
                                     device=self.device).view(-1, 1)

        decoder_hidden = encoder_hidden

        loss = 0

        # Teacher forcing: Feed the target as the next input
        tar_len = self.tar_vocab.max_len
        for di in range(tar_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                          decoder_hidden,
                                                          batch_size=batch_size)

            targets = tar_output[:, di:di+1].view(-1)
            loss += self.criterion(decoder_output, targets)
            decoder_input = tar_output[:, di:di+1] # Teacher forcing

        loss.backward()
        self.encoder_opt.step()
        self.decoder_opt.step()

        return loss.item() / tar_len

    def train(self, X_train, y_train, epochs=10,
              batch_size=20, loss_update=10):
        """
        training function
        """
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

            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                if len(X_batch) < batch_size:
                    continue

                X_batch = torch.tensor(X_batch,
                                       dtype=torch.long,
                                       device=self.device)
                y_batch = torch.tensor(y_batch,
                                       dtype=torch.long,
                                       device=self.device)

                loss = self.run_epoch(X_batch, y_batch,
                                      batch_size=batch_size)
                cum_loss += loss
                epoch_loss += loss

                progress = get_progress(i, len(X_train), length=20)
                out = '%d/%d [%s] loss: %f' % (i, len(X_train), progress, epoch_loss/(i+1))
                sys.stdout.write('{0}\r'.format(out))
                sys.stdout.flush()
            print

            losses.append(epoch_loss)
            if epoch % loss_update == 0:
                self.save('%s_epoch_%d.json' % (self.sess, epoch))
        history['losses'] = losses
        return history

    def predict(self, X_test):
        """
        run predictions on the test data
        """
        with torch.no_grad():
            decoded_text = []

            for i in range(len(X_test)):
                src_input = torch.LongTensor(X_test[i], device=self.device).view(1, -1)

                # encode the source input
                encoder_output, encoder_hidden = self.encoder(src_input)

                SOS_token = self.tar_vocab.word_to_index['<S>']
                EOS_token = self.tar_vocab.word_to_index['</S>']
                decoder_input = torch.tensor([[SOS_token]],
                                             dtype=torch.long,
                                             device=self.device)
                decoder_hidden = encoder_hidden

                decoded_seq = []
                tar_len = self.tar_vocab.max_len
                for di in range(tar_len):
                    decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                                  decoder_hidden)
                    topv, topi = decoder_output.data.topk(1)
                    idx = topi.item()
                    decoded_seq.append(idx)

                    decoder_input = topi.squeeze().detach()
                decoded_seq = torch.tensor(decoded_seq,
                                           dtype=torch.long,
                                           device=self.device)
                decoded_text.append(decoded_seq)
        return decoded_text

    def evaluate(self, X_test, y_test, preds, out=None):
        """
        for seq2seq models, X_test, y_test, and preds are all going to be
        lists of lists of indexes => [[idx]]
        """
        if out:
            outfile = out
            errfile = 'err_'+out
        else:
            outfile = 'logs/sessions/%s.out' % self.sess
            errfile = 'logs/sessions/err_%s.out' % self.sess

        print 'logging to %s...' % outfile

        num_correct = 0

        with open(outfile, 'w') as w:
            with open(errfile, 'w') as err:
                for nl_idx, fol_gold_idx, fol_pred_idx in zip(X_test, y_test, preds):
                    nl_sent = self.src_vocab.reverse(nl_idx)
                    fol_gold = self.tar_vocab.reverse(fol_gold_idx)
                    fol_pred = self.tar_vocab.reverse(fol_pred_idx)

                    if fol_gold != fol_pred:
                        err.write('input:  '+nl_sent+'\n')
                        err.write('gold:   '+fol_gold+'\n')
                        err.write('output: '+fol_pred+'\n')
                        err.write('\n')
                    else:
                        num_correct += 1

                    w.write('%s\t%s\t%s\t\n' % (nl_sent, fol_gold, fol_pred))

        print '########################'
        print '# Evaluation:'
        print '# %d out of %d correct' % (num_correct, len(preds))
        print '# %0.3f accuracy' % (float(num_correct) / len(preds))
        print '########################'
