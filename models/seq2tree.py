import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys

from encoder import Encoder
from decoder import TreeDecoder

from tree_utils import Tree

class Seq2Tree:
    def __init__(self, input_size=None, hidden_size=None, output_size=None,
                 optimizer=optim.Adam, criterion=nn.NLLLoss, lr=0.0001,
                 src_vocab=None, tar_vocab=None,
                 sess='', device='cpu'):
        self.encoder = Encoder(input_size, hidden_size, device)
        self.decoder = TreeDecoder(hidden_size, output_size, device)

        if torch.cuda.is_available():
            self.encoder.cuda()
            self.decoder.cuda()

        self.encoder_opt = optimizer(self.encoder.parameters(), lr=lr)
        self.decoder_opt = optimizer(self.decoder.parameters(), lr=lr)
        self.criterion = criterion()

        self.src_vocab = src_vocab
        self.tar_vocab = tar_vocab

        self.sess = sess
        self.device = device

    def get_idx(self, decoder_output):
        topv, topi = decoder_output.data.topk(1)
        idx = topi.item()
        decoder_input = topi.squeeze().detach()
        return idx, decoder_input

    def run_epoch(self, src_inputs, tar_outputs, batch_size=1):
        """
        one training epoch
        """
        self.encoder_opt.zero_grad()
        self.decoder_opt.zero_grad()

        # encode the source input
        _, (encoder_h, encoder_c) = self.encoder(src_inputs,
                                                 batch_size=batch_size)

        # print encoder_h.shape # (1, 20, 10)
        # print encoder_c.shape # (1, 20, 10)

        SOS_token = self.tar_vocab.word_to_index['<S>']
        EOS_token = self.tar_vocab.word_to_index['</S>']
        NON_token = self.tar_vocab.word_to_index['<N>']

        loss = 0
        decoder_h = encoder_h.view(batch_size, 1, 1, -1)
        decoder_c = encoder_c.view(batch_size, 1, 1, -1)

        tar_count = 0

        for batch in range(batch_size):
            decoder_hidden = decoder_h[batch], decoder_c[batch]
            # (1, 1, hidden), (1, 1, hidden)

            # see Dong et al. (2016) [Algorithm 1]
            root = {
                'parent': decoder_hidden[0],    # (1, 1, 200)
                'hidden': decoder_hidden,       # (1, 1, 200) * 2
            }

            tar_idx = 0

            queue = [root]

            while queue:
                # until no more nonterminals
                subtree = queue.pop(0)
                # get the next subtree in tar_output
                tar_seq = tar_outputs[batch][tar_idx]

                # count items in sequence (for averaging loss)
                tar_count += len(tar_seq)

                # initialize the sequence
                # NOTE: batch_size is 1
                decoder_input = torch.tensor([[SOS_token]],
                                             dtype=torch.long,
                                             device=self.device)

                # get the parent-feeding vector
                parent_input = subtree['parent']

                idx = SOS_token

                # Teacher forcing with trees
                for i in range(1, len(tar_seq)):
                    # decode the input sequence
                    decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                                  hidden=decoder_hidden,
                                                                  parent=parent_input)
                    # interpret the output
                    idx, decoder_input = self.get_idx(decoder_output)

                    # get the desired output
                    target_output = torch.tensor([tar_seq[i]],
                                                 dtype=torch.long,
                                                 device=self.device)

                    # calculate loss
                    loss += self.criterion(decoder_output, target_output)

                    # if we have a non-terminal token
                    if tar_seq[i] == NON_token:
                        # add a subtree to the queue
                        ### parent: the previous state for <n>
                        ### hidden: the hidden state for <n>
                        ### children: subtrees
                        nonterminal = {
                            'parent': decoder_hidden[0],
                            'hidden': decoder_hidden,
                            # 'children': []
                        }

                        queue.append(nonterminal)

                    decoder_input = target_output # Teacher forcing

                # next subtree in tar_output
                tar_idx += 1

        loss.backward()
        self.encoder_opt.step()
        self.decoder_opt.step()

        return loss.item() / tar_count

    def train(self, X_train, y_train, epochs=10, retrain=0, batch_size=10, loss_update=10):
        cum_loss = 0
        history = {}
        losses = []

        def get_progress(num, den, length):
            """
            simple progress bar
            """
            if num == den-1:
                return '='*length
            arrow = int(float(num)/den*length)
            return '='*(arrow-1)+'>'+'.'*(20-arrow)

        for epoch in range(retrain, epochs):
            epoch_loss = 0

            print 'Epoch %d/%d' % (epoch, epochs)

            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i+batch_size]

                if len(X_batch) < batch_size:
                    continue

                if epoch == 0:
                    # initialize training data to trees
                    for j in range(i, i+batch_size):
                        root = Tree(formula=y_train[j])
                        y_train[j] = [self.tar_vocab.sent_to_idx(formula) for formula in root.inorder()]

                X_batch = torch.tensor(X_batch,
                                       dtype=torch.long,
                                       device=self.device)
                y_batch = y_train[i:i+batch_size]

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

    def flatten(self, root):
        decoded_seq = []
        for child in root['children']:
            if isinstance(child, int):
                decoded_seq.append(child)
            else:
                flattened = self.flatten(child)
                decoded_seq += flattened
        return decoded_seq

    def predict(self, X_test):
        with torch.no_grad():
            decoded_text = []

            for i in range(len(X_test)):
                src_input = torch.tensor(X_test[i],
                                         dtype=torch.long,
                                         device=self.device).view(1, -1)

                # encode the source input
                encoder_output, encoder_hidden = self.encoder(src_input)

                SOS_token = self.tar_vocab.word_to_index['<S>']
                EOS_token = self.tar_vocab.word_to_index['</S>']
                NON_token = self.tar_vocab.word_to_index['<N>']

                decoder_hidden = encoder_hidden

                # see Dong et al. (2016) [Algorithm 1]
                root = {
                    'parent': decoder_hidden[0],    # (1, 1, 200)
                    'hidden': decoder_hidden,       # (1, 1, 200) * 2
                    'children': []
                }

                queue = [root]
                depth = 0
                while queue:
                    # until no more nonterminals
                    subtree = queue.pop(0)

                    # initialize the sequence
                    # NOTE: batch_size is 1
                    decoder_input = torch.tensor([[SOS_token]],
                                                 dtype=torch.long,
                                                 device=self.device)

                    # get the parent-feeding vector
                    parent_input = subtree['parent']

                    idx = SOS_token

                    while idx != EOS_token:
                        # decode the input sequence
                        decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                                      hidden=decoder_hidden,
                                                                      parent=parent_input)

                        # interpret the output
                        idx, decoder_input = self.get_idx(decoder_output)

                        # if exceeds max length
                        if len(subtree['children']) == 20:
                            idx = EOS_token
                        # if exceed max depth
                        if depth > 3:
                            idx = EOS_token

                        # if we have a non-terminal token
                        if idx == NON_token:

                            # add a subtree to the queue
                            ### parent: the previous state for <n>
                            ### hidden: the hidden state for <n>
                            ### children: subtrees
                            nonterminal = {
                                'parent': decoder_hidden[0],
                                'hidden': decoder_hidden,
                                'children': []
                            }

                            queue.append(nonterminal)
                            subtree['children'].append(nonterminal)

                            depth += 1
                        else:
                            subtree['children'].append(idx)

                decoded_seq = self.flatten(root)
                decoded_text.append(decoded_seq)

        return decoded_text

    def evaluate(self, X_test, y_test, preds, out=None):
        """
        for seq2tree models, X_test and preds are going to be
        lists of lists of indexes => [[idx]]

        y_test is going to be
        list of sents => [sent]
        """
        if out:
            outfile = out
            errfile = 'err_'+out
        else:
            outfile = 'logs/sessions/%s.out' % self.sess
            errfile = 'logs/sessions/err_%s.out' % self.sess

        print 'logging to %s...' % outfile

        num_correct = 0

        def preprocess(fol_pred_idx):
            fol_pred = self.tar_vocab.reverse(fol_pred_idx)
            return fol_pred.replace('<S>', '').replace('</S>', '')

        with open(outfile, 'w') as w:
            with open(errfile, 'w') as err:
                for nl_idx, fol_gold, fol_pred_idx in zip(X_test, y_test, preds):
                    nl_sent = self.src_vocab.reverse(nl_idx)

                    fol_pred = preprocess(fol_pred_idx)

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

    def save(self, filename):
        torch.save(self.encoder.state_dict(), 'logs/sessions/enc_%s' % filename)
        torch.save(self.decoder.state_dict(), 'logs/sessions/dec_%s' % filename)

    def load(self, filename):
        self.encoder.load_state_dict(torch.load('logs/sessions/enc_%s' % filename))
        self.decoder.load_state_dict(torch.load('logs/sessions/dec_%s' % filename))
