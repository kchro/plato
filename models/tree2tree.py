import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys

from encoder import TreeEncoder
from decoder import TreeDecoder

from tree_utils import DepTree, Tree

class Tree2Tree:
    def __init__(self, input_size=None, hidden_size=None, output_size=None,
                 optimizer=optim.Adam, criterion=nn.NLLLoss, lr=0.0001,
                 src_vocab=None, tar_vocab=None,
                 sess='', device='cpu'):
        self.encoder = TreeEncoder(input_size, hidden_size, device)
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

    def run_epoch(self, src_input, tar_output, batch_size=1):
        """
        one training epoch
        """
        self.encoder_opt.zero_grad()
        self.decoder_opt.zero_grad()

        # encode the source input
        _, encoder_hidden = self.encoder(src_input)

        SOS_token = self.tar_vocab.word_to_index['<S>']
        EOS_token = self.tar_vocab.word_to_index['</S>']
        NON_token = self.tar_vocab.word_to_index['<N>']

        #decoder_input = torch.LongTensor([[SOS_token]], device=self.device)
        decoder_hidden = encoder_hidden

        loss = 0

        tar_root = Tree(formula=tar_output[0])
        tar_queue = [
            self.tar_vocab.sent_to_idx(formula)
            for formula in tar_root.inorder()
        ]

        # see Dong et al. (2016) [Algorithm 1]
        root = {
            'parent': decoder_hidden[0],    # (1, 1, 200)
            'hidden': decoder_hidden,       # (1, 1, 200) * 2
            'children': []
        }

        queue = [root]
        tar_count = 0

        while queue:
            # until no more nonterminals
            subtree = queue.pop(0)
            tar_seq = tar_queue.pop(0)
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
            for i in range(len(tar_seq)):
                # decode the input sequence
                decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                              hidden=decoder_hidden,
                                                              parent=parent_input)
                # interpret the output
                idx, decoder_input = self.get_idx(decoder_output)

                loss += self.criterion(decoder_output, torch.tensor([tar_seq[i]],
                                                                    dtype=torch.long,
                                                                    device=self.device))

                # if we have a non-terminal token
                if tar_seq[i] == NON_token:
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
                else:
                    subtree['children'].append(idx)

        loss.backward()
        self.encoder_opt.step()
        self.decoder_opt.step()

        return loss.item() / tar_count

    def str_to_deptree(self, sent):
        deptree = DepTree(sent=sent,
                          src_vocab=self.src_vocab,
                          device=self.device)
        return deptree

    def train(self, X_train, y_train, epochs=10, batch_size=1, loss_update=10):
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

        for epoch in range(epochs):
            epoch_loss = 0

            print 'Epoch %d/%d' % (epoch, epochs)

            for i in range(0, len(X_train), batch_size):
                if i+batch_size > len(X_train):
                    continue

                if epoch == 0:
                    for j in range(i, i+batch_size):
                        X_train[j] = self.str_to_deptree(X_train[j])

                X_batch = X_train[i]

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
                src_input = self.str_to_deptree(X_test[i])

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
                    decoder_input = torch.LongTensor([[SOS_token]], device=self.device)

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
        for tree2tree models,
        X_test, y_test is going to be a list of sents
        list of sents => [sents]
        preds are going to be
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
                for nl_sent, fol_gold, fol_pred_idx in zip(X_test, y_test, preds):
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

    def save(self, filename):
        torch.save(self.encoder.state_dict(), 'logs/sessions/enc_%s' % filename)
        torch.save(self.decoder.state_dict(), 'logs/sessions/dec_%s' % filename)

    def load(self, filename):
        self.encoder.load_state_dict(torch.load('logs/sessions/enc_%s' % filename))
        self.decoder.load_state_dict(torch.load('logs/sessions/dec_%s' % filename))
