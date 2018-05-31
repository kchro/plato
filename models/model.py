

class Seq2Tree:
    def __init__(self, input_size=0, hidden_size=0, output_size=0,
                 optimizer=optim.Adam, criterion=nn.NLLLoss, lr=0.0001,
                 sess='', device='cpu'):
        self.encoder = Encoder(input_size, hidden_size, device)
        self.decoder = TreeDecoder(hidden_size, output_size, device)

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

    def get_idx(self, decoder_output):
        topv, topi = decoder_output.data.topk(1)
        idx = topi.item()
        decoder_input = topi.squeeze().detach()
        return idx, decoder_input

    def run_epoch(self, src_input, tar_output, batch_size=20):
        self.encoder_opt.zero_grad()
        self.decoder_opt.zero_grad()

        # encode the source input
        encoder_output, encoder_hidden = self.encoder(src_input, batch_size=batch_size)

        SOS_token = self.tar_vocab.word_to_index['<S>']
        EOS_token = self.tar_vocab.word_to_index['</S>']
        NON_token = self.tar_vocab.word_to_index['<N>']
        SON_token = self.tar_vocab.word_to_index['(']

        loss = 0

        decoder_hidden = encoder_hidden

        # see Dong et al. (2016) [Algorithm 1]
        root = {
            'parent': None,
            'hidden': decoder_hidden,
            'sos': SOS_token,
            'children': []
        }

        queue = [root]
        while queue:
            # until no more nonterminals
            subtree = queue.pop(0)

            # get the starting input [either <S> or '(']
            idx = subtree['idx']

            # NOTE: batch_size is 1
            decoder_input = torch.LongTensor([idx], device=self.device).view(-1, 1)

            # get the hidden
            parent_input = subtree['parent']

            # parse the formula until a </S> token
            while idx != EOS_token:
                # decode the input sequence
                decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                              decoder_hidden,
                                                              parent_input)

                # interpret the output
                idx, decoder_input = self.get_idx(decoder_output)

                # if we generate a non-terminal token
                if idx == NON_token:
                    # add a subtree to the queue
                    ### parent: the previous state for <n>
                    ### hidden: the context vector for <n>
                    ### idx: the starting token ['(']
                    ### children: subtrees
                    nonterminal = {
                        'parent': decoder_output,
                        'hidden': decoder_hidden,
                        'idx': SON_token,
                        'children': []
                    }

                    queue.append(nonterminal)
                    subtree['children'].append(nonterminal)
                else:
                    subtree['children'].append(idx)



        loss.backward()
        self.encoder_opt.step()
        self.decoder_opt.step()

        return loss.item() / tar_len

    def train(self, X_train, y_train, epochs=10, batch_size=20, loss_update=10):
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

                X_batch = torch.LongTensor(X_batch, device=self.device)
                y_batch = torch.LongTensor(y_batch, device=self.device)

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
        with torch.no_grad():
            decoded_text = []

            for i in range(len(X_test)):
                src_input = torch.LongTensor(X_test[i], device=self.device).view(1, -1)

                # encode the source input
                encoder_output, encoder_hidden = self.encoder(src_input)

                SOS_token = self.tar_vocab.word_to_index['<S>']
                EOS_token = self.tar_vocab.word_to_index['</S>']
                decoder_input = torch.LongTensor([[SOS_token]], device=self.device)
                decoder_hidden = encoder_hidden

                decoded_seq = []
                # tar_len = self.tar_vocab.max_len
                # for di in range(tar_len):
                while True:
                    decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                                  decoder_hidden)
                    topv, topi = decoder_output.data.topk(1)
                    idx = topi.item()
                    decoded_seq.append(idx)

                    if idx == EOS_token:
                        break

                    decoder_input = topi.squeeze().detach()
                decoded_seq = torch.LongTensor(decoded_seq, device=self.device)
                decoded_text.append(decoded_seq)
        return decoded_text
