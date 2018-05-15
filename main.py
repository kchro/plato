from collections import Counter
import numpy as np
import seq2seq
from seq2seq import *
import re
from keras.optimizers import *

BASELINE = True

datadir = 'platopy/folgen/'

def baseline():
    source = []
    target = []
    with open(datadir+'atomic_sents.out', 'r') as f:
        for line in f:
            sent, atom = line.rstrip().split('\t')
            source.append(sent.replace('|', '').replace('.', '').lower())
            target.append(atom)

    def get_src_vocab(src, n_words=None):
        wc = Counter([w for sent in src for w in sent.split()])
        wc = wc.most_common(n_words) if n_words else wc.items()
        vocab = {w for w, c in wc}
        vocab.add('<UNK>')
        vocab.add('<S>')
        vocab.add('</S>')
        vocab.add('<PAD>')
        vocab = sorted(vocab)
        word_to_index = { vocab[i]: i for i in range(len(vocab)) }
        index_to_word = { i: vocab[i] for i in range(len(vocab)) }
        word_to_index.update(index_to_word)
        return word_to_index

    def aug_source(src, vocab):
        aug_src = []
        max_len = max(len(sent.split())+2 for sent in src)
        for sent in src:
            aug_sent = []
            aug_sent.append('<S>')
            for word in sent.split():
                if word not in vocab:
                    aug_sent.append('<UNK>')
                else:
                    aug_sent.append(word)
            aug_sent.append('</S>')

            while len(aug_sent) < max_len:
                aug_sent.append('<PAD>')

            aug_src.append(aug_sent)
        return aug_src

    def src_to_sequence(sent, vocab):
        seq = [vocab[word] for word in sent]
        return seq

    def preprocess_src(src):
        vocab = get_src_vocab(src)
        aug_src = aug_source(src, vocab)
        ret = [src_to_sequence(sent, vocab) for sent in aug_src]
        ret = np.array(ret)
        m, n = ret.shape
        return ret.reshape((m, n, 1))

    def get_tar_vocab(tar, n_words=None):
        wc = Counter([w for sent in tar for w in re.findall(r"[\w']+", sent)])
        wc = wc.most_common(n_words) if n_words else wc.items()
        vocab = {w for w, c in wc}
        vocab.add('(')
        vocab.add(')')
        vocab.add(',')
        vocab.add('<UNK>')
        vocab.add('<S>')
        vocab.add('</S>')
        vocab.add('<PAD>')
        vocab = sorted(vocab)
        word_to_index = { vocab[i]: i for i in range(len(vocab)) }
        index_to_word = { i: vocab[i] for i in range(len(vocab)) }
        word_to_index.update(index_to_word)
        return word_to_index

    operators = '&'

    def replace(s, chars):
        for ch in chars:
            s = s.replace(ch, ' %s ' % ch)
        return s

    def aug_target(tar, vocab):
        aug_tar = []
        max_len = 0
        tmp_tar = []
        for sent in tar:
            sent = replace(sent, '(),'+operators).split()
            aug_sent = []
            aug_sent.append('<S>')
            for word in sent:
                if word not in vocab:
                    aug_sent.append('<UNK>')
                else:
                    aug_sent.append(word)
            aug_sent.append('</S>')
            max_len = max(max_len, len(aug_sent))
            tmp_tar.append(aug_sent)
        for aug_sent in tmp_tar:
            while len(aug_sent) < max_len:
                aug_sent.append('<PAD>')
            aug_tar.append(aug_sent)
        return aug_tar

    def tar_to_sequence(sent, vocab):
        seq = [vocab[word] for word in sent]
        return seq

    def sequence_to_tar(seq, vocab):
        sent = [vocab[ind] for ind in seq]
        return sent

    def preprocess_tar(tar, vocab):
        aug_tar = aug_target(tar, vocab)
        ret = [tar_to_sequence(sent, vocab) for sent in aug_tar]
        ret = np.array(ret)
        m, n = ret.shape
        return ret.reshape((m, n, 1))

    src_inputs = preprocess_src(source)
    tar_vocab = get_tar_vocab(target)
    tar_inputs = preprocess_tar(target, tar_vocab)

    _, input_length, input_dim = src_inputs.shape
    _, output_length, output_dim = tar_inputs.shape

    mod = Seq2Seq(output_dim=output_dim,
                  hidden_dim=200,
                  output_length=output_length,
                  input_shape=(input_length, input_dim))

    opt = SGD(lr=0.1, momentum=0.0)

    mod.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
    history = mod.fit(src_inputs, tar_inputs, nb_epoch=100)

if __name__ == '__main__':
    if BASELINE:
        baseline()
