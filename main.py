from collections import Counter
import numpy as np
import seq2seq
from seq2seq.models import *
import re
from keras.optimizers import *
from keras.initializers import *
from models.utils import NLVocab, FOLVocab

BASELINE = True

datadir = 'platopy/folgen/'

def baseline2():
    source = []
    target = []
    with open(datadir+'atomic_sents.out', 'r') as f:
        for line in f:
            sent, atom = line.rstrip().split('\t')
            source.append(sent.replace('|', '').replace('.', '').lower())
            target.append(atom)
    print source[:10]
    print target[:10]
    src_vocab = NLVocab(source)
    src_inputs = src_vocab.preprocess(source[:10])
    tar_vocab = FOLVocab(target)
    tar_inputs = tar_vocab.preprocess(target[:10])

    _, input_length, input_dim = src_inputs.shape
    _, output_length, output_dim = tar_inputs.shape

    mod = Seq2Seq(output_dim=output_dim,
                  hidden_dim=100,
                  depth=2,
                  output_length=output_length,
                  input_shape=(input_length, input_dim))

    opt = SGD(lr=0.1)
    mod.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    history = mod.fit(src_inputs, tar_inputs, nb_epoch=1)
    inputs = [src_vocab.sequence_to_text(seq) for seq in src_inputs[:5]]
    gold = [tar_vocab.sequence_to_text(seq) for seq in tar_inputs[:5]]
    predictions = mod.predict(src_inputs[:5])
    pred = [tar_vocab.sequence_to_text(seq) for seq in predictions]

    for _input, _gold, _pred in zip(inputs, gold, pred):
        print _input
        print _gold
        print _pred
        print

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
        """
        the column vectors are the one-hot vectors
        """
        #seq = [vocab[word] for word in sent]
        seq = []
        for word in sent:
            one_hot = np.zeros(len(vocab), dtype=np.float32)
            one_hot[vocab[word]] = 1.0
            seq.append(one_hot)
        seq = np.array(seq)
        return seq.T

    def preprocess_src(src):
        vocab = get_src_vocab(src)
        aug_src = aug_source(src, vocab)
        ret = [src_to_sequence(sent, vocab) for sent in aug_src]
        ret = np.array(ret)
        return ret

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
        #seq = [vocab[word] for word in sent]
        seq = []
        for word in sent:
            one_hot = np.zeros(len(vocab), dtype=np.float32)
            one_hot[vocab[word]] = 1.0
            seq.append(one_hot)
        seq = np.array(seq)
        return seq.T

    def sequence_to_tar(seq, vocab):
        #sent = [vocab[ind] for ind in seq]
        indexes = np.argmax(seq, axis=0)
        sent = [vocab[ind] for ind in indexes]
        return sent

    def preprocess_tar(tar, vocab):
        aug_tar = aug_target(tar, vocab)
        ret = [tar_to_sequence(sent, vocab) for sent in aug_tar]
        ret = np.array(ret)
        return ret

    src_inputs = preprocess_src(source[:10])
    tar_vocab = get_tar_vocab(target[:10])
    tar_inputs = preprocess_tar(target[:10], tar_vocab)

    _, input_length, input_dim = src_inputs.shape
    _, output_length, output_dim = tar_inputs.shape

    print input_length, input_dim
    print output_length, output_dim

    models = []

    models.append(Seq2Seq(output_dim=output_dim,
                          hidden_dim=100,
                          depth=2,
                          output_length=output_length,
                          input_shape=(input_length, input_dim)))


    opt = SGD(lr=0.1)

    for mod in models:
        mod.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        history = mod.fit(src_inputs, tar_inputs, nb_epoch=100)
        sample = src_inputs[:5]
        predictions = mod.predict(sample)
        print source[:5]
        print [sequence_to_tar(seq, tar_vocab) for seq in predictions]

if __name__ == '__main__':
    if BASELINE:
        baseline2()
