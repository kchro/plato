import numpy as np
import re
import torch
import torch.nn as nn
import json
# deprecate soon...
from keras.optimizers import *
from keras.initializers import *
from models.utils import NLVocab, FOLVocab
###################
from sklearn.model_selection import train_test_split
# from models.seq2seq import Seq2Seq
from data.load_data import get_atomic_sents
# from models.seq2seq import EncoderRNN
from models.model import Seq2Seq
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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
    print source[:10]
    print target[:10]
    src_vocab = NLVocab(source)
    src_inputs = src_vocab.preprocess(source)
    tar_vocab = FOLVocab(target)
    tar_inputs = tar_vocab.preprocess(target)

    X_train, X_test, y_train, y_test = train_test_split(src_inputs, tar_inputs, test_size=0.9)

    _, input_length, input_dim = src_inputs.shape
    _, output_length, output_dim = tar_inputs.shape

    mod = AttentionSeq2Seq(output_dim=output_dim,
                           hidden_dim=100,
                           depth=2,
                           output_length=output_length,
                           input_shape=(input_length, input_dim))

    #opt = SGD(lr=0.4)
    opt = Adam()
    mod.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
    history = mod.fit(X_train, y_train, epochs=10)
    inputs = [src_vocab.sequence_to_text(seq) for seq in X_test]
    gold = [tar_vocab.sequence_to_text(seq) for seq in y_test]
    predictions = mod.predict(X_test)
    pred = [tar_vocab.sequence_to_text(seq) for seq in predictions]

    for _input, _gold, _pred in zip(inputs, gold, pred):
        print _input
        print _gold
        print _pred
        print

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print 'running on the %s' % device

    sess = raw_input('session name: ')

    # get the data
    inputs, vocabs = get_atomic_sents(device, get_vocabs=True)
    src_inputs, tar_inputs = inputs
    src_vocab, tar_vocab = vocabs
    X_train, X_test, y_train, y_test = train_test_split(src_inputs, tar_inputs, test_size=0.1)

    # load the model
    embedding_size = 100
    hidden_size = 200

    print 'loading the model...',
    model = Seq2Seq(input_size=len(src_vocab),
                    hidden_size=hidden_size,
                    output_size=len(tar_vocab),
                    sess=sess, device=device)

    # model = Seq2Tree(input_size=len(src_vocab),
    #                  hidden_size=hidden_size,
    #                  output_size=len(tar_vocab),
    #                  sess=sess, device=device)

    model.set_vocab(src_vocab, tar_vocab)
    print 'done.'

    print 'training the model...'
    history = model.train(X_train, y_train)
    print 'done.'

    with open('logs/sessions/%s.json' % sess.replace(' ', '_'), 'w') as w:
        w.write(json.dumps(history))
