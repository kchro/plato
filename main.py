from collections import Counter
import numpy as np
import seq2seq
from seq2seq.models import *
import re
from keras.optimizers import *
from keras.initializers import *
from models.utils import NLVocab, FOLVocab
from sklearn.model_selection import train_test_split

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
    if BASELINE:
        baseline()
