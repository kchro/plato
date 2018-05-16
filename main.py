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

if __name__ == '__main__':
    if BASELINE:
        baseline2()
