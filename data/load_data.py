import os
import random
from utils import NLVocab, FOLVocab
import torch

# load atomic sents
DATADIR = 'data/raw/'

def get_k1_sents(device, get_vocabs=True):
    filename = os.path.join(DATADIR, 'k1_sents.out')

    def normalize_src(s):
        s = s.replace('|', '')
        s = s.replace('.', '')
        s = s.lower()
        return s

    src = []
    tar = []
    with open(filename, 'r') as f:
        for line in f:
            nl_sent, fol_form = line.rstrip().split('\t')
            src.append(normalize_src(nl_sent))
            tar.append(fol_form)

    rand_src, rand_tar = random.choice(list(zip(src, tar)))
    print 'ex. NL sentence:', rand_src
    print 'ex. FOL formula:', rand_tar

    print 'converting source and target to index tensors...',
    src_vocab = NLVocab(src)
    src_inputs = src_vocab.get_idx_tensor(src)

    tar_vocab = FOLVocab(tar)
    tar_inputs = tar_vocab.get_idx_tensor(tar)
    print 'done.'

    if get_vocabs:
        return (src_inputs, tar_inputs), (src_vocab, tar_vocab)
    else:
        return src_inputs, tar_inputs

def get_atomic_sents(device, get_vocabs=True):
    filename = os.path.join(DATADIR, 'atomic_sents.out')

    def normalize_src(s):
        s = s.replace('|', '')
        s = s.replace('.', '')
        s = s.lower()
        return s

    src = []
    tar = []
    with open(filename, 'r') as f:
        for line in f:
            nl_sent, fol_atom = line.rstrip().split('\t')
            src.append(normalize_src(nl_sent))
            tar.append(fol_atom)

    rand_src, rand_tar = random.choice(list(zip(src, tar)))
    print 'ex. NL sentence:', rand_src
    print 'ex. FOL formula:', rand_tar

    print 'converting source and target to index tensors...',
    src_vocab = NLVocab(src)
    src_inputs = src_vocab.get_idx_tensor(src)

    tar_vocab = FOLVocab(tar)
    tar_inputs = tar_vocab.get_idx_tensor(tar)
    print 'done.'

    if get_vocabs:
        return (src_inputs, tar_inputs), (src_vocab, tar_vocab)
    else:
        return src_inputs, tar_inputs

if __name__ == '__main__':
    inputs, vocabs = get_atomic_sents('cpu')
    src_inputs, tar_inputs = inputs
    src_vocab, tar_vocab = vocabs
