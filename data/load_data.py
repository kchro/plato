import os
import random
from utils import NLVocab, FOLVocab
import torch

# load atomic sents
DATADIR = 'data/raw/'

def get_polish_formula(formula):
    operators = ['~', '&', '|', '-->', '<->']

    def count_operators(formula):
        count = 0
        for op in operators:
            count += formula.count(op)
        return count

    # base case:
    # if no operators in formula
    num_ops = count_operators(formula)
    if num_ops == 0:
        return formula
    if num_ops == 1:
        if formula[0] == '(' and formula[-1] == ')':
            formula = formula[1:-1]

    # find the root operator
    paren_count = 0
    for i in range(len(formula)):
        if formula[i] == '(':
            paren_count += 1
        elif formula[i] == ')':
            paren_count -= 1
        elif paren_count == 0:
            for op in operators:
                if op == '~':
                    continue
                if formula[i:].startswith(op):
                    left = get_polish_formula(formula[:i])
                    operator = formula[i:i+len(op)]
                    right = get_polish_formula(formula[i+len(op):])
                    return '%s(%s,%s)' % (operator, left, right)

    return formula

def load_file(filename='',
              encoder='seq',
              decoder='seq',
              device='cpu',
              get_vocabs=True):
    """
    all purpose file-loader
    @params
        filename    (str)
        encoder     ('seq' or 'tree')
        decoder     ('seq' or 'tree')
        device      ('cpu' or 'gpu')
        get_vocabs  (bool)
        - return the vocab or not
    """
    filename = os.path.join(DATADIR, filename)
    def normalize_src(s):
        s = s.replace('|', '')
        s = s.replace('.', '')
        s = s.lower()
        return s

    src = []
    tar = []
    with open(filename, 'r') as f:
        for line in f:
            # kind of hacky, but whatever. i fucked up the dataset.
            # some files are of form: <nl sent>\t<fol>\t<pol>
            # others are            : <nl sent>\t<fol>
            nl_sent, fol_form = line.rstrip().split('\t')[:2]

            # natural language source
            src.append(normalize_src(nl_sent))

            # FOL target (polish or not)
            if decoder == 'seq':
                tar.append(fol_form)
            else:
                pol_form = get_polish_formula(fol_form)
                tar.append(pol_form)

    rand_src, rand_tar = random.choice(list(zip(src, tar)))
    print 'ex. NL sentence:', rand_src
    print 'ex. FOL formula:', rand_tar

    print 'converting source and target to index tensors...',
    src_vocab = NLVocab(src)
    src_inputs = src_vocab.get_idx_tensor(src)

    tar_vocab = FOLVocab(tar)
    if decoder == 'seq':
        tar_inputs = tar_vocab.get_idx_tensor(tar)
    else:
        tar_inputs = tar
    print 'done.'

    if get_vocabs:
        return (src_inputs, tar_inputs), (src_vocab, tar_vocab)
    else:
        return src_inputs, tar_inputs

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
