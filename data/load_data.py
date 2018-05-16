import os
import random
from utils import NLVocab, FOLVocab
import torch

# load atomic sents
DATADIR = 'data/raw/'

def get_atomic_sents(device):
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
    src_aug = src_vocab.aug_text(src)
    src_idx = [src_vocab.text_to_index(sent) for sent in src_aug]
    src_inputs = [
        torch.tensor(src_i, dtype=torch.long, device=device).view(-1, 1)
        for src_i in src_idx
    ]

    tar_vocab = FOLVocab(tar)
    tar_aug = tar_vocab.aug_text(tar)
    tar_idx = [tar_vocab.text_to_index(sent) for sent in tar_aug]
    tar_inputs = [
        torch.tensor(tar_i, dtype=torch.long, device=device).view(-1, 1)
        for tar_i in tar_idx
    ]
    print 'done.'

    return src_inputs, tar_inputs
