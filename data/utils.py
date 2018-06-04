from collections import Counter
import numpy as np
import re
import torch
import spacy

class Vocab(object):
    def __init__(self, text, n_words, charset, join, device):
        def split(self, sent):
            for ch in charset:
                sent = sent.replace(ch, ' %s ' % ch)
            return sent.split()

        self.split = split
        freq = Counter([w for sent in text for w in self.split(sent)])
        freq = freq.most_common(n_words) if n_words else freq.items()
        vocab = {w for w, c in freq}
        vocab.add('<S>')
        vocab.add('</S>')
        vocab.add('<PAD>')
        vocab.add('<UNK>')

        # nonterminal for the target vocab (tree)
        vocab.add('<N>')

        self.vocab = sorted(vocab)
        self.word_to_index = { self.vocab[i]: i for i in range(len(self.vocab)) }
        self.index_to_word = { i: self.vocab[i] for i in range(len(self.vocab)) }
        self.join = join
        self.device = device

    def replace(self, s, chars):
        for ch in chars:
            s = s.replace(ch, ' %s ' % ch)
        return s



    def augment_sent(self, sent):
        aug = []
        aug.append('<S>')
        for word in self.split(sent):
            if word not in self.vocab:
                aug.append('<UNK>')
            else:
                aug.append(word)
        aug.append('</S>')
        return aug

    def augment_text(self, text):
        aug_text = []
        for sent in text:
            aug = self.augment_sent(sent)
            aug_text.append(aug)

        self.max_len = max(len(aug) for aug in aug_text)

        for i in range(len(aug_text)):
            while len(aug_text[i]) < self.max_len:
                aug_text[i].append('<PAD>')

        return aug_text

    def sent_to_idx(self, sent, augmented=False):
        if augmented:
            aug = sent
        else:
            aug = self.augment_sent(sent)
        idx = [self.word_to_index[word] for word in aug]
        return idx

    def get_idx_tensor(self, text):
        tensors = []
        for aug in self.augment_text(text):
            idx = [self.word_to_index[word] for word in aug]
            # idx = torch.tensor(idx, dtype=torch.long, device=self.device)
            tensors.append(idx)
        # tensors = torch.Tensor(tensors, device=self.device)
        return tensors

    def reverse(self, seq):
        idxs = []
        for num in seq:
            if isinstance(num, int):
                idxs.append(num)
            else:
                idxs.append(num.item())
        sent = [self.index_to_word[idx] for idx in idxs]
        sent = self.join.join(sent)
        return sent

    def reverse_text(self, sequences):
        text = []
        for seq in sequences:
            sent = self.reverse(seq)
            text.append(sent)
        return text

    def __len__(self):
        return len(self.vocab)

class NLVocab(Vocab):
    def __init__(self, text, n_words=2000, device='cpu'):
        super(NLVocab, self).__init__(text, n_words,
                                      charset=',;', join=' ',
                                      device=device)

class FOLVocab(Vocab):
    def __init__(self, text, n_words=2000, device='cpu'):
        super(FOLVocab, self).__init__(text, n_words,
                                       charset=',()', join='',
                                       device=device)
