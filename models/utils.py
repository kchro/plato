from collections import Counter
import numpy as np
import re

class NLVocab:
    def __init__(self, text, n_words=None):
        wc = Counter([w for sent in text for w in sent.split()])
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
        self.vocab = vocab
        self.word_to_index = { vocab[i]: i for i in range(len(vocab)) }
        self.index_to_word = { i: vocab[i] for i in range(len(vocab)) }

    def aug_text(self, text):
        aug = []
        max_len = 0

        # add the start, end, and unk tokens
        for sent in text:
            aug_sent = []
            aug_sent.append('<S>')
            for word in sent.split():
                if word not in self.vocab:
                    aug_sent.append('<UNK>')
                else:
                    aug_sent.append(word)
            aug_sent.append('</S>')
            aug.append(aug_sent)
            max_len = max(max_len, len(aug_sent))

        # pad the sentences
        for i in range(len(aug)):
            while len(aug[i]) < max_len:
                aug[i].append('<PAD>')

        return aug

    def text_to_sequence(self, sent):
        seq = []
        for word in sent:
            one_hot = np.zeros(len(self.vocab), dtype=np.float32)
            one_hot[self.word_to_index[word]] = 1.0
            seq.append(one_hot)
        seq = np.array(seq).T
        return seq

    def sequence_to_text(self, seq):
        text = []
        for i in range(1, len(seq.T)-1):
            ind = np.flatnonzero(seq.T[i])[0]
            word = self.index_to_word[ind]
            text.append(word)
        return ' '.join(text)

    def preprocess(self, text):
        aug_text = self.aug_text(text)
        pro_text = [self.text_to_sequence(sent) for sent in aug_text]
        pro_text = np.array(pro_text)
        return pro_text

class FOLVocab:
    def __init__(self, text, n_words=None):
        wc = Counter([w for sent in text for w in re.findall(r"[\w']+", sent)])
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
        self.vocab = vocab
        self.word_to_index = { vocab[i]: i for i in range(len(vocab)) }
        self.index_to_word = { i: vocab[i] for i in range(len(vocab)) }

    def aug_text(self, text, operators='&'):
        aug = []
        max_len = 0

        def replace(s, chars):
            for ch in chars:
                s = s.replace(ch, ' %s ' % ch)
            return s

        for sent in text:
            sent = replace(sent, '(),'+operators).split()
            aug_sent = []
            aug_sent.append('<S>')
            for word in sent:
                if word not in self.vocab:
                    aug_sent.append('<UNK>')
                else:
                    aug_sent.append(word)
            aug_sent.append('</S>')
            max_len = max(max_len, len(aug_sent))
            aug.append(aug_sent)

        for i in range(len(aug)):
            while len(aug[i]) < max_len:
                aug[i].append('<PAD>')
        return aug

    def text_to_sequence(self, sent):
        seq = []
        for word in sent:
            one_hot = np.zeros(len(self.vocab), dtype=np.float32)
            one_hot[self.word_to_index[word]] = 1.0
            seq.append(one_hot)
        seq = np.array(seq).T
        return seq

    def sequence_to_text(self, seq):
        text = []
        for i in range(len(seq.T)):
            ind = np.argmax(seq.T[i])
            word = self.index_to_word[ind]
            text.append(word)
        return ''.join(text)

    def preprocess(self, text):
        aug_text = self.aug_text(text)
        pro_text = [self.text_to_sequence(sent) for sent in aug_text]
        pro_text = np.array(pro_text)
        return pro_text
