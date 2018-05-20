from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    source = []
    target = []
    with open('data/raw/k1_sents.out', 'r') as f:
        for line in f:
            sent, atom = line.rstrip().split('\t')
            source.append(sent.replace('|', '').replace('.', '').lower())
            target.append(atom)
    X = {}
    y = {}
    X['traindev'], X['test'], y['traindev'], y['test'] = train_test_split(source[:100], target[:100], test_size=0.1)
    X['train'], X['dev'], y['train'], y['dev'] = train_test_split(X['traindev'], y['traindev'], test_size=0.3)

    for dataname in ['train', 'dev', 'test']:
        with open('data/raw/k1_sents_%s.tsv' % dataname, 'w') as w:
            for xi, yi in zip(X[dataname], y[dataname]):
                w.write('%s\t%s\n' % (xi, yi))
