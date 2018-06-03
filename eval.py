import argparse
# models
from models.seq2seq import Seq2Seq
from models.seq2tree import Seq2Tree
from models.tree2seq import Tree2Seq
from models.tree2tree import Tree2Tree

# tools
from data.load_data import load_file
from sklearn.model_selection import train_test_split

import torch
import json

MODELS = {
    'seq2seq': Seq2Seq,
    'seq2tree': Seq2Tree,
    'tree2seq': Tree2Seq,
    'tree2tree': Tree2Tree
}

DATASETS = {
    'toy': 'k3_tree_mini.out',
    'sm': 'atomic_sents.out',
    'md': 'k3_med.out',
    'lg': 'k3_tree.out'
}

def get_parser():
    '''
    Set up argument parser
    Returns:
        parser: (ArgumentParser) the created parser
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_only',
                        action='store_true')
    parser.add_argument('-e', '--encoder',
                        required=True,
                        choices={'seq', 'tree'})
    parser.add_argument('-d', '--decoder',
                        required=True,
                        choices={'seq', 'tree'})
    parser.add_argument('-D', '--data',
                        required=True,
                        choices={'toy', 'sm', 'md', 'lg'})
    return parser

def get_model_name(args):
    return '%s2%s' % (args.encoder, args.decoder)

def get_dataset_name(args):
    if args.data in DATASETS:
        return DATASETS[args.data]
    return DATASETS['toy']

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print 'running on the %s' % device
    sess = raw_input('load session: ')

    # parse arguments
    parser = get_parser()
    args = parser.parse_args()

    # model name
    name = get_model_name(args)
    print 'running the %s model' % name

    # select dataset
    dataset = get_dataset_name(args)
    print 'using the %s dataset' % dataset

    # load data
    inputs, vocabs = load_file(filename=dataset,
                               encoder=args.encoder,
                               decoder=args.decoder,
                               device=device)

    src_inputs, tar_inputs = inputs
    src_vocab, tar_vocab = vocabs

    # split data
    X_train, X_test, y_train, y_test = train_test_split(src_inputs, tar_inputs, test_size=0.1)

    print '%d training examples & %d testing examples.' % (len(X_train), len(X_test))

    # load the model parameters
    input_size = len(src_vocab)
    hidden_size = 200
    output_size = len(tar_vocab)
    model = MODELS[name](input_size=input_size,
                         hidden_size=hidden_size,
                         output_size=output_size,
                         src_vocab=src_vocab,
                         tar_vocab=tar_vocab,
                         sess=sess, device=device)

    # load the saved model
    print 'loading model parameters...',
    model.load('%s.json' % sess)
    print 'done.'

    preds = model.predict(X_test)
    model.evaluate(X_test, y_test, preds, out='tmp')

    raise

    # enter the input string
    while True:
        src_input = raw_input('enter sentence (Q to quit): ')
        if src_input == 'Q':
            break

        x_test = src_vocab.get_idx_tensor([src_input])
        # make the prediction
        print 'running the model on test set...'
        preds = model.predict(x_test)
        print 'done.'

        print 'input: ' + src_input
        print 'output: ' + tar_vocab.reverse(preds[0])
