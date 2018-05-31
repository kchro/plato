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

models = {
    'seq2seq': Seq2Seq,
    'seq2tree': Seq2Tree,
    'tree2seq': Tree2Seq,
    'tree2tree': Tree2Tree
}

DATA = ''

def get_parser():
    '''
    Set up argument parser
    Returns:
        parser: (ArgumentParser) the created parser
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',
                        action='store_true')
    parser.add_argument('--eval_only',
                        action='store_true')
    parser.add_argument('-e', '--encoder',
                        required=True,
                        choices={'seq', 'tree'})
    parser.add_argument('-d', '--decoder',
                        required=True,
                        choices={'seq', 'tree'})
    return parser

def get_model_name(args):
    return '%s2%s' % (args.encoder, args.decoder)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print 'running on the %s' % device
    sess = raw_input('session name: ')

    # parse arguments
    parser = get_parser()
    args = parser.parse_args()

    # model name
    name = get_model_name(args)

    # load data
    inputs, vocabs = load_file(filename='k3_tree_mini.out',
                               encoder=args.encoder,
                               decoder=args.decoder,
                               device=device)

    src_inputs, tar_inputs = inputs
    src_vocab, tar_vocab = vocabs

    # split data
    X_train, X_test, y_train, y_test = train_test_split(src_inputs, tar_inputs, test_size=0.1)

    # load the model parameters
    input_size = len(src_vocab)
    hidden_size = 200
    output_size = len(tar_vocab)
    model = models[name](input_size=input_size,
                         hidden_size=hidden_size,
                         output_size=output_size,
                         sess=sess, device=device)
    model.set_vocab(src_vocab, tar_vocab)

    # model.train(X_train, y_train)
    preds = model.predict(X_test)

    nl_sents = [src_vocab.reverse(nl_sent) for nl_sent in X_test]
    fol_forms = [fol_form for fol_form in y_test]
    fol_preds = [tar_vocab.reverse(fol_pred) for fol_pred in preds]

    print 'logging...',
    with open('logs/sessions/%s.out' % sess.replace(' ', '_'), 'w') as w:
        for nl_sent, fol_form, fol_pred in zip(nl_sents, fol_forms, fol_preds):
            w.write('%s\t%s\t%s\t\n' % (nl_sent, fol_form, fol_pred))
    print 'done.'
