import argparse
from models.seq2seq import Seq2Seq
from models.seq2tree import Seq2Tree
from models.tree2seq import Tree2Seq
from models.tree2tree import Tree2Tree

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
    # parser.add_argument('--num_epochs',
    #                     required=True,
    #                     type=int)
    # parser.add_argument('--num_iterations',
    #                     required=True,
    #                     type=int)
    # parser.add_argument('--num_layers',
    #                     required=True,
    #                     type=int)
    # parser.add_argument('--num_dim',
    #                     required=True,
    #                     type=int)
    return parser

def get_model_name(args):
    return '%s2%s' % (args.encoder, args.decoder)

if __name__ == '__main__':
    # parse arguments
    parser = get_parser()
    args = parser.parse_args()

    name = get_model_name(args)

    # load data



    model = models[name]
