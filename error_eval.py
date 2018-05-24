import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename',
                        type=str,
                        required=True,
                        help='filename')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    with open(args.filename, 'r') as f:
        for line in f:
            _, gold, pred = line.split('\t')
            if gold != pred:
                print gold, pred

    print args.filename
