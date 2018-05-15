import argparse
import os
import os.path
import subprocess

from scripts.e2e import e2e

dir = '~/Desktop/plato/platopy/nlgen_scripts'
dat_dir = '%s/dat' % dir
rules_dir = '%s/rules' % dir

def get_parser():
    '''
    Set up argument parser
    Returns:
        parser: (ArgumentParser) the created parser
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fol', type=str, required=True,
                        help='FOL string')
    parser.add_argument('-l', '--rule_list', type=str, nargs='*',
                        help='Rule List')
    parser.add_argument('-r', '--rule_file', type=str, nargs='*',
                        help='Rule File')
    parser.add_argument('-b', '--block_list', type=str, nargs='*',
                        help='Blocked rule list')
    parser.add_argument('-x', '--block_file', type=str, nargs='*',
                        help='Blocked rule file')
    parser.add_argument('-a', '--add_file', type=str, nargs='*',
                        help='Added rule file')
    return parser

def get_rules(filename):
    if not os.path.exists(rules_dir+filename):
        return []

    with open(rules_dir+filename, 'r') as f:
        rules = [line for line in f]
    return rules

def runfol(fol, rule_file='rules.all',
           add_file=None,
           rule_list=None,
           block_file=None,
           block_list=None):
    """
    @params:
        fol         string
        rules       strings
    @returns:
        out         [strings]
    """

    rules = set()

    # add rules to set:
    add_rules = []
    if rule_file:
        add_rules += rule_file
    if add_file:
        add_rules += add_file
    if rule_list:
        add_rules += rule_list

    for rule_file in add_rules:
        rules.update(get_rules(rule_file))

    # del rules from set
    del_rules = []
    if block_file:
        del_rules += block_file
    if block_list:
        del_rules += block_list

    for rule_file in del_rules:
        rules.difference_update(get_rules(rule_file))

    # write out the rules in temp file
    with open('rules.tmp', 'w') as f:
        for rule in sorted(rules):
            f.write(rule)

    command = 'python %s/scripts/e2e.py "%s" | ' % (dir, fol) + \
              'ace -g %s/inflatemrs.dat -f | ' % dat_dir + \
              'ace -g %s/paraphrase-op.dat --transfer-config rules.tmp | ' % dat_dir + \
              'ace -g %s/ergopen.dat -e' % dat_dir

    stdout = subprocess.check_output([command], shell=True)
    sents = set(filter(lambda x: len(x), stdout.split('\n')))
    out = []
    for sent in sents:
        out.append(sent.rsplit('.', 1)[0])
    return out

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    fol = args.fol

    rules = set()

    # add rules to set:
    add_rules = []
    if args.rule_file:
        add_rules += args.rule_file
    if args.add_file:
        add_rules += args.add_file
    if args.rule_list:
        add_rules += args.rule_list

    for rule_file in add_rules:
        rules.update(get_rules(rule_file))

    # del rules from set
    del_rules = []
    if args.block_file:
        del_rules += args.block_file
    if args.block_list:
        del_rules += args.block_list

    for rule_file in del_rules:
        rules.difference_update(get_rules(rule_file))

    # write out the rules in temp file
    with open('rules.tmp', 'w') as f:
        for rule in sorted(rules):
            f.write(rule)

    command = 'python %s/scripts/e2e.py "%s" | ' % (dir, fol) + \
              'ace -g %s/inflatemrs.dat -f | ' % dat_dir + \
              'ace -g %s/paraphrase-op.dat --transfer-config rules.tmp | ' % dat_dir + \
              'ace -g %s/ergopen.dat -e' % dat_dir

    stdout = subprocess.check_output([command], shell=True)
    sents = set(filter(lambda x: len(x), stdout.split('\n')))
    out = []
    for sent in sents:
        out.append(sent.rsplit('.', 1)[0])

    # write the sentences to out
    with open('dataset.txt', 'a') as f:
        for line in out:
            f.write(line+'\n')
