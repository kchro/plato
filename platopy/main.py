from folgen.fol_util import *
from nlgen_scripts.runfol import runfol
import random
import requests
from tqdm import tqdm

import time
import sys, os
import shutil

NUM_SENTENCES = 25

def get_post_request(fol):
    url = 'http://cypriot.stanford.edu:8080/ace/'
    payload = {
        'sig':fol,
        'blockfile':'rules.none'
    }

    response = requests.post(url, data=payload)
    if response.status_code == 200:
        return filter(lambda x: len(x) > 0, response.text.split('\n'))
    else:
        return []

def generate_atomic_sentences():
    with open('folgen/atomic_sents.out', 'w') as w:
        with open('folgen/atoms.out', 'r') as f:
            for line in tqdm(f):
                atom = line.rstrip()
                sents = get_post_request(atom)
                for sent in sents:
                    if len(sent) == 0:
                        continue
                    w.write(sent+'\t'+atom+'\n')

def generate_depth_k_formulas(k=0, filename='', dropout=0.1):
    tmpdir = '/dev/shm/'
    # tmpdir = './'

    # all trees depth <= k
    predicates = read_predicate_file('folgen/predicates.txt')
    operators = read_operator_file('folgen/operators.txt')

    atoms = [
        "%s(%s)" % (pred, ','.join(['%s' for _ in range(arity)]))
        for pred, arity in predicates
    ]

    for n in range(k):
        if n == 0:
            # write subtrees of depth 0
            with open(tmpdir+'total.out', 'w') as total:
                for atom in atoms:
                    total.write(atom+'\n')
            # write subtrees of depth 0 to prev
            with open(tmpdir+'prev.out', 'w') as prev:
                for atom in atoms:
                    prev.write(atom+'\n')
        else:
            print 'depth %d' % n

            with open(tmpdir+'curr.out', 'w') as curr:
                for op, arity in operators:
                    print op
                    if arity == 1:
                        # write to curr: negated subtrees of prev
                        with open(tmpdir+'prev.out', 'r') as prev:
                            for line in prev:
                                # dropout some of the negations
                                if random.random() < dropout*n:
                                    continue

                                subtree = line.rstrip()
                                curr.write('%s(%s)\n' % (op, subtree))
                    else:
                        # write to curr: joined subtrees of prev and total
                        with open(tmpdir+'prev.out', 'r') as prev:
                            for line in prev:
                                # dropout some of the prevs connections
                                if random.random() < dropout*n:
                                    continue

                                subtree_prev = line.rstrip()

                                for subtree_total in atoms:

                                    # dropout some of the prev-total connections
                                    if random.random() < dropout*n:
                                        continue

                                    if random.random() < 0.5:
                                        params = (subtree_prev, op, subtree_total)
                                    else:
                                        params = (subtree_total, op, subtree_prev)

                                    # curr.write('%s(%s,%s)\n' % (op, subtree_prev, subtree_total))
                                    if n == k-1:
                                        curr.write('%s%s%s\n' % params)
                                    else:
                                        curr.write('(%s%s%s)\n' % params)

                                # with open(tmpdir+'total.out', 'r') as total:
                                #     for line2 in total:
                                #
                                #         # dropout some of the prev-total connections
                                #         if random.random() < dropout*n:
                                #             continue
                                #
                                #         subtree_total = line2.rstrip()
                                #         if random.random() < 0.5:
                                #             params = (subtree_prev, op, subtree_total)
                                #         else:
                                #             params = (subtree_total, op, subtree_prev)
                                #
                                #         # curr.write('%s(%s,%s)\n' % (op, subtree_prev, subtree_total))
                                #         if n == k-1:
                                #             curr.write('%s%s%s\n' % (subtree_prev, op, subtree_total))
                                #         else:
                                #             curr.write('(%s%s%s)\n' % (subtree_prev, op, subtree_total))

            # overwrite prev.out with curr.out
            shutil.copy(tmpdir+'curr.out', tmpdir+'prev.out')

            # dump total.out into curr.out, and rename curr.out => total.out
            # faster because total.out << curr.out
            with open(tmpdir+'curr.out', 'a') as curr:
                with open(tmpdir+'total.out', 'r') as total:
                    for line in total:
                        curr.write(line)

            os.rename(tmpdir+'curr.out', tmpdir+'total.out')

def generate_formulas(filename, out='formulas.out'):
    constants = read_constant_file('folgen/constants.txt')

    with open(filename, 'r') as f:
        with open(out, 'w') as w:
            print 'loading...'
            lines = [line for line in f]
            print 'done.'

            for line in tqdm(lines):
                num_constants = line.count('%s')

                for _ in range(NUM_SENTENCES):
                    formula = line % tuple([random.choice(constants) for _ in range(num_constants)])
                    w.write(formula)

operators = ['&', '|', '-->', '<->']

def count_operators(formula):
    count = 0
    for op in operators:
        count += formula.count(op)
    return count

def get_polish_formula(formula):
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
                if formula[i:].startswith(op):
                    left = get_polish_formula(formula[:i])
                    operator = formula[i:i+len(op)]
                    right = get_polish_formula(formula[i+len(op):])
                    return '%s(%s,%s)' % (operator, left, right)

    print formula

# def generate_dataset(infile, outfile):
#     # operators = [op for op, _ in read_operator_file('folgen/operators.txt')]
#     # NOTE: hard coding: could be a problem if we extend operators
#     operators = ['~', '&', '|', '-->', '<->']
#
#     def count_operators(formula):
#         count = 0
#         for op in operators:
#             count += formula.count(op)
#         return count
#
#     def get_polish_formula(formula):
#         # base case:
#         # if no operators in formula
#         num_ops = count_operators(formula)
#         if num_ops == 0:
#             return formula
#         if num_ops == 1:
#             if formula[0] == '(' and formula[-1] == ')':
#                 formula = formula[1:-1]
#
#         # find the root operator
#         paren_count = 0
#         for i in range(len(formula)):
#             if formula[i] == '(':
#                 paren_count += 1
#             elif formula[i] == ')':
#                 paren_count -= 1
#             elif paren_count == 0:
#                 for op in operators:
#                     if op == '~':
#                         continue
#                     if formula[i:].startswith(op):
#                         left = get_polish_formula(formula[:i])
#                         operator = formula[i:i+len(op)]
#                         right = get_polish_formula(formula[i+len(op):])
#                         return '%s(%s,%s)' % (operator, left, right)
#
#         print formula
#
#     with open(outfile, 'w') as w:
#         with open(infile, 'r') as f:
#             start = time.time()
#             print 'loading...'
#             formulas = [line.rstrip() for line in f]
#             formulas = [random.choice(formulas) for _ in range(10000)]
#             # formulas = [f.readline().rstrip() for _ in range(100)]
#             print 'done. %0.3f' % float(time.time() - start)
#
#             for fol in tqdm(formulas):
#                 nl_sents = get_post_request(fol)
#                 if not nl_sents:
#                     continue
#                 polish = get_polish_formula(fol)
#                 if not polish:
#                     continue
#                 for sent in nl_sents:
#                     w.write('%s\t%s\t%s\n' % (sent, fol, polish))

if __name__ == '__main__':
    """
    generate all the formulas of depth k
    """
    # generate_depth_k_formulas(k=2, filename='k2_formulas.out', dropout=0)
    generate_depth_k_formulas(k=3, filename='k3_formulas.out', dropout=0)

    """
    from the templates in total.out, make arbitrary formulas
    """
    generate_formulas('/dev/shm/total.out', out='/dev/shm/k3_tree.in')
    generate_dataset('/dev/shm/k3_tree.in', '/dev/shm/k3_tree.out')

    """
    from the full sentences, make posts
    """
    # with open('../data/raw/k3_tree.in', 'r') as f:
    #     print 'loading...'
    #     formulas = [line.rstrip() for line in f]
    #     print 'done.'
    #
    #     with open('../data/raw/k3_tree.out', 'w') as w:
    #         for formula in tqdm(formulas[50000:50000]):
    #             # try:
    #             nl_sents = get_post_request(formula)
    #
    #             for sent in nl_sents[:10]:
    #                 w.write('%s\t%s\n' % (sent, formula))
    #
    #             # except:
    #             #     with open('failed_sents.out', 'a') as err:
    #             #         err.write(formula+'\n')
