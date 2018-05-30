from folgen.fol_util import *
from nlgen_scripts.runfol import runfol
import random
import requests
from tqdm import tqdm
import sys

NUM_SENTENCES = 10

# def generate_depth_k_sentences(k=1):
#     # build the template formulas
#     predicates = read_predicate_file('folgen/predicates.txt')
#     constants = read_constant_file('folgen/constants.txt')
#     preds = tuple([
#         "%s(%s)" % (pred, ','.join(['%s' for _ in range(arity)]))
#         for pred, arity in predicates
#     ])
#
#     operators = read_operator_file('folgen/operators.txt')
#
#     # memoize results
#     memo = {}
#
#     def recurse(depth):
#         if depth == 0:
#             return preds
#
#         if depth in memo:
#             return memo[depth]
#
#         # generate all depth k trees
#         if depth-1 not in memo:
#             subtrees = recurse(depth-1)
#             memo[depth-1] = tuple(subtrees)
#         else:
#             subtrees = memo[depth-1]
#
#         ret = []
#         for operator, arity in operators:
#             if arity == 1:
#                 for i in range(len(subtrees)):
#                     if depth == k:
#                         ret.append('%s%s' % (operator, subtrees[i]))
#                     else:
#                         ret.append('%s(%s)' % (operator, subtrees[i]))
#             if arity == 2:
#                 for i in range(len(subtrees)):
#                     for j in range(len(subtrees)):
#                         if depth == k:
#                             ret.append('%s%s%s' % (subtrees[i], operator, subtrees[j]))
#                         else:
#                             ret.append('(%s%s%s)' % (subtrees[i], operator, subtrees[j]))
#
#         memo[depth] = ret
#         return ret
#
#     depth_k_trees = recurse(k)
#     for tree in depth_k_trees[:5]:
#         print tree
#
#     count = 0
#     for tree in depth_k_trees:
#         count += (tree.count('%s') * NUM_SENTENCES)
#
#     # get
#     arg_train = [constants[i%len(constants)] for i in range(count)]
#     random.shuffle(arg_train)
#
#     with open('folgen/k%d_sents.out' % k, 'w') as w:
#         for tree in tqdm(depth_k_trees):
#             n = tree.count('%s')
#             for i in range(NUM_SENTENCES):
#                 formula = tree % tuple(arg_train[:n])
#                 arg_train = arg_train[n+1:]
#                 sents = get_post_request(formula)
#                 for sent in set(sents):
#                     if len(sent) == 0:
#                         continue
#                     w.write(sent+'\t'+formula+'\n')

def get_post_request(fol):
    url = 'http://cypriot.stanford.edu:8080/ace/'
    payload = {
        'sig':fol,
        'rulefile':'rules.all',
        'blockfile':'rules.none'
    }

    response = requests.post(url, data=payload)
    if response.status_code == 200:
        return response.text.split('\n')
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

def generate_depth_k_sentences(k=0, filename='', dropout=0.1):
    tmpdir = '/dev/shm/'

    # all trees depth <= k
    predicates = read_predicate_file('folgen/predicates.txt')
    constants = read_constant_file('folgen/constants.txt')
    operators = read_operator_file('folgen/operators.txt')

    atoms = [
        "%s(%s)" % (pred, ','.join(['%s' for _ in range(arity)]))
        for pred, arity in predicates
    ]

    prev = atoms
    curr = []

    for n in range(k+1):
        if n == 0:
            # write subtrees of depth 0
            with open(tmpdir+'total.out', 'a') as total:
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
                                subtree = line.rstrip()
                                curr.write('%s(%s)\n' % (op, subtree))
                    else:
                        # write to curr: joined subtrees of prev and total
                        with open(tmpdir+'prev.out', 'r') as prev:
                            for line in prev:
                                subtree_prev = line.rsplit()
                                with open(tmpdir+'total.out', 'r') as total:
                                    for line2 in total:

                                        # dropout some of the prev-total connections
                                        if random.random() < dropout:
                                            continue

                                        subtree_total = line2.rsplit()
                                        if n == k:
                                            curr.write('%s%s%s\n' % (subtree_prev, op, subtree_total))
                                            # reverse order
                                            curr.write('%s%s%s\n' % (subtree_total, op, subtree_prev))
                                        else:
                                            curr.write('(%s%s%s)\n' % (subtree_prev, op, subtree_total))
                                            # reverse order
                                            curr.write('(%s%s%s)\n' % (subtree_total, op, subtree_prev))

            with open(tmpdir+'curr.out', 'r') as curr:
                with open(tmpdir+'total.out', 'a') as total:
                    with open(tmpdir+'prev.out', 'w') as prev:
                        for line in curr:
                            total.write(line)
                            prev.write(line)

if __name__ == '__main__':
    """
    generate all the formulas of depth k
    """
    # generate_depth_k_sentences(k=2, filename='k2_formulas.out', dropout=0)

    generate_depth_k_sentences(k=3, filename='k3_formulas.out', dropout=0.4)

    # with open('k2_formulas.out', 'w') as w:
    #     for formula in generate_depth_k_sentences(k=2):
    #         w.write(formula+'\n')
    #
    # with open('k3_formulas.out', 'w') as w:
    #     for formula in generate_depth_k_sentences(k=3):
    #         w.write(formula+'\n')

    # predicates = read_predicate_file('folgen/predicates.txt')
    # constants = read_constant_file('folgen/constants.txt')
    # atoms = generate_all_atoms(predicates, constants)
    # with open('folgen/atoms.out', 'w') as w:
    #     for atom in atoms:
    #         w.write('%s\n' % atom)

    # generate_all_sentences()
    # generate_atomic_sentences()
    # generate_depth_k_sentences(k=1)
    # print get_post_request('(tet(FORMAT)&tet(FORMAT))')
