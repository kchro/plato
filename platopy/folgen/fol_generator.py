import random
from fol_util import *

class FOLGenerator:
    def __init__(self, static=True):
        pass

    def generate_FOL(self, policy):
        pass

    def generate_atom(self, policy, vocab):
        pass

    def generate_term(self, policy, vocab, symbol_table):
        # get constant strings from symtab
        # see getConstants() in OPHPSymbolTable.java for details
        constants = symbol_table.get_constants()
        rand_constant = random.choice(constants)
        return rand_constant

if __name__ == '__main__':
    predicates = read_predicate_file('predicates.txt')
    constants = read_constant_file('constants.txt')
    atoms = generate_all_atoms(predicates, constants)
    with open('atoms.out', 'w') as w:
        for atom in atoms:
            w.write('%s\n' % atom)
