import random

dir = '/scratch/danf/openproof/Server/platopy/'

class FOLGenerator:
    def __init__(self, filename=''):
        with open(dir+filename, 'r') as f:
            self.templates = list(set([line.split('\t')[1] for line in f]))
        self.constants = 'abcdefxyz'

    def get_formula(self):
        rand_template = random.choice(self.templates)
        num_constants = rand_template.count('%s')
        constants = [random.choice(self.constants) for _ in range(num_constants)]
        return rand_template % tuple(constants)
