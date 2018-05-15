def generate_all_atoms(predicates, constants):
    """
    @params:
        predicates [strings]
        constants [strings]
    @return:
        generated_atoms [strings]
    """
    # maintain a set of all atom formulas
    generated_atoms = set()

    def recurse(pred, arity, params):
        if len(params) == arity:
            # base case: when params is fully populated
            formula = '%s(%s)' % (pred, ','.join(params))
            generated_atoms.add(formula)
            return

        for const in constants:
            # recursive: add a const until the params are filled
            recurse(pred, arity, params+[const])

    for pred, arity in predicates:
        recurse(pred, arity, [])

    return list(generated_atoms)

def read_predicate_file(filename):
    with open(filename, 'r') as f:
        predicates = []
        for line in f:
            pred, arity = line.split(',')
            arity = int(arity)
            predicates.append((pred, arity))
    return predicates

def read_file(filename):
    with open(filename, 'r') as f:
        arr = [line.rstrip() for line in f]
    return arr

def read_constant_file(filename):
    return read_file(filename)

def read_atom_file(filename='atoms.out'):
    return read_file(filename)

def read_operator_file(filename):
    return read_file(filename)
