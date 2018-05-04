symbol_table = None

class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def convert_string_to_formula(text):
    # parse text
    # set symbol_table
    # return one wff with binaryJuncts
