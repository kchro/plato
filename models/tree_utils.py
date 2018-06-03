import spacy
import re

nlp = spacy.load('en')

class Tree(object):
    def __init__(self, val=None, children=None):
        self.val = val
        if children:
            self.children = children
        else:
            self.children = []

class FOLTree(Tree):
    def __init__(self, val=None, formula=None):
        super(FOLTree, self).__init__(val)
        if formula:
            root = self.parse(formula)
            self.val = root.val
            self.children = root.children

    def parse(self, formula):
        """
        parse the (polish) FOL formula tree
        """
        m = re.search('(.*?)\\((.*)\\)', formula)

        if m:
            root = FOLTree(val=m.group(1))
            subformula = m.group(2)
            splits = []
            paren_count = 0
            start = 0
            for end in range(len(subformula)):
                if subformula[end] == '(':
                    paren_count += 1
                elif subformula[end] == ')':
                    paren_count -= 1
                elif paren_count == 0:
                    if subformula[end] == ',':
                        splits.append(subformula[start:end])
                        start = end + 1

            splits.append(subformula[start:end + 1])
            root.children = [ self.parse(sub) for sub in splits ]
        else:
            root = FOLTree(val=formula)

        return root

    def flatten(self):
        """
        flatten a tree into string
        """
        if len(self.children) == 0:
            return self.val
        params = (',').join([ child.flatten() for child in self.children ])
        return '%s(%s)' % (self.val, params)

    def inorder(self):
        """
        generate the sequence inorder (include the non-terminals)
        """
        if len(self.children) == 0:
            return ['%s' % self.val]
        params = (' , ').join([' <N> '] * len(self.children))
        inorder = ['%s ( %s ) ' % (self.val, params)]
        for child in self.children:
            inorder += child.inorder()

        return inorder

    def __str__(self):
        """
        print a tree in hierarchical structure
        """
        if len(self.children) == 0:
            return self.val
        ret = [
         self.val]
        for child in self.children:
            ret += [ '\t' + child_s for child_s in str(child).split('\n') ]

        return ('\n').join(ret)

class NoRootFoundException(Exception):
    pass

class DepTree(Tree):
    def __init__(self, val=None, sent=None, node=None):
        super(DepTree, self).__init__(val)
        if sent:
            doc = nlp(unicode(sent))
            node = self.get_root(doc)
        if node:
            self.val = node.text
            self.children = []
            for child in node.children:
                self.children.append(DepTree(node=child))

    def get_root(self, doc):
        for token in doc:
            if token.dep_ == 'ROOT':
                return token
        raise NoRootFoundException

    def print_level_order(self):
        queue = [self, None]
        curr_level = []
        while queue:
            curr = queue.pop(0)
            if curr is None:
                print curr_level
                curr_level = []
                if queue:
                    queue.append(None)
            else:
                curr_level.append(curr.val)
                for child in curr.children:
                    queue.append(child)

    def flatten(self):
        if len(self.children) == 0:
            return self.val
        params = ' '.join([ child.flatten() for child in self.children ])
        return self.val + ' ' + params
