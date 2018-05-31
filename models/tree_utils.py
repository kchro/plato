import re

class Tree:
    def __init__(self, val=None, formula=None):
        if val:
            self.val = val
            self.children = []
        if formula:
            root = self.parse(formula)
            self.val = root.val
            self.children = root.children

    def parse(self, formula):
        m = re.search('(.*?)\\((.*)\\)', formula)
        if m:
            if m.group(1) not in '&|$~':
                return Tree(val=m.group(0))
            root = Tree(val=m.group(1))
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
            return root
        return Tree(val=formula)

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
