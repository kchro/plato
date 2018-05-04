rule_set = {
    'none': 'rules.one',
    'all': 'rules.all',
    'pronouns': 'rules.pro',
    'ellipses': 'rules.ellip',
    'aggregation': 'rules.agg',
    'modification': 'rules.mod',
    'partitives': 'rules.part',
    'connectives': 'rules.conn'
}

class RuleSet:
    def __init__(self, display_name, filename):
        self.display_name = display_name
        self.filename = filename

    def set_filename(self, filename):
        self.filename = filename

    def get_filename(self):
        return self.filename

    def __str__(self):
        return self.display_name
