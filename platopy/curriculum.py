"""
Curriculum

This defines what each of the 'levels' are.

Levels are defined to be the rules we want to test the student on.
    - in the original Java code, this was left unimplemented.
    I believe that the original formulation was to have a Policy,
    which has a ruleset associated with it. Similarly, we have
    a Level, which can be configured with a ruleset.

    - The next step would be to design a curriculum such that the
    Levels are of increasing difficulty.
"""
LEVELS = ['agg', 'all', 'conn', 'ellip', 'mod', 'one', 'part', 'pro']

class Level:
    def __init__(self, name, level_num=None, level_rep=1):
        """
        @params:
        name        (str) the type of problem
        level_num   (int) the level number
        level_rep   (int) the number of repetitions to complete level
        """
        self.name = name
        self.level_num = level_num
        self.rulefile = 'rules.%s' % name
        self.level_rep = level_rep

class Curriculum:
    def __init__(self, level_rep=10):
        self.levels = [
            Level(name, level_num=i, level_rep=level_rep)
            for i, name in enumerate(LEVELS)
        ]

        self.num_levels = len(self.levels)

    def get_progress_report(self, user):
        progress_report = {
            'username': user['name'],
            'level': user['level'],
            'progress': user['progress'],
            'total_levels': self.num_levels,
            'total_progress': self.levels[user['level']].level_rep
        }

        return progress_report

    def get_rulefile(self, level_num):
        level = self.levels[level_num]
        return level.rulefile

    def get_level_rep(self, level_num):
        level = self.levels[level_num]
        return level.level_rep
