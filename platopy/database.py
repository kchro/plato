from curriculum import Curriculum
import json

class Database:
    def __init__(self):
        self.db = {}
        self.add_user('guest')
        self.curriculum = Curriculum()

    def has_user(self, username):
        return username in self.db

    def add_user(self, username):
        user = {}
        user['name'] = username
        user['level'] = 1
        user['progress'] = 0
        user['feedback'] = 'flag' # 'backtranslate', 'recast'
        self.db[username] = user

    def get_user_profile(self, username):
        user = self.db[username]
        progress_report = self.curriculum.get_progress_report(user)
        progress_report = json.dumps(progress_report)
        return progress_report

    def update_user(self, username, attr, val):
        self.db[username][attr] = val

    def get_user_attr(self, username, attr):
        return self.db[username][attr]

    def get_user_rulefile(self, username):
        level_num = self.db[username]['level']
        return self.curriculum.get_rulefile(level_num)

    def increment_user_progress(self, username):
        self.db[username]['progress'] += 1

        level_num = self.db[username]['level']
        total_progress = self.curriculum.get_level_rep(level_num)
        if self.db[username]['progress'] == total_progress:
            self.db[username]['level'] += 1
            self.db[username]['progress'] = 0
