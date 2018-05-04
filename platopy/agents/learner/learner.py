class Learner:
    def __init__(self, profile=None):
        self.achieved_goals = []
        self.streak = 0
        self.profile = profile

    def get_streak(self):
        return self.streak

    def set_streak(self, streak):
        self.streak = streak

    def increment_streak(self):
        self.streak += 1

    def reset_streak(self):
        self.streak = 0

    def is_competent(self, learning_goal):
        return learning_goal in self.achieved_goals

    def set_competent(self, learning_goal, competent):
        if competent:
            if learning_goal in self.achieved_goals:
                return
            self.achieved_goals.append(learning_goal)
        else:
            self.achieved_goals.remove(learning_goal)
