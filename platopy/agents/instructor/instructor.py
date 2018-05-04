class InstructorFace:
    def __init__(self,
                 learning_goal=None,
                 policies=(None,None,None),
                 learner=None):
        self.learning_goal = learning_goal


    def get_learning_goal():
        return self.learning_goal

    def set_learning_goal(learning_goal):
        self.learning_goal = learning_goal

    def increase_difficulty(learner):
        raise NotImplemented

    def get_FOL_policy():
        raise NotImplemented

    def get_NL_policy():
        raise NotImplemented

    def get_feed_policy():
        raise NotImplemented

    def get_learner():
        raise NotImplemented

    def get_next_target_for_learner():
        raise NotImplemented

    def learner_wants_more_practice():
        raise NotImplemented

    def prompt_for_increased_complexity():
        raise NotImplemented

    def new_answer(result, problem):
        raise NotImplemented

    def generate_new_problem(fol_gen, nl_gen):
        raise NotImplemented

    def get_learner_progress():
        raise NotImplemented

    def set_problem(problem):
        raise NotImplemented

    def get_problem():
        raise NotImplemented
