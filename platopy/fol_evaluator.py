from nltk.sem.logic import *

class FOLEvaluator:
    def preprocess(self, formula):
        # & | <-> are okay
        formula = formula.replace('-->', ' -> ')
        formula = formula.replace('~', '!')
        return formula

    def check_formula(self, answer, submission):
        answer = Expression.fromstring(self.preprocess(answer))
        submission = Expression.fromstring(self.preprocess(submission))
        return answer.equiv(submission)
