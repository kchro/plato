import random
from time import sleep

class Dialog(object):
    def __init__(self, role='', messages=None, condition=None):
        self.role = role
        self.messages = messages
        self.condition = condition

    def choice(self):
        message = random.choice(self.messages)
        while True:
            start = message.find('(')
            end = message.find(')')
            if start == -1 or end == -1:
                break
            phrases = message[start+1:end].split('|')
            phrase = random.choice(phrases)
            message = message[:start] + phrase + message[end+1:]
        return message

class Plato:
    def __init__(self):
        self.dialogs = {
            'hello': Dialog(role='hello', messages=[
                "(Hello,|Hi,) (my name is|I am|I'm) Plato.",
                "Welcome. (My name is|I am|I'm) Plato.",
            ]),
            'nl2fol_init': Dialog(role='nl2fol_init', messages=[
                "Let's (do|try|practice) some translation exercises.",
                "Why don't we (do|try|start|practice) some translation exercises?",
                "Shall we (do|try|start|begin) some translation exercises?"
            ]),
            'button': Dialog(role='button', messages=[
                'Please press the "New Problem" button to get started.'
            ]),
            'nl2fol_instructions': Dialog(role='nl2fol_instructions', messages=[
                'Please translate this sentence into First-Order Logic.'
            ]),
            'correct': Dialog(role='correct', messages=[
                'Correct!', "That's (correct|right).", "Very (good|nice).",
                'Excellent.', "Perfect.", "Right."
            ]),
            'incorrect': Dialog(role='incorrect', messages=[
                "(No, that is|No, that's|That is|That's) (incorrect|wrong|not correct|not the answer I was looking for).",
                "No, that is incorrect."
            ]),
            'nl2fol_again': Dialog(role='tryagain', messages=[
                "(Shall while|Do you want to ) (try|do) (another|one more)(?| problem?| one?)",
                "(Do you want to|Would you like to) (keep going|continue)?"
            ]),
            'level_up': Dialog(role='level_up', messages=[
                "Nice! You have leveled up."
            ]),
            'backtranslate': Dialog(role='backtranslate', messages=[
                "Here is what your answer would look like in English."
            ]),
            'recast': Dialog(role='recast', messages=[
                "Let me rephrase the prompt."
            ])
        }

    def get_message(self, roles):
        messages = [self.dialogs[role].choice() for role in roles]
        return '\n'.join(messages)

if __name__ == '__main__':
    print Plato().get_message(['hello', 'incorrect', 'backtranslate'])
