from utils import convert_string_to_formula

class Answer:
    def __init__(self, text):
        self.text = text
        try:
            self.formula = convert_string_to_formula(text)
