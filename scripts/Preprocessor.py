import string
import re
class Preprocessor:
    def __init__(self):
        pass

    @staticmethod
    def remove_punctuation(text:str) -> str:
        punctuationfree="".join([i for i in text if i not in string.punctuation])
        return punctuationfree
    @staticmethod
    def lower_text(text:str) -> str:
        return text.lower()

    @staticmethod
    def lower_text(text:str) -> str:
        return text.lower()

