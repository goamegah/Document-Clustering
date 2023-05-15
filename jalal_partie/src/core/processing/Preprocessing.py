import string
import re
import emoji
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class Preprocessing:
    def __init__(self):
        nltk.download('stopwords')

    @classmethod
    def lower_text(cls,text:str) -> str:
        return text.lower()

    @classmethod
    def remove_punctuation(cls,text:str) -> str:
        punctuationfree="".join([i for i in text if i not in string.punctuation])
        return punctuationfree

    @classmethod
    def remove_http(cls,text:str) -> str:
        return re.sub(r'http\S+', '', text)

    @classmethod
    def remove_hashtag(cls,text:str) -> str:
        return re.sub(r"#\S+", "",text)

    @classmethod
    def remove_at(cls,text:str) -> str:
        return re.sub("@\S+","",text)

    @classmethod
    def remove_emoji(cls,text:str) -> str:
        return emoji.replace_emoji(text, replace='')

    @classmethod
    def remove_number(cls,text:str) -> str:
        """
        :param text: text for preprocessing
        :param lg: stopwords language
        :param pos: pos for lemmatizer (verbs,nouns,...)
        :return: list of tokens
        """
        return re.sub(r"\S*[0-9]+\S*","",text)


    @classmethod
    def tokenize_text(cls,text:str) -> list:
        return [word for word in word_tokenize(text)]
    @classmethod
    def remove_stopwords(cls,tokens:list,lg="english") ->list:
        return [t for t in tokens if not t.lower() in stopwords.words(lg)]

    @classmethod
    def lematization(cls,tokens:list,pos="n") ->list:
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(word,pos=pos) for word in tokens]


    @classmethod
    def pipeline(cls,text:str,lg="english",pos="n") -> list:
        return \
            cls.lematization(
                cls.remove_stopwords(
                    cls.tokenize_text(
                        cls.remove_emoji(
                            cls.remove_http(
                                cls.remove_punctuation(
                                    cls.lower_text(text)
                                )
                            )
                        )
                    ),\
                lg=lg), \
            pos=pos)


