import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd

nltk.download('punkt')  # At first, you have to download these nltk packages.
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


class Corpus:
    def __init__(
            self,
    ):
        self.stop_words = stopwords.words('english')  # defining stop_words
        # stop_words.remove('not') # removing not from the stop_words list as it contains value in negative movies
        self.lemmatizer = WordNetLemmatizer()

    def _process_document(self, document):
        # data cleaning
        document = re.sub(re.compile('<.*?>'), '', document)  # removing html tags
        document = re.sub('[^A-Za-z0-9]+', ' ', document)  # taking only words

        # lowercase
        document = document.lower()

        # tokenization
        tokens = nltk.word_tokenize(document)  # converts review to tokens

        # stop_words removal
        document = [word for word in tokens if word not in self.stop_words]  # removing stop words

        # lemmatization
        document = [self.lemmatizer.lemmatize(word) for word in document]

        # join words in preprocessed review
        document = ' '.join(document)

        return document

    def process_documents(self, frame) -> pd.DataFrame:
        # On applique le pré-traitement à nos données
        _processed_documents = frame.copy()
        _processed_documents['text'] = frame['text'].apply(lambda text: self._process_document(text))
        return _processed_documents
