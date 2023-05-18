# -*- coding: utf-8 -*-

__author__ = "Godwin AMEGAH"
__license__ = "GPL"

import re
import nltk

nltk.download('punkt')  # At first, you have to download these nltk packages.
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = stopwords.words('english')  # defining stop_words
# stop_words.remove('not') # removing not from the stop_words list as it contains value in
# negative movies
lemmatizer = WordNetLemmatizer()


def data_preprocessing(document):
    # data cleaning
    document = re.sub(re.compile('<.*?>'), '', document)  # removing html tags
    document = re.sub('[^A-Za-z0-9]+', ' ', document)  # taking only words

    # lowercase
    document = document.lower()

    # tokenization
    tokens = nltk.word_tokenize(document)  # converts review to tokens

    # stop_words removal
    document = [word for word in tokens if word not in stop_words]  # removing stop words

    # lemmatization
    document = [lemmatizer.lemmatize(word) for word in document]

    # join words in preprocessed review
    document = ' '.join(document)

    return document
