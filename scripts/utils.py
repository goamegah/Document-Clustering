#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Godwin AMEGAH"
__license__ = "GPL"

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

from pathlib import Path
import pandas as pd
import zipfile  # use tarfile for tgz file
import urllib.request
import datetime


def download_embedding_data():
    zip_path = Path("datasets/glove.840B.300d.zip")
    if not zip_path.is_file():
        Path("../core/datasets").mkdir(parents=True, exist_ok=True)
        url = "https://nlp.stanford.edu/data/glove.840B.300d.zip"
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path) as embedding_zip:
            embedding_zip.extractall(path="../core/datasets")
            for info in embedding_zip.infolist():
                print(f"Filename: {info.filename}")
                print(f"Modified: {datetime.datetime(*info.date_time)}")
                print(f"Normal size: {info.file_size} bytes")
                print(f"Compressed size: {info.compress_size} bytes")
                print("-" * 20)
    print("*" * 20)
    print(" Done! ")
    print("*" * 20)
    # return pd.read_csv(Path("datasets/housing.csv"))


def ndocs_label(df_doc: pd.DataFrame) -> {str: int}:
    """
    Count number of document per label
    :param df_doc:
    :return:
    """
    all_labels = list(df_doc.label)
    docs_counter = Counter(all_labels)  # -> dico
    return docs_counter


def text2vec(model, vectorised_matrice, vectoriser):
    """

    :param model:
    :param vectorised_matrice:
    :param vectoriser:
    :return: pd.Dataframe
    """
    # create words dataframe
    df_vec_data = pd.DataFrame(vectorised_matrice.toarray(), columns=vectoriser.get_feature_names())

    # Creating the list of words which are present in the Document term matrix
    words_vocab = df_vec_data.columns[:-1]

    # Creating empty dataframe to hold sentences
    df_w2v_data = pd.DataFrame()

    # Looping through each row for the data
    for i in range(df_vec_data.shape[0]):

        # initiating a sentence with all zeros
        sentence = np.zeros(300)

        # Looping through each word in the sentence and if its present in
        # the Word2Vec model then storing its vector
        for word in words_vocab[df_vec_data.iloc[i, :] >= 1]:
            # print(word)
            if word in model.key_to_index.keys():
                sentence = sentence + model[word]
        # Appending the sentence to the dataframe
        df_w2v_data = df_w2v_data.append(pd.DataFrame([sentence]))
    return df_w2v_data


def embed_words(doc_term, method="bow"):
    if method == 'bow':
        pass
    else:
        pass

#%%
