import numpy as np
import pandas as pd
from nltk.lm.vocabulary import Vocabulary
from sklearn.feature_extraction.text import CountVectorizer

class Processing:
    def __init__(self):
        pass
    @classmethod
    def create_vocab(cls,list_words:list[str],min_count:int=10,unk_label="<UNK>") -> Vocabulary:
        return Vocabulary(list_words,unk_cutoff=min_count,unk_label=unk_label)

    @classmethod
    def list_words_from_sentences(cls,sentences:list[str]) -> list:
        sentences=[sentence.split() for sentence in sentences]
        return [word for sentences in sentences for word in sentences]

    @classmethod
    def get_dt_matrix(cls,tweets:pd.Series) -> tuple:
        cv=CountVectorizer()
        corpus=[str(tweet) for tweet in list(tweets)]
        return cv.fit_transform(corpus),cv.vocabulary_


    @classmethod
    def create_td_matrix(cls, docs: list[str], vocab: dict[str, int]) -> np.ndarray:
        n_docs = len(docs)
        m_terms = len(vocab.keys())
        td_mat = np.zeros((m_terms, n_docs))
        for i in range(n_docs):
            for token in docs[i].split():
                td_mat[vocab[token], i] += 1
        return td_mat

    @classmethod
    def create_co_occurences(cls, docs: list[str], vocab: dict[str, int]) -> np.ndarray:
        """
        docs: list of documents
        vocab: mapping word:index
        :return:  co_occurences matrix
        """
        n_terms = len(vocab.keys())
        SS_mat = np.zeros([n_terms, n_terms])
        for s in docs:
            tokens = s.split()
            for i in range(len(tokens)):
                for j in range(len(tokens)):
                    if i != j:
                        w1,w2=tokens[i],tokens[j]
                        SS_mat[vocab[w1], vocab[w2]] += 1.0
        return SS_mat

    @classmethod
    def create_log_co_occurences(cls, docs: list[str], vocab: dict[str, int]) -> np.ndarray:
        """
        docs: list of documents
        vocab: mapping word:index
        :return: log co_occurences matrix returned (ndarray)
        """
        SS_mat = cls.create_co_occurences(docs, vocab)
        n_terms = SS_mat.shape[0]
        D1 = np.sum(SS_mat)
        lSS_mat = D1 * SS_mat
        for k in range(n_terms):
            if not np.sum(SS_mat[k]):
                print([word for word in vocab.keys() if vocab[word] == k])
            lSS_mat[k] /= np.sum(SS_mat[k])
        for k in range(n_terms):
            lSS_mat[:, k] /= np.sum(SS_mat[:, k])
        SS_mat=[]
        lSS_mat[lSS_mat == 0] = 1.0
        lSS_mat = np.log(lSS_mat)
        lSS_mat[lSS_mat < 0.0] = 0.0
        return lSS_mat


