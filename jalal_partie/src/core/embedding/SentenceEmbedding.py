import numpy as np


class SentencesEmbedding:
    def __init__(self,embeddings:np.ndarray,mapping:dict[str,int]):
        """
        :param embeddings: n words time d (dimension of rows)
        :param mapping: mapping word --> index for embedding numpy array
        """
        if not isinstance(embeddings,np.ndarray):
            raise TypeError(f"{hex(id(embeddings))} should be a numpy array")
        self.embeddings=embeddings
        self.mapping=mapping

    def create_sentence_embedding(self,sentence:str,method="average")-> np.ndarray:
        tokens=[word for word in sentence.split()]
        id_tokens=[self.mapping[word] for word in tokens]
        if method == "average":
            return np.mean([self.embeddings[id_token] for id_token in id_tokens],axis=0).reshape(-1)

    def create_sentences_embedding(self,sentences:list[str]) -> np.ndarray:
        snts_embedding=np.zeros((len(sentences),self.embeddings.shape[1]))
        for i in range(len(sentences)):
            snts_embedding[i,:]=self.create_sentence_embedding(sentences[i])
        return snts_embedding