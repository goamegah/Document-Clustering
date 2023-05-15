import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score
from src.core.clustering.globals import DEFAULT_PARAMS

class Kmeans:
    def __init__(self,hyperparams=DEFAULT_PARAMS["Kmeans"]):
        self.model = KMeans(**hyperparams)


    def fit(self,X,y=None) -> None:
        return self.model.fit(X,y,sample_weight=None)

    def fit_predict(self,X, y=None) -> np.ndarray:
        return self.model.fit_predict(X,y,sample_weight=None)

    def fit_transform(self,X, y=None) -> np.ndarray:
        return self.model.fit_transform(X,y,sample_weight=None)

    def get_pmi(self,labels_true,X,y=None) -> float:
        return normalized_mutual_info_score(labels_true,self.fit_predict(X,y))

    def get_ari(self,labels_true,X,y=None) -> float:
        return adjusted_rand_score(labels_true,self.fit_predict(X,y))