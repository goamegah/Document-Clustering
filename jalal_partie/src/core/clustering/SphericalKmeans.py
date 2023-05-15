import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from src.core.clustering.KMeans import Kmeans
from src.core.clustering.globals import DEFAULT_PARAMS
class SphericalKmeans(Kmeans):
    def __init__(self,hyperparams=DEFAULT_PARAMS["SphericalKmeans"]):
        super().__init__(hyperparams)
    def normalize(self,X,eps=10**-3) -> np.ndarray:
        row_norms = np.linalg.norm(X, ord=2, axis=1)
        # Normalize each row by dividing it by its corresponding 2-norm
        return X / (row_norms[:, np.newaxis]+eps)
    def fit(self,X,y=None) -> None:
        return self.model.fit(self.normalize(X),y,sample_weight=None)

    def fit_predict(self,X, y=None) -> np.ndarray:
        return self.model.fit_predict(self.normalize(X),y,sample_weight=None)

    def fit_transform(self,X, y=None) -> np.ndarray:
        return self.model.fit_transform(self.normalize(X),y,sample_weight=None)

    def get_pmi(self,labels_true,X,y=None) -> float:
        return normalized_mutual_info_score(labels_true,self.fit_predict(X,y))

    def get_ari(self,labels_true,X,y=None) -> float:
        return adjusted_rand_score(labels_true,self.fit_predict(X,y))

