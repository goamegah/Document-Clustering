import numpy as np
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score
from src.core.clustering.globals import DEFAULT_PARAMS
class CAH:
    def __init__(self,hyperparams=DEFAULT_PARAMS["CAH"]):
        self.method=hyperparams["method"]



    def create_cluster(self,X:np.ndarray):
        return hierarchy.linkage(X,method=self.method)
    def create_dendogram(self,X:np.ndarray,ax):
        return hierarchy.dendrogram(self.create_cluster(X),ax=ax,orientation="top")

    def fit_predict(self,X, y=None,criterion="distance",t=3) -> np.ndarray:
        return fcluster(self.create_cluster(X),
                        criterion=criterion,t=t)

    def get_pmi(self,labels_true,X,criterion="distance",t=3) -> float:
        return normalized_mutual_info_score(labels_true,
        self.fit_predict(X,criterion=criterion,t=t))

    def get_ari(self,labels_true,X,y=None) -> float:
        return adjusted_rand_score(labels_true,self.fit_predict(X,y))
