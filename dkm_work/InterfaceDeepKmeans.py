import numpy as np
from dkm_work.globals import PARAMS
class InterfaceDeepKmeans:
    def __init__(self,X:np.ndarray,params_model=PARAMS):
        """
            X: matrix n_sample * n_features,numerical representation of all documents
        """
        self.X
