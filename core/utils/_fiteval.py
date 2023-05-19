from collections import defaultdict
from sklearn import metrics
from time import time


class ClustFitEval(object):

    def __init__(self):
        self.scores = {}
        self.train_times = []
        self.evaluations = []
        self.evaluations_std = []

    def fit(
            self,
            km,
            X,
            name=None,
            dset=None,
            n_runs=5
    ):
        _name = km.__class__.__name__ if name is None else name
        self.scores = defaultdict(list)
        for seed in range(n_runs):
            if name.casefold() not in ["spericalkmeans", "skm", "sperical_kmeans", "sperical-kmeans", "sperical-k-means"]:
                km.set_params(random_state=seed)
        t0 = time()
        km.fit(X)
        self.train_times.append(time() - t0)

    set

