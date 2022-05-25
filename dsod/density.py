import numpy as np

from .base import BaseDetector, NotFittedError
from .utils import search_kNN, compute_k_distance, compute_rd, compute_lrd, compute_lof, update_when_adding, update_when_removing


class ILOF(BaseDetector):
    def __init__(self, k: int, win_size=None):
        self.k = k
        self.win_size = win_size
        if self.win_size is None:
            self.update_f = self.__update_without_window__
            self.select_f = self.__select_without_window__
        else:
            self.update_f = self.__update_with_window__
            self.select_f = self.__select_with_window__
        self.points = None
        self.k_distances = None
        self.kNNs = None
        self.rds = None
        self.lrds = None
        self.lofs = None
        self.p = None

    def fit(self, x):
        if len(x.shape) != 2:
            raise ValueError("The expected array shape: (N, p) do not match the given array shape: {}".format(x.shape))
        x_ = self.select_f(x)
        self.p = x_.shape[1]
        self.points = {i: x_[i] for i in range(len(x_))}
        self.kNNs = {i: search_kNN(self.points, i, self.k) for i in range(len(x_))}
        self.k_distances = {i: compute_k_distance(self.points, i, self.kNNs) for i in range(len(x_))}
        self.rds = {i: {o: compute_rd(self.points, i, o, self.k_distances) for o in list(self.kNNs[i].keys())} for i in range(len(x_))}
        self.lrds = {i: compute_lrd(self.rds, i) for i in range(len(x_))}
        self.lofs = {i: compute_lof(self.lrds, self.kNNs, i) for i in range(len(x_))}
        return self

    def update(self, x):
        if len(x.shape) != 2 or x.shape[1] != self.p:
            raise ValueError("The expected array shape: (N, {}) do not match the given array shape: {}".format(self.p, x.shape))
        if self.points is None:
            raise NotFittedError("This ILOF instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        x_ = self.select_f(x)
        for x_new in x_:
            self.update_f(x_new)
        return self

    def score_samples(self, x):
        if len(x.shape) != 2 or x.shape[1] != self.p:
            raise ValueError("The expected array shape: (N, {}) do not match the given array shape: {}".format(self.p, x.shape))
        if self.points is None:
            raise NotFittedError("This ILOF instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        lofs = np.zeros(len(x))
        for i, point in enumerate(x):
            temp_points = self.points.copy()
            temp_points[-1] = point
            kNN = search_kNN(temp_points, -1, self.k)
            rds = {-1: {o: compute_rd(temp_points, -1, o, self.k_distances) for o in list(kNN.keys())}}
            temp_lrd = self.lrds.copy()
            temp_lrd[-1] = compute_lrd(rds, -1)
            lofs[i] = compute_lof(temp_lrd, {-1: kNN}, -1)
        return lofs

    def decision_function(self, x):
        raise NotImplementedError("Oops! This has yet to be implemented.")
        pass

    def predict(self, x):
        raise NotImplementedError("Oops! This has yet to be implemented.")
        pass

    def eval_update(self, x):
        scores = np.zeros(len(x))
        for i in range(len(x)):
            self.update(x[i])
            scores = self.lofs[max(list(self.lofs.keys()))]
        return scores

    def __select_with_window__(self, x):
        x_windowed = x[-self.win_size:]
        return x_windowed

    def __select_without_window__(self, x):
        return x

    def __update_with_window__(self, x):
        index = max(list(self.points.keys())) + 1
        while len(self.points) >= self.win_size:  # remove points if we exceed the win_size
            self.__remove_point__(min(list(self.points.keys())))
        self.__add_point__(x, index)

    def __update_without_window__(self, x):
        index = list(self.points.keys())[-1] + 1
        self.__add_point__(x, index)

    def __add_point__(self, x, index):
        if len(x.shape) == 1 and len(x) == self.p:
            x_new = x
        elif len(x.shape) == 2 and x.shape[1] == self.p:
            x_new = x.reshape(-1)
        else:
            raise ValueError("The expected array shape: (N, {}) do not match the given array shape: {}".format(self.p, x.shape))

        self.kNNs, self.k_distances, self.rds, self.lrds, self.lofs = \
            update_when_adding(self.points, index, x_new, self.kNNs, self.k_distances, self.rds, self.lrds, self.lofs, self.k)

    def __remove_point__(self, index):
        self.kNNs, self.k_distances, self.rds, self.lrds, self.lofs = \
            update_when_removing(self.points, index, self.kNNs, self.k_distances, self.rds, self.lrds, self.lofs, self.k)

    def copy(self):
        model_bis = ILOF(self.k, self.win_size)
        model_bis.points = self.points
        model_bis.k_distances = self.k_distances
        model_bis.kNNs = self.kNNs
        model_bis.rds = self.rds
        model_bis.lrds = self.lrds
        model_bis.lofs = self.lofs
        model_bis.p = self.p
        return model_bis


class MILOF(BaseDetector):
    pass