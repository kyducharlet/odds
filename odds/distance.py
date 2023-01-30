from typing import Union
import numpy as np

from .base import BaseDetector
from .utils import MTree


class OSCOD(BaseDetector):
    """
    One-Shot COD

    Attributes
    ----------
    k: int
        a threshold on the number of neighbours needed to consider the point as normal
    R: float
        the distance defining the neighborhood around a point
    win_size: int
        number of points in the sliding window used in neighbours count
    M: int (optional)
        max size of a node in the M-tree containing all points

    Methods
    -------
    See BaseDetector methods
    """

    def __init__(self, k: int, R: float, win_size: int, M: int = 5):
        self.p = None
        self.k = k
        self.R = R
        self.win_size = win_size
        self.M = M
        self.mt = MTree(M, R)
        self.__offset__ = 1 / (1 + k)

    def fit(self, x):
        self.assert_shape_unfitted(x)
        self.p = x.shape[1]
        for xx in x[-self.win_size:]:
            self.mt.insert_data(xx)
        return self

    def update(self, x):
        self.assert_shape_fitted(x)
        for xx in x[-self.win_size:]:
            # Removal
            self.mt.remove_oldest()
            # Insertion
            self.mt.insert_data(xx)
        return self

    def score_samples(self, x):
        self.assert_shape_fitted(x)
        res = np.zeros(x.shape[0])
        for i, xx in enumerate(x):
            res[i] = self.mt.score_point(xx)
        return 1 / (1 + res)

    def decision_function(self, x):
        self.assert_shape_fitted(x)
        return self.__offset__ - self.score_samples(x)

    def predict(self, x):
        self.assert_shape_fitted(x)
        return np.where(self.decision_function(x) < 0, -1, 1)

    def predict_update(self, x):
        self.assert_shape_fitted(x)
        preds = np.zeros(len(x))
        for i, xx in enumerate(x):
            preds[i] = self.predict(xx.reshape(1, -1))
            self.update(xx.reshape(1, -1))
        return preds

    def eval_update(self, x):
        self.assert_shape_fitted(x)
        evals = np.zeros(len(x))
        for i, xx in enumerate(x):
            evals[i] = self.decision_function(xx.reshape(1, -1))
            self.update(xx.reshape(1, -1))
        return evals

    def copy(self):
        raise NotImplementedError("The copy method for OSCOD has not been implemented yet.")

    def method_name(self):
        return "One-Shot COD"
