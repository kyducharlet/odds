from typing import Union
import numpy as np

from .base import BaseDetector
from .utils import IMPLEMENTED_BANDWIDTH_ESTIMATORS, neighbours_count
from tqdm import trange


class DBOKDE(BaseDetector):
    """
    Distance-Based Outliers using KDE

    Attributes
    ----------
    k: int
        a threshold on the number of neighbours needed to consider the point as normal
    R: float or str
        the distance defining the neighborhood around a point, can be computed dynamically, in this case set R="dynamic"
    win_size: int
        the number of points in the sliding window used in neighbours count
    sample_size: int, optional
        the number of points used as kernel centers for the KDE, if sample_size=-1 then sample_size is set to win_size (default is -1)

    Methods
    -------
    See BaseDetector methods
    """
    def __init__(self,  k: int, R: Union[float, str], win_size: int, sample_size: int = -1):
        self.win_size = win_size
        self.sample_size = win_size if sample_size == -1 else sample_size
        assert self.sample_size <= self.win_size
        assert self.sample_size > 0
        self.k = k
        self.__offset__ = 1 / (1 + k)
        self.R = R if R != "dynamic" else None
        self.R_strategy = R if R == "dynamic" else None
        self.be = IMPLEMENTED_BANDWIDTH_ESTIMATORS["scott"] if self.R_strategy is None else IMPLEMENTED_BANDWIDTH_ESTIMATORS["scott_with_R"]
        self.points = None  # kernel centers
        self.rd_s = None # random sample
        self.bsi = None  # inverse of the sqrt of the bandwidth
        self.bdsi = None  # inverse of the sqrt of the bandwidth determinant
        self.p = None  # number of variables

    def fit(self, x):
        self.assert_shape_unfitted(x)
        assert x.shape[0] >= self.win_size, "For this method, fit shape should at least be equal to win_size."
        self.p = x.shape[1]
        self.points = x[-self.win_size:]
        self.rd_s = np.random.choice(self.win_size, self.sample_size, replace=False)
        if self.R_strategy is None:
            self.bsi, self.bdsi = self.be(self.points[self.rd_s])
        else:
            self.R, self.bsi, self.bdsi = self.be(self.points[self.rd_s])
        return self

    def update(self, x):
        self.assert_shape_fitted(x)
        self.points = np.concatenate([self.points[max(0, len(self.points) - self.win_size + len(x)):], x[-self.win_size:]])
        if self.R_strategy is None:
            self.bsi, self.bdsi = self.be(self.points[self.rd_s])
        else:
            self.R, self.bsi, self.bdsi = self.be(self.points[self.rd_s])
        return self

    def score_samples(self, x):
        self.assert_shape_fitted(x)
        res = np.zeros(x.shape[0])
        for i, point in enumerate(x):
            overlapping_kc = [kc for kc in self.points[self.rd_s] if (np.abs(kc - point) <= (1 / np.diagonal(self.bsi)) + self.R).all()]
            res[i] = neighbours_count(point, overlapping_kc, self.bsi, self.bdsi, self.win_size, self.sample_size, self.R)
        return 1 / (1 + res)

    def decision_function(self, x):
        self.assert_shape_fitted(x)
        return self.__offset__ - self.score_samples(x)

    def predict(self, x):
        self.assert_shape_fitted(x)
        return np.where(self.decision_function(x) < 0, -1, 1)

    def eval_update(self, x):
        self.assert_shape_fitted(x)
        evals = np.zeros(len(x))
        for i in trange(len(x), ascii=True, desc=f"KDE (W={self.win_size})"):
            evals[i] = self.decision_function(x[i].reshape(1, -1))
            self.update(x[i].reshape(1, -1))
        return evals

    def predict_update(self, x):
        self.assert_shape_fitted(x)
        preds = np.zeros(len(x))
        for i in range(len(x)):
            preds[i] = self.predict(x[i].reshape(1, -1))
            self.update(x[i].reshape(1, -1))
        return preds

    def save_model(self):
        raise NotImplementedError("Not implemented yet.")

    def load_model(self, model_dict: dict):
        raise NotImplementedError("Not implemented yet.")

    def copy(self):
        model_bis = DBOKDE(self.k, self.R, self.sample_size, self.win_size)
        model_bis.__offset__ = self.__offset__
        model_bis.be = self.be
        model_bis.points = self.points
        model_bis.rd_s = self.rd_s
        model_bis.bsi = self.bsi
        model_bis.bdsi = self.bdsi
        model_bis.p = self.p
        return model_bis

    def method_name(self):
        return "DBO with KDE"
