from typing import Union
import numpy as np

from .base import BaseDetector
from .utils import MTree
from .utils import IMPLEMENTED_BANDWIDTH_ESTIMATORS, neighbours_count
from .utils import MomentsMatrix, update_params, compute_R
from tqdm import trange


class DBOKDE(BaseDetector):
    """
    Distance-Based Outliers using KDE

    Attributes
    ----------
    k: int
        a threshold on the number of neighbours needed to consider the point as normal
    R: float
        the distance defining the neighborhood around a point
    sample_size: int
        the number of points used as kernel centers for the KDE
    win_size: int
        the number of points in the sliding window used in neighbours count

    Methods
    -------
    See BaseDetector methods
    """
    def __init__(self,  k: int, R: Union[float, str], sample_size: int, win_size: int):
        assert sample_size <= win_size
        assert sample_size > 0
        self.k = k
        self.__offset__ = 1 / (1 + k)
        self.R = R if R != "dynamic" else None
        self.R_strategy = R if R == "dynamic" else None
        self.sample_size = sample_size
        self.win_size = win_size
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

    def copy(self):
        model_bis = DBOKDE(self.k, self.R, self.sample_size, self.win_size)
        model_bis.__offset__ = self.__offset__
        model_bis.be = self.be
        model_bis.points = self.points
        model_bis.rd_s = self.rd_s
        model_bis.bsi = self.bsi
        model_bis.bdsi = self.bdsi
        model_bis.p = self.p

    def method_name(self):
        return "DBO with KDE"


class DBOECF(BaseDetector):
    """
    Distance-Based Outliers using the Empirical Christoffel Function

    Attributes
    ----------
    threshold: float
        a threshold on the "unknownly ratioed" number of neighbours needed to consider the point as normal
    R: float
        the distance defining the neighborhood around a point
    d: int
        the degree for the ECF
    incr_opt: str, optional
        whether "inverse" to inverse the moments matrix each iteration or "sherman" to use the Sherman-Morrison formula (default is "inv")
    polynomial_basis: str, optional
        whether "monomials" to use the monomials basis, "legendre" to use the Legendre polynomials or "chebyshev" to use the Chebyshev polynomials (default is "monomials")

    Methods
    -------
    See BaseDetector methods
    """

    def __init__(self, threshold: float, R: Union[float, str], d: int, N_sample: int = 100, incr_opt: str = "inverse", polynomial_basis: str = "monomials"):
        self.N = 0  # number of points integrated in the moments matrix
        self.__offset__ = 1 / (1 + threshold)
        self.R = R if R != "dynamic" else None
        self.R_strategy = R if R == "dynamic" else None
        self.mean = None  # mean of the distribution, used to compute R dynamically
        self.std = None  # std of the distribution, used to compute R dynamically
        self.p = None
        self.d = d
        self.moments_matrix = MomentsMatrix(d, incr_opt=incr_opt, polynomial_basis=polynomial_basis)
        self.N_sample = N_sample

    def fit(self, x: np.ndarray):
        self.assert_shape_unfitted(x)
        self.N = x.shape[0]
        self.p = x.shape[1]
        if self.R_strategy is not None:
            self.mean = np.mean(x, axis=0)
            self.std = np.std(x, axis=0)
            self.R = compute_R(self.std, self.N, self.p)
        self.moments_matrix.fit(x)
        return self

    def update(self, x):
        self.assert_shape_fitted(x)
        self.moments_matrix.update(x, self.N)
        if self.R_strategy is not None:
            for xx in x:
                self.N += 1
                self.mean, self.std = update_params(self.mean, self.std, xx, self.N)
            self.R = compute_R(self.std, self.N, self.p)
        else:
            self.N += x.shape[0]
        return self

    def __score_samples__(self, x):
        self.assert_shape_fitted(x)
        res = []
        err_total = []
        for xx in x:
            value, err = self.moments_matrix.estimate_neighbours(xx, self.R, self.N_sample)
            res.append(value)
            err_total.append(err)
        return 1 / (1 + np.array(res)), np.mean(err_total)

    def score_samples(self, x):
        scores, err = self.__score_samples__(x)
        return scores

    def __decision_function__(self, x):
        self.assert_shape_fitted(x)
        score, err = self.__score_samples__(x)
        return self.__offset__ - score, err

    def decision_function(self, x):
        self.assert_shape_fitted(x)
        return self.score_samples(x)

    def predict(self, x):
        self.assert_shape_fitted(x)
        return np.where(self.decision_function(x) < 0, -1, 1)

    def eval_update(self, x):
        evals = np.zeros(len(x))
        err = np.zeros(len(x))
        for i in trange(len(x), ascii=True, desc=f"DyCF (d={self.d})"):
            evals[i], err[i] = self.__decision_function__(x[i].reshape(-1, self.p))
            self.update(x[i].reshape(-1, self.p))
        # print(f"Err: {np.mean(err)}")
        return evals

    def predict_update(self, x):
        evals = self.eval_update(x)
        return np.where(evals < 0, -1, 1)

    def copy(self):
        raise NotImplementedError("Not implemented yet for DBOECF.")

    def method_name(self):
        return "DBO with the ECF"
