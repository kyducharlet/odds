import numpy as np
from typing import Union
from scipy.special import comb
from scipy.optimize import least_squares

from .base import BaseDetector, NotFittedError
from .utils import IMPLEMENTED_KERNEL_FUNCTIONS, IMPLEMENTED_BANDWIDTH_ESTIMATORS
from .utils import MomentsMatrix


class SlidingMKDE(BaseDetector):
    """
    Multivariate Kernel Density Estimation with Sliding Windows

    Attributes
    ----------
    kernel: str, optional
        the type of kernel to use (default is "gaussian")
    bandwidth: str, optional
        rule of thumb to compute the bandwidth (default is "scott")
    win_size: int, optional
        size of the window of kernel centers to keep in memory (default is 2000)

    Methods
    -------
    See BaseDetector methods
    """

    def __init__(self, threshold: float = .1, kernel: str = "gaussian", bandwidth: str = "scott", win_size: int = 2000):
        assert threshold > 0
        assert kernel in IMPLEMENTED_KERNEL_FUNCTIONS.keys()
        assert bandwidth in IMPLEMENTED_BANDWIDTH_ESTIMATORS.keys()
        assert win_size > 0
        self.threshold = threshold
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.win_size = win_size
        self.kf = IMPLEMENTED_KERNEL_FUNCTIONS[kernel]
        self.be = IMPLEMENTED_BANDWIDTH_ESTIMATORS[bandwidth]
        self.kc = None  # kernel centers
        self.bsi = None  # inverse of the sqrt of the bandwidth
        self.bdsi = None  # inverse of the sqrt of the bandwidth determinant
        self.p = None  # number of variables

    def fit(self, x: np.ndarray):
        if len(x.shape) != 2:
            raise ValueError("The expected array shape: (N, p) do not match the given array shape: {}".format(x.shape))
        self.kc = x[-self.win_size:]
        self.p = x.shape[1]
        self.bsi, self.bdsi = self.be(self.kc)
        return self

    def update(self, x: np.ndarray):
        if len(x.shape) != 2 or x.shape[1] != self.p:
            raise ValueError("The expected array shape: (N, {}) do not match the given array shape: {}".format(self.p, x.shape))
        if self.kc is None:
            raise NotFittedError("This SlidingMKDE instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        self.kc = np.concatenate(self.kc[len(self.kc) - self.win_size + len(x):], x[-self.win_size:])
        self.bsi, self.bdsi = self.be(self.kc)

    def score_samples(self, x):
        if len(x.shape) != 2 or x.shape[1] != self.p:
            raise ValueError("The expected array shape: (N, {}) do not match the given array shape: {}".format(self.p, x.shape))
        if self.kc is None:
            raise NotFittedError("This SlidingMKDE instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        res = np.zeros(x.shape[0])
        for i, point in enumerate(x):
            ke = self.bdsi * self.kf(np.dot(self.bsi, (self.kc - point).T).T)
            res[i] = np.mean(ke)
        return 1 / (1 + res)

    def decision_function(self, x):
        return (1 / (1 + self.threshold)) - self.score_samples(x)

    def predict(self, x):
        return np.where(self.decision_function(x) < 0, -1, 1)

    def eval_update(self, x):
        preds = np.zeros(len(x))
        for i in range(len(x)):
            preds[i] = self.predict(x[i].reshape(1, -1))
            self.update(x[i].reshape(1, -1))
        return preds

    def copy(self):
        model_bis = SlidingMKDE(self.threshold, self.kernel, self.bandwidth, self.win_size)
        model_bis.kc = self.kc
        model_bis.bsi = self.bsi
        model_bis.bdsi = self.bdsi
        model_bis.p = self.p
        return model_bis


class SimpleChristoffel(BaseDetector):
    def __init__(self, d: int = 2, r: float = 0.5, forget_factor: Union[float, type(None)] = None):
        assert 0 < r <= 1
        self.d = d
        self.r = r
        self.moments_matrix = MomentsMatrix(d, forget_factor=forget_factor)

    def fit(self, x: np.ndarray):
        if len(x.shape) != 2:
            raise ValueError("The expected array shape: (N, p) do not match the given array shape: {}".format(x.shape))
        self.__dict__["fit_shape"] = x.shape
        self.moments_matrix.fit(x)
        self.__dict__["offset_"] = self.r * self.d * comb(self.d + x.shape[1], self.d)
        return self

    def update(self, x):
        if (not self.moments_matrix.learned()) or (self.__dict__.get("fit_shape") is None):
            raise NotFittedError("This Christoffel instance is not fitted yet. Call 'fit' with appropriate arguments before using this updating method.")
        elif len(x.shape) != 2 or x.shape[1] != self.__dict__["fit_shape"][1]:
            raise ValueError("The expected array shape: (N, {}) do not match the given array shape: {}".format(self.__dict__["fit_shape"][1], x.shape))
        self.moments_matrix.update(x, float(self.__dict__["fit_shape"][0]))
        self.__dict__["fit_shape"] = (self.__dict__["fit_shape"][0] + x.shape[0], self.__dict__["fit_shape"][1])
        return self

    def score_samples(self, x):
        if (not self.moments_matrix.learned()) or (self.__dict__.get("fit_shape") is None):
            raise NotFittedError("This Christoffel instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        elif len(x.shape) != 2 or x.shape[1] != self.__dict__["fit_shape"][1]:
            raise ValueError("The expected array shape: (N, {}) do not match the given array shape: {}".format(self.__dict__["fit_shape"][1], x.shape))
        # return self.moments_matrix.score_samples(x)
        return self.moments_matrix.score_samples(x) / self.__dict__["offset_"]

    def decision_function(self, x):
        return 1 - self.score_samples(x)

    def predict(self, x):
        if self.__dict__.get("offset_") is None:
            raise NotFittedError("This Christoffel instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        scores = self.score_samples(x)
        return np.where(scores > 1, -1, 1)

    def fit_predict(self, x):
        self.fit(x)
        return self.predict(x)

    def eval_update(self, x):
        if len(x.shape) != 2 or x.shape[1] != self.__dict__["fit_shape"]:
            raise ValueError("The expected array shape: (N, {}) do not match the given array shape: {}".format(self.__dict__["fit_shape"], x.shape))
        res = np.zeros(len(x))
        for i, xx in enumerate(x):
            xx.reshape(1, -1)
            res[i] = self.predict(xx)
            self.update(xx)
        return res

    def copy(self):
        c_bis = SimpleChristoffel(d=self.d, r=self.r)
        c_bis.moments_matrix = self.moments_matrix.copy()
        if self.__dict__.get("fit_shape") is not None:
            c_bis.__dict__["fit_shape"] = self.__dict__["fit_shape"]
        if self.__dict__.get("offset_") is not None:
            c_bis.__dict__["offset_"] = self.__dict__["offset_"]
        return c_bis


class DyCG(BaseDetector):
    def __init__(self, degrees: np.ndarray = np.array(range(2, 9)), r: float = 0.5, forget_factor: Union[float, type(None)] = None):
        assert len(degrees) > 1
        self.degrees = degrees
        self.models = [SimpleChristoffel(d=d, r=r, forget_factor=forget_factor) for d in degrees]

    def fit(self, x):
        if len(x.shape) != 2:
            raise ValueError("The expected array shape: (N, p) do not match the given array shape: {}".format(x.shape))
        self.p = x.shape[1]
        for model in self.models:
            model.fit(x)
        return self

    def update(self, x: np.ndarray):
        for model in self.models:
            model.update(x)
        return self

    def score_samples(self, x):
        if len(x.shape) != 2 or x.shape[1] != self.p:
            raise ValueError("The expected array shape: (N, {}) do not match the given array shape: {}".format(self.p, x.shape))
        scores = np.zeros((len(x), len(self.models)))
        score = np.zeros((len(x), 1))
        for i, d in enumerate(x):
            d_ = d.reshape(1, -1)
            scores[i] = [m.score_samples(d_) for m in self.models]
            s_diff = np.diff(scores[i]) / np.diff(self.degrees)
            score[i] = least_squares(
                lambda x, t, y: x[0] * t + x[1] - y,
                x0=np.array([1, 0]),
                args=(s_diff, self.degrees[:-1])
            ).x[0]
        return score

    def decision_function(self, x):
        if len(x.shape) != 2 or x.shape[1] != self.p:
            raise ValueError("The expected array shape: (N, {}) do not match the given array shape: {}".format(self.p, x.shape))
        return -1 * self.score_samples(x)

    def predict(self, x):
        if len(x.shape) != 2 or x.shape[1] != self.p:
            raise ValueError("The expected array shape: (N, {}) do not match the given array shape: {}".format(self.p, x.shape))
        return np.where(self.decision_function(x) < 0, -1, 1)

    def eval_update(self, x):
        if len(x.shape) != 2 or x.shape[1] != self.p:
            raise ValueError("The expected array shape: (N, {}) do not match the given array shape: {}".format(self.p, x.shape))
        decisions = np.zeros(len(x))
        for i, d in enumerate(x):
            d_ = d.reshape(1, -1)
            decisions[0] = self.predict(d_)
            self.update(d_)
        return decisions

    def copy(self):
        mc_bis = DyCG(degrees=self.degrees)
        mc_bis.models = [model.copy() for model in self.models]
        return mc_bis
