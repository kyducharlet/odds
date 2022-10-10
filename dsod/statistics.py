import numpy as np
from typing import Union
from scipy.special import comb
from scipy.optimize import curve_fit

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
        self.kc = np.concatenate([self.kc[len(self.kc) - self.win_size + len(x):], x[-self.win_size:]])
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
        evals = np.zeros(len(x))
        for i in range(len(x)):
            evals[i] = self.decision_function(x[i].reshape(1, -1))
            self.update(x[i].reshape(1, -1))
        return evals

    def predict_update(self, x):
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
    def __init__(self, d: int = 2, r: float = 0.5, forget_factor: Union[float, type(None)] = None, reg="0"):
        assert 0 < r <= 1
        self.p = None
        self.d = d
        self.r = r
        self.moments_matrix = MomentsMatrix(d, forget_factor=forget_factor)
        self.reg = reg

    def fit(self, x: np.ndarray):
        if len(x.shape) != 2:
            raise ValueError("The expected array shape: (N, p) do not match the given array shape: {}".format(x.shape))
        self.__dict__["fit_shape"] = x.shape
        self.p = self.__dict__["fit_shape"][1]
        self.moments_matrix.fit(x)
        if self.reg == "0":  # \alpha is the most basic regularizer
            self.__dict__["regularizer"] = 1
        elif self.reg == "1":  # \alpha is the most basic regularizer
            self.__dict__["regularizer"] =comb(self.d + x.shape[1], self.d)
        elif self.reg == "2":  # r * d * \alpha was the regulizer we used initially
            self.__dict__["regularizer"] = self.r * self.d * comb(self.d + x.shape[1], self.d)
        elif self.reg == "3":  # d ** {p+1} is the most inner border of the support
            self.__dict__["regularizer"] = np.power(self.d, x.shape[1] + 1)
        elif self.reg == "4":  # d ** {p+2} is the second most inner border of the support
            self.__dict__["regularizer"] = np.power(self.d, x.shape[1] + 2)
        elif self.reg == "5":  # exp(\srt(d) * \alpha) is the most outter border of the support
            self.__dict__["regularizer"] = np.exp(comb(self.d + x.shape[1], self.d) * np.sqrt(self.d))
        elif self.reg == "6":  # d ** {p+3} is between d ** {p+2} and exp(\srt(d) * \alpha)
            self.__dict__["regularizer"] = np.power(self.d, x.shape[1] + 3)
        else:
            raise ValueError("reg should be one of 1, 2, 3, 4, 5 or 6")
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
        return self.moments_matrix.score_samples(x.reshape(-1, self.__dict__["fit_shape"][1])) / self.__dict__["regularizer"]

    def decision_function(self, x):
        return 1 - self.score_samples(x)

    def predict(self, x):
        return np.where(self.decision_function(x) < 0, -1, 1)

    def fit_predict(self, x):
        self.fit(x.reshape(-1, self.__dict__["fit_shape"][1]))
        return self.predict(x.reshape(-1, self.__dict__["fit_shape"][1]))

    def eval_update(self, x):
        if len(x.shape) != 2 or x.shape[1] != self.__dict__["fit_shape"][1]:
            raise ValueError("The expected array shape: (N, {}) do not match the given array shape: {}".format(self.__dict__["fit_shape"][1], x.shape))
        evals = np.zeros(len(x))
        for i, xx in enumerate(x):
            xx.reshape(1, -1)
            evals[i] = self.decision_function(xx.reshape(-1, self.__dict__["fit_shape"][1]))
            self.update(xx.reshape(-1, self.__dict__["fit_shape"][1]))
        return evals

    def predict_update(self, x):
        if len(x.shape) != 2 or x.shape[1] != self.__dict__["fit_shape"][1]:
            raise ValueError("The expected array shape: (N, {}) do not match the given array shape: {}".format(self.__dict__["fit_shape"][1], x.shape))
        preds = np.zeros(len(x))
        for i, xx in enumerate(x):
            xx.reshape(1, -1)
            preds[i] = self.predict(xx.reshape(-1, self.__dict__["fit_shape"][1]))
            self.update(xx.reshape(-1, self.__dict__["fit_shape"][1]))
        return preds

    def copy(self):
        c_bis = SimpleChristoffel(d=self.d, r=self.r)
        c_bis.moments_matrix = self.moments_matrix.copy()
        if self.__dict__.get("fit_shape") is not None:
            c_bis.__dict__["fit_shape"] = self.__dict__["fit_shape"]
        if self.__dict__.get("regularizer") is not None:
            c_bis.__dict__["regularizer"]  = self.__dict__["regularizer"]
        return c_bis


class DyCG(BaseDetector):
    def __init__(self, degrees: np.ndarray = np.array(range(2, 9)), r: float = 0.5, forget_factor: Union[float, type(None)] = None,
                 decision="mean_growth", reg="6"):
        assert len(degrees) > 1
        self.degrees = degrees
        self.models = [SimpleChristoffel(d=d, r=r, forget_factor=forget_factor, reg=reg) for d in degrees]
        self.decision = decision
        self.p = None

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
        score = np.zeros((len(x), 1))
        for i, d in enumerate(x):
            d_ = d.reshape(1, -1)
            scores = np.array([m.score_samples(d_)[0] for m in self.models])
            s_diff = np.diff(scores) / np.diff(self.degrees)
            if self.decision == "sign_poly_2_reg":
                try:
                    score[i] = curve_fit(
                        lambda x, a, b: a * x ** 2 + b,
                        xdata=self.degrees,
                        ydata=scores,
                    )[0][0]
                except RuntimeError:
                    score[i] = np.nan
            elif self.decision == "mean_growth":
                score[i] = np.mean(s_diff)
            elif self.decision == "mean_growth_rate":
                score[i] = np.mean(s_diff / scores[:-1])
            elif self.decision == "mean_score":
                score[i] = np.mean(scores) - 1
            else:
                raise ValueError("decision should be one of sign_poly_2_reg, mean_growth, mean_score or mean_growth_rate, but we should have told you before :p")
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
        evals = np.zeros(len(x))
        for i, d in enumerate(x):
            d_ = d.reshape(1, -1)
            evals[i] = self.decision_function(d_)
            self.update(d_)
        return evals

    def predict_update(self, x):
        if len(x.shape) != 2 or x.shape[1] != self.p:
            raise ValueError("The expected array shape: (N, {}) do not match the given array shape: {}".format(self.p, x.shape))
        preds = np.zeros(len(x))
        for i, d in enumerate(x):
            d_ = d.reshape(1, -1)
            preds[i] = self.predict(d_)
            self.update(d_)
        return preds

    def copy(self):
        mc_bis = DyCG(degrees=self.degrees)
        mc_bis.models = [model.copy() for model in self.models]
        return mc_bis
