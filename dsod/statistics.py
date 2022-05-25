import numpy as np

from .base import BaseDetector, NotFittedError
from .utils import IMPLEMENTED_KERNEL_FUNCTIONS, IMPLEMENTED_BANDWIDTH_ESTIMATORS


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

    def __init__(self, kernel="gaussian", bandwidth="scott", win_size: int = 2000):
        assert kernel in IMPLEMENTED_KERNEL_FUNCTIONS.keys()
        assert bandwidth in IMPLEMENTED_BANDWIDTH_ESTIMATORS.keys()
        assert win_size > 0
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
        self.kc = x
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
        return 1 / res

    def decision_function(self, x):
        raise NotImplementedError("Oops! This has yet to be implemented.")
        pass

    def predict(self, x):
        raise NotImplementedError("Oops! This has yet to be implemented.")
        pass

    def eval_update(self, x):
        raise NotImplementedError("Oops! This has yet to be implemented.")
        pass

    def copy(self):
        model_bis = SlidingMKDE(self.kernel, self.bandwidth, self.win_size)
        model_bis.kc = self.kc
        model_bis.bsi = self.bsi
        model_bis.bdsi = self.bdsi
        model_bis.p = self.p
        return model_bis


class SparseKDE(BaseDetector):
    pass


class Christoffel(BaseDetector):
    pass


class DyCG(BaseDetector):
    pass
