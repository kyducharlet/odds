from typing import Union
from .base import BaseDetector
from .utils import np
from .utils import IMPLEMENTED_BANDWIDTH_ESTIMATORS, neighbours_count, neighbours_counts_in_grid
from .utils import MomentsMatrix, update_params, compute_R
from tqdm import trange


class MDEFKDE(BaseDetector):
    """
    MDEF using KDE

    Attributes
    ----------
    k_sigma: float
        a threshold on the MDEF score: MDEF > k_sigma * sigmaMDEF (standard deviation of local MDEF) is an outlier (3 is often used)
    R: float
        the distance defining the neighborhood around a point
    alpha: int (alpha>1)
        a parameter for the second radius of neighborhood, defined as a 1/(2**alpha) * R neighborhood
    win_size: int
        the number of points in the sliding window used in neighbours count
    sample_size: int, optional
        the number of points used as kernel centers for the KDE, if sample_size=-1 then sample_size is set to win_size (default is -1)

    Methods
    -------
    See BaseDetector methods
    """

    def __init__(self, k_sigma: int, R: Union[float, str], alpha: int, win_size: int, sample_size: int = -1):
        assert sample_size <= win_size
        assert sample_size > 0
        assert alpha >= 1, "alpha should be set greater than 1."
        self.k = k_sigma
        self.R = R if R != "dynamic" else None
        self.R_strategy = R if R == "dynamic" else None
        self.alpha = alpha
        self.r = R / (2 ** alpha) if R != "dynamic" else None
        self.win_size = win_size
        self.sample_size = win_size if sample_size == -1 else sample_size
        self.be = IMPLEMENTED_BANDWIDTH_ESTIMATORS["scott"] if self.R_strategy is None else IMPLEMENTED_BANDWIDTH_ESTIMATORS["scott_with_R"]
        self.points = None  # kernel centers
        self.rd_s = None  # random sample
        self.bsi = None  # inverse of the sqrt of the bandwidth
        self.bdsi = None  # inverse of the sqrt of the bandwidth determinant
        self.p = None  # number of variables

    def fit(self, x):
        self.assert_shape_unfitted(x)
        assert x.shape[0] >= self.win_size, "For this method, fit shape should at leas be equal to win_size."
        self.p = x.shape[1]
        self.points = x[-self.win_size:]
        self.rd_s = np.random.choice(self.win_size, self.sample_size, replace=False)
        if self.R_strategy is None:
            self.bsi, self.bdsi = self.be(self.points[self.rd_s])
        else:
            self.r, self.bsi, self.bdsi = self.be(self.points[self.rd_s])
            self.R = (2 ** self.alpha) * self.r
        return self

    def update(self, x):
        self.assert_shape_fitted(x)
        self.points = np.concatenate([self.points[max(0, len(self.points) - self.win_size + len(x)):], x[-self.win_size:]])
        if self.R_strategy is None:
            self.bsi, self.bdsi = self.be(self.points[self.rd_s])
        else:
            self.r, self.bsi, self.bdsi = self.be(self.points[self.rd_s])
            self.R = (2 ** self.alpha) * self.r

    def score_samples(self, x):
        self.assert_shape_fitted(x)
        res = np.zeros(x.shape[0])
        for i, point in enumerate(x):
            overlapping_kc = [kc for kc in self.points[self.rd_s] if (np.abs(kc - point) < (1 / np.diagonal(self.bsi)) + self.R).all()]
            if len(overlapping_kc) == 0:
                res[i] = 1.0
            else:
                nc = neighbours_count(point, overlapping_kc, self.bsi, self.bdsi, self.win_size, self.sample_size, self.r)  # estimate the number of points in r-neighborhood
                ncig = neighbours_counts_in_grid(point, overlapping_kc, self.bsi, self.bdsi, self.win_size, self.sample_size, self.alpha, self.r)  # estimate the number of points in every r-neihborhood
                if (ncig == 0).all():
                    res[i] = 1.0
                else:
                    nc_mean = np.sum(np.square(ncig)) / np.sum(ncig)
                    estimate = (np.sum(np.power(ncig, 3)) / np.sum(ncig)) - np.square(nc_mean)
                    if estimate < 0:
                        if abs(estimate) < 1e-8:
                            estimate = abs(estimate)
                        else:
                            raise ValueError(f"Estimate should be positive but is {estimate}.")
                    nc_std = np.sqrt(estimate)
                    mdef = 1 - nc / nc_mean
                    std_mdef = nc_std / nc_mean
                    res[i] = mdef - self.k * std_mdef
        return res

    def decision_function(self, x):
        self.assert_shape_fitted(x)
        return -1 * self.score_samples(x)

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
        model_bis = MDEFKDE(self.k, self.R, self.alpha, self.sample_size, self.win_size)
        model_bis.be = self.be
        model_bis.points = self.points
        model_bis.rd_s = self.rd_s
        model_bis.bsi = self.bsi
        model_bis.bdsi = self.bdsi
        model_bis.p = self.p

    def method_name(self):
        return "MDEF with KDE"


class MDEFECF(BaseDetector):
    """
    Distance-Based Outliers using the Empirical Christoffel Function

    Attributes
    ----------
    k_sigma: float
        a threshold on the MDEF score: MDEF > k_sigma * sigmaMDEF (standard deviation of local MDEF) is an outlier (3 is often used)
    R: float
        the distance defining the neighborhood around a point
    alpha: int (alpha>1)
        a parameter for the second radius of neighborhood, defined as a 1/(2**alpha) * R neighborhood
    d: int
        the degree for the ECF
    incr_opt: str, optional
        whether "inverse" to inverse the moments matrix each iteration or "sherman" to use the Sherman-Morrison formula (default is "inv")
    polynomial_basis: str, optional
        polynomial basis used to compute moment matrix, either "monomials", "chebyshev_t_1", "chebyshev_t_2", "chebyshev_u" or "legendre",
        varying this parameter can bring stability to the score in some cases (default is "monomials")

    Methods
    -------
    See BaseDetector methods
    """

    def __init__(self, k_sigma: int, R: Union[float, str], alpha: int, d: int, N_sample: int = 100, incr_opt: str = "inverse", polynomial_basis: str = "monomials"):
        assert alpha >= 1, "alpha should be set greater than 1."
        self.N = 0  # number of points integrated in the moments matrix
        self.k = k_sigma
        self.R = R if R != "dynamic" else None
        self.R_strategy = R if R == "dynamic" else None
        self.alpha = alpha
        self.r = R / (2 ** alpha) if R != "dynamic" else None
        self.mean = None  # mean of the distribution, used to compute R dynamically
        self.std = None  # std of the distribution, used to compute R dynamically
        self.p = None
        self.d = d
        self.N_sample = N_sample
        self.moments_matrix = MomentsMatrix(d, incr_opt=incr_opt, polynomial_basis=polynomial_basis)

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
                self.mean, self.std = update_params(self.mean, self.std, self.N, xx)
            self.R = compute_R(self.std, self.N, self.p)
        else:
            self.N += x.shape[0]
        return self

    def score_samples(self, x):
        self.assert_shape_fitted(x)
        res = np.zeros(x.shape[0])
        for i, xx in enumerate(x):
            nc, err = self.moments_matrix.estimate_neighbours(xx, self.r, self.N_sample)
            ncig = self.moments_matrix.estimate_neighbours_in_grid(xx, self.r, self.alpha, self.N_sample)
            """if np.quantile(ncig, 0.7) < 0.1 * (1 / (self.d ** (3 * self.p / 2))) * ((2 * self.r) ** 2):
                res[i] = 1.0"""
            """if (ncig < 0).all():
                res[i] = 1.0"""
            if (ncig < (1 / (self.d ** (3 * self.p / 2))) * ((2 * self.r) ** 2)).all():
                res[i] = 1.0
            else:
                nc_mean = np.sum(np.square(ncig)) / np.sum(ncig)
                estimate = (np.sum(np.power(ncig, 3)) / np.sum(ncig)) - np.square(nc_mean)
                if estimate < 0:
                    if abs(estimate) < 1e-8:
                        estimate = abs(estimate)
                    else:
                        raise ValueError
                nc_std = np.sqrt(estimate)
                mdef = 1 - nc / nc_mean
                std_mdef = nc_std / nc_mean
                res[i] = mdef - self.k * std_mdef
        return res

    def decision_function(self, x):
        return -1 * self.score_samples(x)

    def predict(self, x):
        return np.where(self.decision_function(x) < 0, -1, 1)

    def eval_update(self, x):
        res = np.zeros(len(x))
        for i in trange(len(x), ascii=True, desc=f"DyCF (d={self.d})"):
            res[i] = self.decision_function(x[i].reshape(1, -1))
            self.update(x[i].reshape(1, -1))
        return res

    def predict_update(self, x):
        return np.where(self.eval_update(x) < 0, -1, 1)

    def save_model(self):
        raise NotImplementedError("Not implemented yet.")

    def load_model(self, model_dict: dict):
        raise NotImplementedError("Not implemented yet.")

    def copy(self):
        raise NotImplementedError("Not implemented yet for MDEFECF.")

    def method_name(self):
        return "MDEF with ECF"
