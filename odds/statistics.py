from .base import BaseDetector
from .utils import np
from .utils import IMPLEMENTED_KERNEL_FUNCTIONS, IMPLEMENTED_BANDWIDTH_ESTIMATORS
from .utils import MomentsMatrix
from .utils import SDEM, IMPLEMENTED_SS_SCORING_FUNCTIONS
from math import comb


class KDE(BaseDetector):
    """
    Multivariate Kernel Density Estimation with Sliding Windows

    Attributes
    ----------
    d: int
        the degree of polynomials, usually set between 2 and 8
    C: float, optional
        define a threshold on the score; when used with regularization="vu", usually C<=1 (default is 1)
    incr_opt: str, optional
        can be either "inverse" to inverse the moments matrix each iteration or "sherman" to use the Sherman-Morrison formula (default is "inv")
    polynomial_basis: str, optional
        polynomial basis used to compute moment matrix, either "monomials", "chebyshev_t_1", "chebyshev_t_2", "chebyshev_u" or "legendre",
        varying this parameter can bring stability to the score in some cases (default is "monomials")
    regularization: str, optional
        one of "vu" (score divided by d^{3p/2}) or "none" (no regularization), "none" is used for cf vs mkde comparison (default is "vu")
    inv: str, optional
        inversion method, one of "inv" for classical matrix inversion or "pinv" for Moore-Penrose pseudo-inversion (default is "inv")

    Methods
    -------
    See BaseDetector methods
    """

    def __init__(self, threshold: float, win_size: int, kernel: str = "gaussian", bandwidth: str = "scott"):
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
        self.assert_shape_unfitted(x)
        self.kc = x[-self.win_size:]
        self.p = x.shape[1]
        self.bsi, self.bdsi = self.be(self.kc)
        return self

    def update(self, x: np.ndarray):
        self.assert_shape_fitted(x)
        self.kc = np.concatenate([self.kc[max(0, len(self.kc) - self.win_size + len(x)):], x[-self.win_size:]])
        self.bsi, self.bdsi = self.be(self.kc)
        return self

    def score_samples(self, x):
        self.assert_shape_fitted(x)
        res = np.zeros(x.shape[0])
        for i, point in enumerate(x):
            ke = self.bdsi * self.kf(np.dot(self.bsi, (self.kc - point).T).T)
            res[i] = np.mean(ke)
        return 1 / (1 + res)

    def decision_function(self, x):
        self.assert_shape_fitted(x)
        return (1 / self.score_samples(x)) - 1 - self.threshold

    def predict(self, x):
        self.assert_shape_fitted(x)
        return np.where(self.decision_function(x) < 0, -1, 1)

    def eval_update(self, x):
        self.assert_shape_fitted(x)
        evals = np.zeros(len(x))
        for i in range(len(x)):
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
        model_bis = KDE(self.threshold, self.win_size, self.kernel, self.bandwidth)
        model_bis.kc = self.kc
        model_bis.bsi = self.bsi
        model_bis.bdsi = self.bdsi
        model_bis.p = self.p
        return model_bis

    def method_name(self):
        return "KDE"


class SmartSifter(BaseDetector):
    """
    Smart Sifter reduced to continuous domain only with its Sequentially Discounting Expectation and Maximizing (SDEM) algorithm
    (see https://github.com/sk1010k/SmartSifter and https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)

    Attributes
    ----------
    threshold: float
        the threshold on the pdf, if the pdf at a point is greater than the threshold then the point is considered normal
    k: int
        number of gaussian mixture components ("n_components" from sklearn.mixture.GaussianMixture)
    r: float
        discounting parameter for the SDEM algorithm ("r" from smartsifter.SDEM)
    alpha: float
        stability parameter for the weights of gaussian mixture components ("alpha" from smartsifter.SDEM)
    scoring_function: str
        scoring function used, either "logloss" for logarithmic loss or "hellinger" for hellinger score, both proposed by the original article,
        or "likelihood" for the likelihood that a point is issued from the learned mixture (default is "likelihood")

    Methods
    -------
    See BaseDetector methods
    """
    def __init__(self, threshold: float, k: int, r: float, alpha: float, scoring_function: str = "likelihood"):
        assert scoring_function in IMPLEMENTED_SS_SCORING_FUNCTIONS.keys()
        self.r = r
        self.alpha = alpha
        self.sdem = SDEM(r, alpha, n_components=k)
        self.scoring_function_str = scoring_function
        self.scoring_function = IMPLEMENTED_SS_SCORING_FUNCTIONS[scoring_function]
        self.p = None
        self.threshold = threshold

    def fit(self, x):
        self.assert_shape_unfitted(x)
        self.sdem.fit(x)
        self.p = x.shape[1]
        return self

    def update(self, x):
        self.assert_shape_fitted(x)
        self.sdem.update(x)
        return self

    def score_samples(self, x):
        self.assert_shape_fitted(x)
        if self.scoring_function_str == "hellinger":
            scores = np.zeros(len(x))
            for i in range(len(x)):
                scores[i], = self.scoring_function(x[i].reshape(1, -1), self.sdem)
            return scores
        elif self.scoring_function_str == "likelihood":
            return 1 / (1 + self.scoring_function(x, self.sdem))
        else:
            return self.scoring_function(x, self.sdem)

    def decision_function(self, x):
        self.assert_shape_fitted(x)
        if self.scoring_function_str == "likelihood":
            return (1 / self.score_samples(x)) - 1 - self.threshold
        else:
            return self.threshold - self.score_samples(x)

    def predict(self, x):
        self.assert_shape_fitted(x)
        return np.where(self.decision_function(x) < 0, -1, 1)

    def eval_update(self, x):
        self.assert_shape_fitted(x)
        evals = np.zeros(len(x))
        if self.scoring_function_str == "hellinger":
            for i in range(len(x)):
                score, new_params = self.scoring_function(x[i].reshape(1, self.p), self.sdem)
                evals[i] = self.threshold - score
                self.covariances_, self.covariances_bar, self.means_, self.means_bar, self.precisions_, self.precisions_cholesky_, self.weights_ = new_params
        else:
            for i, xx in enumerate(x):
                xx.reshape(1, -1)
                evals[i] = self.decision_function(xx.reshape(-1, self.p))
                self.update(xx.reshape(-1, self.p))
        return evals

    def predict_update(self, x):
        self.assert_shape_fitted(x)
        evals = self.eval_update(x)
        return np.where(evals < 0, -1, 1)

    def save_model(self):
        raise NotImplementedError("Not implemented yet.")

    def load_model(self, model_dict: dict):
        raise NotImplementedError("Not implemented yet.")

    def copy(self):
        raise NotImplementedError("The copy method for SmartSifter has not been implemented yet.")

    def method_name(self):
        return "SmartSifter"


class DyCF(BaseDetector):
    """
    Dynamical Christoffel Function

    Attributes
    ----------
    d: int
        the degree of polynomials, usually set between 2 and 8
    incr_opt: str, optional
        can be either "inverse" to inverse the moments matrix each iteration or "sherman" to use the Sherman-Morrison formula (default is "inv")
    polynomial_basis: str, optional
        polynomial basis used to compute moment matrix, either "monomials", "chebyshev_t_1", "chebyshev_t_2", "chebyshev_u" or "legendre",
        varying this parameter can bring stability to the score in some cases (default is "monomials")
    regularization: str, optional
        one of "vu" (score divided by d^{3p/2}), "vu_C" (score divided by d^{3p/2}/C), "comb" (score divided by comb(p+d, d)) or "none" (no regularization), "none" is used for cf vs mkde comparison (default is "vu_C")
    C: float, optional
        define a threshold on the score when used with regularization="vu_C", usually C<=1 (default is 1)
    inv: str, optional
        inversion method, one of "inv" for classical matrix inversion or "pinv" for Moore-Penrose pseudo-inversion (default is "inv")

    Methods
    -------
    See BaseDetector methods
    """

    def __init__(self, d: int, C: float = 1, incr_opt: str = "inverse", polynomial_basis: str = "monomials", regularization: str = "vu", inv: str = "inv"):
        self.N = 0  # number of points integrated in the moments matrix
        self.C = C
        self.p = None
        self.d = d
        self.moments_matrix = MomentsMatrix(d, incr_opt=incr_opt, polynomial_basis=polynomial_basis, inv_opt=inv)
        self.regularization = regularization
        self.regularizer = None

    def fit(self, x: np.ndarray):
        self.assert_shape_unfitted(x)
        self.N = x.shape[0]
        self.p = x.shape[1]
        self.moments_matrix.fit(x)
        if self.regularization == "vu":
            self.regularizer = self.d ** (3 * self.p / 2)
        elif self.regularization == "vu_C":
            self.regularizer = (self.d ** (3 * self.p / 2)) / self.C
        elif self.regularization == "comb":
            self.regularizer = comb(self.d + x.shape[1], x.shape[1])
        else:
            self.regularizer = 1
        return self

    def update(self, x):
        self.assert_shape_fitted(x)
        self.moments_matrix.update(x, self.N)
        self.N += x.shape[0]
        return self

    def score_samples(self, x):
        self.assert_shape_fitted(x)
        return self.moments_matrix.score_samples(x.reshape(-1, self.p)) / self.regularizer

    def decision_function(self, x):
        self.assert_shape_fitted(x)
        return (1 / self.score_samples(x)) - 1

    def predict(self, x):
        self.assert_shape_fitted(x)
        return np.where(self.decision_function(x) < 0, -1, 1)

    def fit_predict(self, x):
        self.assert_shape_fitted(x)
        self.fit(x.reshape(-1, self.p))
        return self.predict(x.reshape(-1, self.p))

    def eval_update(self, x):
        self.assert_shape_fitted(x)
        evals = np.zeros(len(x))
        for i, xx in enumerate(x):
            xx.reshape(1, -1)
            evals[i] = self.decision_function(xx.reshape(-1, self.p))
            self.update(xx.reshape(-1, self.p))
        return evals

    def predict_update(self, x):
        self.assert_shape_fitted(x)
        preds = np.zeros(len(x))
        for i, xx in enumerate(x):
            xx.reshape(1, -1)
            preds[i] = self.predict(xx.reshape(-1, self.p))
            self.update(xx.reshape(-1, self.p))
        return preds

    def save_model(self):
        return {
            "N": self.N,
            "p": self.p,
            "moment_matrix": self.moments_matrix.save_model()
        }

    def load_model(self, model_dict: dict):
        self.N = model_dict["N"]
        self.p = model_dict["p"]
        self.moments_matrix = self.moments_matrix.load_model(model_dict)

    def copy(self):
        c_bis = DyCF(d=self.d)
        c_bis.moments_matrix = self.moments_matrix.copy()
        c_bis.N = self.N
        if self.p is not None:
            c_bis.p = self.p
        if self.regularizer is not None:
            c_bis.regularizer = self.regularizer
        return c_bis

    def method_name(self):
        return "DyCF"


class DyCG(BaseDetector):
    """
    Dynamical Christoffel Growth

    Attributes
    ----------
    degrees: ndarray, optional
        the degrees of two DyCF models inside (default is np.array([2, 8]))
    dycf_kwargs:
        see DyCF args others than d

    Methods
    -------
    See BaseDetector methods
    """

    def __init__(self, degrees: np.ndarray = np.array([2, 8]), **dycf_kwargs):
        assert len(degrees) > 1
        self.degrees = degrees
        self.models = [DyCF(d=d, **dycf_kwargs) for d in self.degrees]
        self.p = None

    def fit(self, x):
        self.assert_shape_unfitted(x)
        self.p = x.shape[1]
        for model in self.models:
            model.fit(x)
        return self

    def update(self, x: np.ndarray):
        self.assert_shape_fitted(x)
        for model in self.models:
            model.update(x)
        return self

    def score_samples(self, x):
        self.assert_shape_fitted(x)
        score = np.zeros((len(x), 1))
        for i, d in enumerate(x):
            d_ = d.reshape(1, -1)
            scores = np.array([m.score_samples(d_)[0] for m in self.models])
            s_diff = np.diff(scores) / np.diff(self.degrees)
            score[i] = np.mean(s_diff)
        return score

    def decision_function(self, x):
        self.assert_shape_fitted(x)
        return -1 * self.score_samples(x)

    def predict(self, x):
        self.assert_shape_fitted(x)
        return np.where(self.decision_function(x) < 0, -1, 1)

    def eval_update(self, x):
        self.assert_shape_fitted(x)
        evals = np.zeros(len(x))
        for i, d in enumerate(x):
            d_ = d.reshape(1, -1)
            evals[i] = self.decision_function(d_)
            self.update(d_)
        return evals

    def predict_update(self, x):
        self.assert_shape_fitted(x)
        preds = np.zeros(len(x))
        for i, d in enumerate(x):
            d_ = d.reshape(1, -1)
            preds[i] = self.predict(d_)
            self.update(d_)
        return preds

    def save_model(self):
        raise NotImplementedError("Not implemented yet.")

    def load_model(self, model_dict: dict):
        raise NotImplementedError("Not implemented yet.")

    def copy(self):
        mc_bis = DyCG(degrees=self.degrees)
        mc_bis.models = [model.copy() for model in self.models]
        return mc_bis

    def method_name(self):
        return "DyCG"
