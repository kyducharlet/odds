from typing import Union
from .base import BaseDetector
from .utils import np
from .utils import IMPLEMENTED_BANDWIDTH_ESTIMATORS, neighbours_count, neighbours_counts_in_grid
from .utils import MomentsMatrix, update_params, compute_R
from .utils import RStarTree
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


class ILOF(BaseDetector):
    """
    Incremental Local Outlier Factor

    Attributes
    ----------
    k: int
        the number of neighbors to compute the LOF score on
    threshold: float
        a threshold on the LOF score to separate normal points and outliers
    win_size: int
        number of points in the sliding window used in kNN search
    min_size: int (optional)
        minimal number of points in a node, it is mandatory that 2 <= min_size <= max_size / 2 (default is 3)
    max_size: int (optional)
        maximal number of points in a node, it is mandatory that 2 <= min_size <= max_size / 2 (default is 12)
    p_reinsert_tol: int (optional)
        tolerance on reinsertion, used to deal with overflow in a node (default is 4)
    reinsert_strategy: str (optional)
        either "close" or "far", tells if we try to reinsert in the closest rectangles first on in the farthest (default is "close")

    Methods
    -------
    See BaseDetector methods
    """

    def __init__(self, k: int, threshold: float, win_size: int, min_size: int = 3, max_size: int = 12, p_reinsert_tol: int = 4, reinsert_strategy: str = "close"):
        self.k = k
        self.threshold = threshold
        self.win_size = win_size
        self.rst = RStarTree(k, min_size, max_size, p_reinsert_tol, reinsert_strategy)
        self.p = None

    def fit(self, x):
        self.assert_shape_unfitted(x)
        self.p = x.shape[1]
        x_fit = x[-self.win_size:]
        added_objects = []
        for xx in x_fit:
            added_objects.append(self.rst.insert_data(xx.reshape(1, -1)))
        self.__fit_metrics_compute__(added_objects)

    def update(self, x):
        self.assert_shape_fitted(x)
        if len(self.rst.objects) < self.win_size:
            x_add, x_update = x[:self.win_size - len(self.rst.objects)], x[self.win_size - len(self.rst.objects):]
        else:
            x_add, x_update = x[:0], x[0:]
        objects = []
        for xx in x_add:
            obj = self.rst.insert_data(xx.reshape(1, -1))
            self.__update_metrics_addition__(obj)
            objects.append(obj)
        for xx in x_update:
            obj, = self.rst.remove_oldest(1)
            self.__update_metrics_deletion__(obj)
            obj = self.rst.insert_data(xx.reshape(1, -1))
            self.__update_metrics_addition__(obj)
            objects.append(obj)
        return objects

    def score_samples(self, x):
        self.assert_shape_fitted(x)
        lof_scores = []
        for xx in x:
            kNNs = self.rst.search_kNN(xx.reshape(1, -1))
            rds = [max(np.linalg.norm(xx.reshape(1, -1) - o.low), o.__dict__["__k_dist__"]) for o in kNNs]
            lrd = 1 / np.mean(rds)
            lof_scores.append(np.mean([o.__dict__["__lrd__"] for o in kNNs]) / lrd)
        return np.array(lof_scores)

    def decision_function(self, x):
        self.assert_shape_fitted(x)
        return self.threshold - self.score_samples(x)

    def predict(self, x):
        self.assert_shape_fitted(x)
        return np.where(self.decision_function(x) < 0, -1, 1)

    def eval_update(self, x):
        self.assert_shape_fitted(x)
        evals = np.zeros(len(x))
        for i in range(len(x)):
            obj, = self.update(x[i].reshape(1, -1))
            evals[i] = self.threshold - obj.__dict__["__lof__"]
        return evals

    def predict_update(self, x):
        self.assert_shape_fitted(x)
        preds = np.zeros(len(x))
        for i in range(len(x)):
            obj, = self.update(x[i].reshape(1, -1))
            preds[i] = -1 if self.threshold - obj.__dict__["__lof__"] < 0 else 1
        return preds

    def __fit_metrics_compute__(self, objects_list):
        """ Method used to compute all LOF metrics when fitting data """
        for obj in objects_list:
            obj.__dict__["__kNNs__"] = self.rst.search_kNN(obj)
            obj.__dict__["__k_dist__"] = obj.__compute_dist__(obj.__dict__["__kNNs__"][-1])
            obj.parent.__update_k_dist__(obj)
        for obj in objects_list:
            obj.__dict__["__rds__"] = []
            if obj.__dict__.get("__RkNNs__") is None:
                obj.__dict__["__RkNNs__"] = []
            for p in obj.__dict__["__kNNs__"]:
                obj.__dict__["__rds__"].append(max(obj.__compute_dist__(p), p.__dict__["__k_dist__"]))
                if p.__dict__.get("__RkNNs__") is None:
                    p.__dict__["__RkNNs__"] = [obj]
                else:
                    p.__dict__["__RkNNs__"].append(obj)
            with np.errstate(divide='ignore'):
                obj.__dict__["__lrd__"] = np.power(np.mean(obj.__dict__["__rds__"]), -1)
        for obj in objects_list:
            obj.__dict__["__lof__"] = np.mean([p.__dict__["__lrd__"] for p in obj.__dict__["__kNNs__"]]) / obj.__dict__["__lrd__"] if obj.__dict__["__lrd__"] < np.infty else 0

    def __update_metrics_addition__(self, obj):
        """ Method used to adjust all LOF metrics when adding a new point """
        # Set obj kNNs
        kNNs = self.rst.search_kNN(obj)
        obj.__dict__["__kNNs__"] = kNNs
        # Set obj k-distance
        obj.__dict__["__k_dist__"] = obj.__compute_dist__(kNNs[-1])
        obj.parent.__update_k_dist__(obj)
        # Set obj RkNNs
        S_update_k_distance = self.rst.search_RkNN(obj)
        obj.__dict__["__RkNNs__"] = S_update_k_distance
        # Update k-distance (if required), kNNs and rds
        S_k_distance_updated = []
        for o in S_update_k_distance:
            if obj.__compute_dist__(o) == o.__dict__["__k_dist__"]:
                # The k-distance will not change
                o.__dict__["__kNNs__"].append(obj)
                o.__dict__["__rds__"].append(max(obj.__compute_dist__(o), obj.__dict__["__k_dist__"]))
                with np.errstate(divide='ignore'):
                    o.__dict__["__lrd__"] = 1 / np.mean(o.__dict__["__rds__"])
            elif obj.__compute_dist__(o) < o.__dict__["__k_dist__"]:
                # The k-distance may change
                new_kNNs = o.__dict__["__kNNs__"] + [obj]
                new_rds = o.__dict__["__rds__"] + [max(obj.__compute_dist__(o), obj.__dict__["__k_dist__"])]
                o_new_metrics = sorted([(o.__compute_dist__(p), p, new_rds[i]) for i, p in enumerate(new_kNNs)], key=lambda elt: elt[0])
                if len([elt for elt in o_new_metrics if elt[0] < o.__dict__["__k_dist__"]]) < self.k:  # The old kthNN will remain and so the k-distance will not change
                    o.__dict__["__kNNs__"] = [elt[1] for elt in o_new_metrics]
                    o.__dict__["__rds__"] = [elt[2] for elt in o_new_metrics]
                else:
                    for old_kNN in [elt[1] for elt in o_new_metrics if elt[0] >= o.__dict__["__k_dist__"]]:  # The old kthNN has to be discarded and the k-distance will change
                        old_kNN.__dict__["__RkNNs__"].remove(o)
                    o.__dict__["__kNNs__"] = [elt[1] for elt in o_new_metrics if elt[0] < o.__dict__["__k_dist__"]]
                    o.__dict__["__rds__"] = [elt[2] for elt in o_new_metrics if elt[0] < o.__dict__["__k_dist__"]]
                    o.__dict__["__k_dist__"] = [elt[0] for elt in o_new_metrics if elt[0] < o.__dict__["__k_dist__"]][-1]
                o.parent.__update_k_dist__(o)
                S_k_distance_updated.append(o)
            else:  # theoretically impossible unless search_RkNN does not work properly
                raise ValueError()
        # Compute obj rds and add obj to RkNNs of its kNNs
        obj.__dict__["__rds__"] = []
        for p in kNNs:
            obj.__dict__["__rds__"].append(max(obj.__compute_dist__(p), p.__dict__["__k_dist__"]))
            p.__dict__["__RkNNs__"].append(obj)
        # Update rds
        S_update_lrd = 1 * S_update_k_distance
        for o in S_k_distance_updated:
            for kNN in o.__dict__["__kNNs__"]:
                if kNN != obj and o in kNN.__dict__["__kNNs__"]:
                    S_update_lrd.append(kNN)
                    kNN.__dict__["__rds__"][kNN.__dict__["__kNNs__"].index(o)] = o.__dict__["__k_dist__"]
        # Update lrds (no need to update lof as we do not follow its evolution)
        S_update_lrd = list(set(S_update_lrd))
        for o in S_update_lrd:
            with np.errstate(divide='ignore'):
                o.__dict__["__lrd__"] = 1 / np.mean(o.__dict__["__rds__"])
        # Compute lrd
        with np.errstate(divide='ignore'):
            obj.__dict__["__lrd__"] = 1 / np.mean(obj.__dict__["__rds__"])
        # Compute lof
        obj.__dict__["__lof__"] = np.mean([o.__dict__["__lrd__"] for o in kNNs]) / obj.__dict__["__lrd__"] if obj.__dict__["__lrd__"] < np.infty else 0

    def __update_metrics_deletion__(self, obj):
        """ Method used to adjust all LOF metrics when removing a point """
        obj.parent.__update_k_dist__(obj)
        S_update_k_distance = [o for o in obj.__dict__["__RkNNs__"] if o in self.rst.objects]
        # Remove obj from the RkNNs list of its kNNs
        for o in obj.__dict__["__kNNs__"]:
            o.__dict__["__RkNNs__"].remove(obj)
        # Update k-distances
        S_k_distance_updated = []
        for o in S_update_k_distance:
            if len(o.__dict__["__kNNs__"]) > self.k:
                # We can remove obj without updating the k-distance
                o.__dict__["__rds__"].pop(o.__dict__["__kNNs__"].index(obj))
                o.__dict__["__kNNs__"].remove(obj)
                with np.errstate(divide='ignore'):
                    o.__dict__["__lrd__"] = 1 / np.mean(o.__dict__["__rds__"])
            else:
                # We need to update the k-distance with a new kthNN
                old_kNNs = o.__dict__["__kNNs__"]
                o.__dict__["__kNNs__"] = self.rst.search_kNN(o)
                o.__dict__["__k_dist__"] = o.__compute_dist__(o.__dict__["__kNNs__"][-1])
                o.parent.__update_k_dist__(o)
                # o needs to be added to the RkNNs of its new kNNs
                new_kNNs = [kNN for kNN in o.__dict__["__kNNs__"] if kNN not in old_kNNs]
                for new_kNN in new_kNNs:
                    new_kNN.__dict__["__RkNNs__"].append(o)
                # o need to be removed from the RkNNs of its old kNNs
                removed_kNNs = [kNN for kNN in old_kNNs if kNN not in o.__dict__["__kNNs__"]]
                for removed_kNN in removed_kNNs:
                    if o in removed_kNN.__dict__["__RkNNs__"]:
                        removed_kNN.__dict__["__RkNNs__"].remove(o)
                o.__dict__["__rds__"] = [max(o.__compute_dist__(p), p.__dict__["__k_dist__"]) for p in o.__dict__["__kNNs__"]]
                S_k_distance_updated.append(o)
        # Update rds
        S_update_lrd = 1 * S_k_distance_updated
        for o in S_k_distance_updated:
            for old_kNN in o.__dict__["__kNNs__"][:-1 - (len(o.__dict__["__kNNs__"]) - self.k)]:
                if o in old_kNN.__dict__["__kNNs__"]:
                    S_update_lrd.append(old_kNN)
                    old_kNN.__dict__["__rds__"][old_kNN.__dict__["__kNNs__"].index(o)] = o.__dict__["__k_dist__"]
        # Update lrds
        S_update_lrd = list(set(S_update_lrd))
        for o in S_update_lrd:
            o.__dict__["__lrd__"] = 1 / np.mean(o.__dict__["__rds__"])

    def copy(self):
        ilof_bis = ILOF(self.k, self.threshold, self.win_size)
        ilof_bis.rst = self.rst.copy()
        ilof_bis.p = self.p

    def method_name(self):
        return "ILOF"
