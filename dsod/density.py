from typing import Union

import time

from .base import BaseDetector, NotFittedError
from .utils import np, linalg, optimize
from .utils import RStarTree
from .utils import DILOFPoint, sigmoid


class ILOF(BaseDetector):
    def __init__(self, k: int, threshold=1.1, win_size: Union[type(None), int] = None,
                 # R*Tree parameters:
                 min_size=3, max_size=12, p_reinsert_tol=4, reinsert_strategy="close", n_trim_iterations=3):
        self.threshold = threshold
        self.k = k
        self.win_size = win_size
        if self.win_size is None:
            self.select_update_f = self.__select_update_without_window__
            self.select_fit_f = self.__select_fit_without_window__
        else:
            self.select_update_f = self.__select_update_with_window__
            self.select_fit_f = self.__select_fit_with_window__
        self.rst = RStarTree(k, min_size, max_size, p_reinsert_tol, reinsert_strategy, n_trim_iterations)
        self.p = None

    def fit(self, x):
        if len(x.shape) != 2:
            raise ValueError("The expected array shape: (N, p) do not match the given array shape: {}".format(x.shape))
        x_fit = self.select_fit_f(x)
        added_objects = []
        for xx in x_fit:
            added_objects.append(self.rst.insert_data(xx.reshape(1, -1)))
        self.__fit_metrics_compute__(added_objects)

    def update(self, x):
        if len(x.shape) != 2:
            raise ValueError("The expected array shape: (N, p) do not match the given array shape: {}".format(x.shape))
        x_add, x_update = self.select_update_f(x)
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
            self.__check_consistency__()
        return objects

    def score_samples(self, x):
        lof_scores = []
        for xx in x:
            kNNs = self.rst.search_kNN(xx.reshape(1, -1))
            rds = [max(linalg.norm(xx.reshape(1, -1) - o.low), o.__dict__["__k_dist__"]) for o in kNNs]
            lrd = 1 / np.mean(rds)
            lof_scores.append(np.mean([o.__dict__["__lrd__"] for o in kNNs]) / lrd)
        return np.array(lof_scores)

    def decision_function(self, x):
        return self.threshold - self.score_samples(x)

    def predict(self, x):
        return np.where(self.decision_function(x) < 0, -1, 1)

    def eval_update(self, x):
        evals = np.zeros(len(x))
        for i in range(len(x)):
            obj, = self.update(x[i].reshape(1, -1))
            evals[i] = self.threshold - obj.__dict__["__lof__"]
        return evals

    def predict_update(self, x):
        preds = np.zeros(len(x))
        for i in range(len(x)):
            obj, = self.update(x[i].reshape(1, -1))
            preds[i] = -1 if self.threshold - obj.__dict__["__lof__"] < 0 else 1
        return preds

    def __select_fit_with_window__(self, x):
        return x[-self.win_size:]

    def __select_fit_without_window__(self, x):
        return x

    def __select_update_with_window__(self, x):
        if len(self.rst.objects) < self.win_size:
            return x[:self.win_size - len(self.rst.objects)], x[self.win_size - len(self.rst.objects):]
        else:
            return x[:0], x[0:]

    def __select_update_without_window__(self, x):
        return x[0:], x[:0]

    def __fit_metrics_compute__(self, objects_list):
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
        ### Set obj kNNs
        kNNs = self.rst.search_kNN(obj)
        obj.__dict__["__kNNs__"] = kNNs
        ### Set obj k-distance
        obj.__dict__["__k_dist__"] = obj.__compute_dist__(kNNs[-1])
        obj.parent.__update_k_dist__(obj)
        ### Set obj RkNNs
        S_update_k_distance = self.rst.search_RkNN(obj)
        # S_update_k_distance = self.__search_RkNNs__(obj)
        obj.__dict__["__RkNNs__"] = S_update_k_distance
        ### Update k-distance (if required), kNNs and rds
        S_k_distance_updated = []
        for o in S_update_k_distance:
            if obj.__compute_dist__(o) == o.__dict__["__k_dist__"]:
                ### The k-distance will not change
                o.__dict__["__kNNs__"].append(obj)
                o.__dict__["__rds__"].append(max(obj.__compute_dist__(o), obj.__dict__["__k_dist__"]))
                with np.errstate(divide='ignore'):
                    o.__dict__["__lrd__"] = 1 / np.mean(o.__dict__["__rds__"])
            else:
                ### The k-distance may change
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
        ### Compute obj rds and add obj to RkNNs of its kNNs
        obj.__dict__["__rds__"] = []
        for p in kNNs:
            obj.__dict__["__rds__"].append(max(obj.__compute_dist__(p), p.__dict__["__k_dist__"]))
            p.__dict__["__RkNNs__"].append(obj)
        ### Update rds
        S_update_lrd = 1 * S_update_k_distance
        for o in S_k_distance_updated:
            for kNN in o.__dict__["__kNNs__"]:
                if kNN != obj and o in kNN.__dict__["__kNNs__"]:
                    S_update_lrd.append(kNN)
                    kNN.__dict__["__rds__"][kNN.__dict__["__kNNs__"].index(o)] = o.__dict__["__k_dist__"]
        ### Update lrds (no need to update lof as we do not follow its evolution)
        S_update_lrd = list(set(S_update_lrd))
        for o in S_update_lrd:
            with np.errstate(divide='ignore'):
                o.__dict__["__lrd__"] = 1 / np.mean(o.__dict__["__rds__"])
        ### Compute lrd
        with np.errstate(divide='ignore'):
            obj.__dict__["__lrd__"] = 1 / np.mean(obj.__dict__["__rds__"])
        ### Compute lof
        obj.__dict__["__lof__"] = np.mean([o.__dict__["__lrd__"] for o in kNNs]) / obj.__dict__["__lrd__"] if obj.__dict__["__lrd__"] < np.infty else 0

    def __update_metrics_deletion__(self, obj):
        """if obj.low[0, 0] == -0.09589041095890405 and obj.low[0, 1] == 0.7615992662100013:
            pass"""
        obj.parent.__update_k_dist__(obj)
        S_update_k_distance = [o for o in obj.__dict__["__RkNNs__"] if o in self.rst.objects]
        ### Remove obj from the RkNNs list of its kNNs
        for o in obj.__dict__["__kNNs__"]:
            o.__dict__["__RkNNs__"].remove(obj)
        ### Update k-distances
        S_k_distance_updated = []
        for o in S_update_k_distance:
            if len(o.__dict__["__kNNs__"]) > self.k:
                ### We can remove obj without updating the k-distance
                o.__dict__["__rds__"].pop(o.__dict__["__kNNs__"].index(obj))
                o.__dict__["__kNNs__"].remove(obj)
                with np.errstate(divide='ignore'):
                    o.__dict__["__lrd__"] = 1 / np.mean(o.__dict__["__rds__"])
            else:
                ### We need to update the k-distance with a new kthNN
                old_kNNs = o.__dict__["__kNNs__"]
                o.__dict__["__kNNs__"] = self.rst.search_kNN(o)
                o.__dict__["__k_dist__"] = o.__compute_dist__(o.__dict__["__kNNs__"][-1])
                o.parent.__update_k_dist__(o)
                new_kNNs = [kNN for kNN in o.__dict__["__kNNs__"] if kNN not in old_kNNs]
                for new_kNN in new_kNNs:
                    new_kNN.__dict__["__RkNNs__"].append(o)
                # self.__check_consistency__()
                o.__dict__["__rds__"] = [max(o.__compute_dist__(p), p.__dict__["__k_dist__"]) for p in o.__dict__["__kNNs__"]]
                S_k_distance_updated.append(o)
        ### Update rds
        S_update_lrd = 1 * S_k_distance_updated
        for o in S_k_distance_updated:
            for old_kNN in o.__dict__["__kNNs__"][:-1 - (len(o.__dict__["__kNNs__"]) - self.k)]:
                if o in old_kNN.__dict__["__kNNs__"]:
                    S_update_lrd.append(old_kNN)
                    old_kNN.__dict__["__rds__"][old_kNN.__dict__["__kNNs__"].index(o)] = o.__dict__["__k_dist__"]
        ### Update lrds
        S_update_lrd = list(set(S_update_lrd))
        for o in S_update_lrd:
            o.__dict__["__lrd__"] = 1 / np.mean(o.__dict__["__rds__"])
        # self.__check_consistency__()

    def __search_RkNNs__(self, obj):
        RkNNs = []
        for o in self.rst.objects:
            if o.__compute_dist__(obj) <= o.__dict__["__k_dist__"] and o != obj:
                RkNNs.append(o)
        return RkNNs

    def __check_consistency__(self):
        for obj in self.rst.objects:
            if obj not in obj.parent.children:
                raise ValueError
            """for kNN in self.rst.search_kNN(obj):
                if kNN not in obj.__kNNs__:
                    raise ValueError"""
            """for RkNN in self.rst.search_RkNN(obj):
                if RkNN not in obj.__RkNNs__:
                    raise ValueError"""
            """for kNN in obj.__dict__["__kNNs__"]:
                if obj not in kNN.__dict__["__RkNNs__"]:
                    raise ValueError"""
            """for RkNN in obj.__dict__["__RkNNs__"]:
                if obj not in RkNN.__dict__["__kNNs__"]:
                    raise ValueError"""

    def __check_kNNs_size__(self):
        for obj in self.rst.objects:
            if len(obj.__kNNs__) < self.k:
                raise ValueError

    def copy(self):
        pass


class DILOF(BaseDetector):
    def __init__(self, k, threshold, win_size: int, step_size=0.3, reg_const=1, max_iter=100, use_Ckn=True):
        self.k = k
        self.threshold = threshold
        self.win_size = win_size
        self.step_size = step_size
        self.reg_const = reg_const
        self.max_iter = max_iter
        self.use_Ckn = use_Ckn
        self.points = None
        self.p = None

    def fit(self, x: np.ndarray):
        if len(x.shape) != 2:
            raise ValueError("The expected array shape: (N, p) do not match the given array shape: {}".format(x.shape))
        self.p = x.shape[1]
        self.points = []
        self.__fit_lof__(x[:self.win_size])
        for xx in x[self.win_size:]:
            point = DILOFPoint(xx)
            self.__dilof__(point)

    def update(self, x):
        if len(x.shape) != 2:
            raise ValueError("The expected array shape: (N, p) do not match the given array shape: {}".format(x.shape))
        for xx in x:
            point = DILOFPoint(xx)
            self.__dilof__(point)

    def score_samples(self, x):
        if len(x.shape) != 2:
            raise ValueError("The expected array shape: (N, p) do not match the given array shape: {}".format(x.shape))
        scores = np.zeros(len(x))
        for i, xx in enumerate(x):
            point = DILOFPoint(xx)
            point.compute_without_updates(self.points, self.k)
            scores[i] = point.lof
        return scores

    def decision_function(self, x):
        return self.threshold - self.score_samples(x)

    def predict(self, x):
        return np.where(self.decision_function(x) < 0, -1, 1)

    def eval_update(self, x):
        if len(x.shape) != 2:
            raise ValueError("The expected array shape: (N, p) do not match the given array shape: {}".format(x.shape))
        evals = np.zeros(len(x))
        for i, xx in enumerate(x):
            point = DILOFPoint(xx)
            self.__dilof__(point)
            evals[i] = self.threshold - point.lof
        return evals

    def predict_update(self, x):
        if len(x.shape) != 2:
            raise ValueError("The expected array shape: (N, p) do not match the given array shape: {}".format(x.shape))
        preds = np.zeros(len(x))
        for i, xx in enumerate(x):
            point = DILOFPoint(xx)
            self.__dilof__(point)
            preds[i] = -1 if point.lof > self.threshold else 1
        return preds

    def __fit_lof__(self, data):
        for xx in data:
            self.points.append(DILOFPoint(xx))
        for p in self.points:
            p.compute_kNNs_kdist(self.points, self.k)
        for p in self.points:
            p.compute_rds_lrd()
        for p in self.points:
            p.compute_lof()

    def __dilof__(self, point):
        self.__lod__(point)
        self.points.append(point)
        if len(self.points) >= self.win_size:
            self.points = self.__nds__() + self.points[self.win_size//2:]

    def __nds__(self):
        if self.use_Ckn:
            return self.__nds_using_Ckn__()
        else:
            return self.__nds_without_Ckn__()

    def __nds_using_Ckn__(self):
        y = 0.5 * np.ones(self.win_size // 2)
        ss = self.step_size
        nus = [p.get_local_kdist(self.points[:self.win_size//2], self.k) for p in self.points[:self.win_size//2]]
        sum_exp_lof = np.sum([np.exp(sigmoid(p.lof)) for p in self.points[:self.win_size//2]])
        betas = [np.sum([np.exp(sigmoid(q.lof)) for q in p.kNNs]) / sum_exp_lof for p in self.points[:self.win_size//2]]
        rhos = [nus[i] + betas[i] * np.max([linalg.norm(p.values - q.values) - nus[i] for q in self.points if q != p]) for (i, p) in enumerate(self.points[:self.win_size//2])]
        C = self.__estimate_Ckns__(nus)
        for i in range(self.max_iter):
            new_y = y.copy()
            ss = 0.95 * ss
            for j in range(len(y)):
                p = self.points[j]
                if y[j] > 1:
                    psi = 2 * (y[j] - 1)
                elif y[j] < 0:
                    psi = 2 * y[j]
                else:
                    psi = 0
                new_y[j] = y[j] - ss * (np.sum([linalg.norm(p.values - self.points[:self.win_size//2][n].values) / nus[n] for n in C[j]]) + (rhos[j] / nus[j]) - p.lof + psi + self.reg_const * (np.sum(y) - (self.win_size / 4)))
            y = new_y
        selected_points = np.argsort(y)[-self.win_size//4:]
        return list(np.array(self.points)[selected_points])

    def __nds_without_Ckn__(self):
        lofs = np.array([p.lof for p in self.points[:self.win_size//2]])
        nus = np.array([p.get_local_kdist(self.points[:self.win_size//2], self.k) for p in self.points[:self.win_size//2]])
        sum_exp_lof = np.sum([np.exp(sigmoid(p.lof)) for p in self.points[:self.win_size//2]])
        betas = [np.sum([np.exp(sigmoid(q.lof)) for q in p.kNNs]) / sum_exp_lof for p in self.points[:self.win_size//2]]
        rhos = np.array([nus[i] + betas[i] * np.max([linalg.norm(p.values - q.values) - nus[i] for q in self.points if q != p]) for (i, p) in enumerate(self.points[:self.win_size//2])])
        def objective_func(y, rhos, nus, lofs, W):
            psi = np.zeros(len(y))
            psi[np.where(y > 1)] = np.square(y[np.where(y > 1)] - 1)
            psi[np.where(y < 0)] = np.square(y[np.where(y < 0)])
            return np.dot(y, (rhos / nus) - lofs) + np.sum(psi) + (1 / 2) * np.square(np.sum(y) - (W / 4))
        res = optimize.minimize(objective_func, x0=0.5*np.ones(self.win_size//2), args=(rhos, nus, lofs, self.win_size))
        selected_points = np.argsort(res.x)[-self.win_size//4:]
        return list(np.array(self.points)[selected_points])

    def __estimate_Ckns__(self, nus):
        s = [np.sum([np.exp(sigmoid(q.lof)) for q in p.kNNs]) for p in self.points[:self.win_size//2]]
        s_ = np.mean(s)
        C = [[] for p in self.points[:self.win_size//2]]
        for i in range(self.win_size // 2):
            p = self.points[i]
            if s[i] > s_:
                for j in range(self.win_size // 2):
                    if nus[i] < linalg.norm(p.values - self.points[j].values) < 2 * np.exp(sigmoid(p.lof)) * nus[i]:
                        C[j].append(i)
                        break
        return C

    def __lod__(self, point):
        point.compute_with_updates(self.points, self.k)

    def copy(self):
        raise NotImplementedError("Copy has not been implemented yet for DILOF.")
