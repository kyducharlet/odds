import numpy as np
from typing import Union

import time

from .base import BaseDetector, NotFittedError
from .utils import search_kNN, compute_k_distance, compute_rd, compute_lrd, compute_lof, update_when_adding, update_when_removing
from .utils import RStarTree


class ILOF(BaseDetector):
    def __init__(self, k: int, threshold=1.1, win_size: Union[type(None), int] = None):
        self.threshold = threshold
        self.k = k
        self.win_size = win_size
        if self.win_size is None:
            self.update_f = self.__update_without_window__
            self.select_f = self.__select_without_window__
        else:
            self.update_f = self.__update_with_window__
            self.select_f = self.__select_with_window__
        self.points = None
        self.k_distances = None
        self.kNNs = None
        self.rds = None
        self.lrds = None
        self.lofs = None
        self.p = None

    def fit(self, x):
        if len(x.shape) != 2:
            raise ValueError("The expected array shape: (N, p) do not match the given array shape: {}".format(x.shape))
        x_ = self.select_f(x)
        self.p = x_.shape[1]
        self.points = {i: x_[i] for i in range(len(x_))}
        self.kNNs = {i: search_kNN(self.points, i, self.k) for i in range(len(x_))}
        self.k_distances = {i: compute_k_distance(self.points, i, self.kNNs) for i in range(len(x_))}
        self.rds = {i: {o: compute_rd(self.points, i, o, self.k_distances) for o in list(self.kNNs[i].keys())} for i in range(len(x_))}
        self.lrds = {i: compute_lrd(self.rds, i) for i in range(len(x_))}
        self.lofs = {i: compute_lof(self.lrds, self.kNNs, i) for i in range(len(x_))}
        return self

    def update(self, x):
        if len(x.shape) != 2 or x.shape[1] != self.p:
            raise ValueError("The expected array shape: (N, {}) do not match the given array shape: {}".format(self.p, x.shape))
        if self.points is None:
            raise NotFittedError("This ILOF instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        x_ = self.select_f(x)
        for x_new in x_:
            self.update_f(x_new)
        return self

    def score_samples(self, x):
        if len(x.shape) != 2 or x.shape[1] != self.p:
            raise ValueError("The expected array shape: (N, {}) do not match the given array shape: {}".format(self.p, x.shape))
        if self.points is None:
            raise NotFittedError("This ILOF instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        lofs = np.zeros(len(x))
        for i, point in enumerate(x):
            temp_points = self.points.copy()
            temp_points[-1] = point
            kNN = search_kNN(temp_points, -1, self.k)
            rds = {-1: {o: compute_rd(temp_points, -1, o, self.k_distances) for o in list(kNN.keys())}}
            temp_lrd = self.lrds.copy()
            temp_lrd[-1] = compute_lrd(rds, -1)
            lofs[i] = compute_lof(temp_lrd, {-1: kNN}, -1)
        return lofs

    def decision_function(self, x):
        return self.threshold - self.score_samples(x)

    def predict(self, x):
        return np.where(self.decision_function(x) < 0, -1, 1)

    def eval_update(self, x):
        evals = np.zeros(len(x))
        for i in range(len(x)):
            self.update(x[i].reshape(1, -1))
            evals[i] = self.threshold - self.lofs[max(list(self.lofs.keys()))]
        return evals

    def predict_update(self, x):
        preds = np.zeros(len(x))
        for i in range(len(x)):
            self.update(x[i].reshape(1, -1))
            preds[i] = -1 if self.threshold - self.lofs[max(list(self.lofs.keys()))] < 0 else 1
        return preds

    def __select_with_window__(self, x):
        x_windowed = x[-self.win_size:]
        return x_windowed

    def __select_without_window__(self, x):
        return x

    def __update_with_window__(self, x):
        index = max(list(self.points.keys())) + 1
        while len(self.points) >= self.win_size:  # remove points if we exceed the win_size
            self.__remove_point__(min(list(self.points.keys())))
        self.__add_point__(x, index)

    def __update_without_window__(self, x):
        index = list(self.points.keys())[-1] + 1
        self.__add_point__(x, index)

    def __add_point__(self, x, index):
        if len(x.shape) == 1 and len(x) == self.p:
            x_new = x
        elif len(x.shape) == 2 and x.shape[1] == self.p:
            x_new = x.reshape(-1)
        else:
            raise ValueError("The expected array shape: (N, {}) do not match the given array shape: {}".format(self.p, x.shape))

        self.kNNs, self.k_distances, self.rds, self.lrds, self.lofs = \
            update_when_adding(self.points, index, x_new, self.kNNs, self.k_distances, self.rds, self.lrds, self.lofs, self.k)

    def __remove_point__(self, index):
        self.kNNs, self.k_distances, self.rds, self.lrds, self.lofs = \
            update_when_removing(self.points, index, self.kNNs, self.k_distances, self.rds, self.lrds, self.lofs, self.k)

    def copy(self):
        model_bis = ILOF(self.k, self.win_size)
        model_bis.points = self.points
        model_bis.k_distances = self.k_distances
        model_bis.kNNs = self.kNNs
        model_bis.rds = self.rds
        model_bis.lrds = self.lrds
        model_bis.lofs = self.lofs
        model_bis.p = self.p
        return model_bis


class ILOFv2(BaseDetector):
    def __init__(self, k: int, threshold=1.1, win_size: Union[type(None), int] = None, min_size=3, max_size=12, p_reinsert_tol=4, reinsert_strategy="close", n_trim_iterations=3):
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
        return objects

    def score_samples(self, x):
        lof_scores = []
        for xx in x:
            kNNs = self.rst.search_kNN(xx.reshape(1, -1))
            rds = [max(np.linalg.norm(xx.reshape(1, -1) - o.low), o.__dict__["__k_dist__"]) for o in kNNs]
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
            obj.__dict__["__lrd__"] = 1 / np.mean(obj.__dict__["__rds__"])
        for obj in objects_list:
            obj.__dict__["__lof__"] = np.mean([p.__dict__["__lrd__"] for p in obj.__dict__["__kNNs__"]]) / obj.__dict__["__lrd__"]

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
                o.__dict__["__lrd__"] = 1 / np.mean(o.__dict__["__rds__"])
            else:
                ### The k-distance will change as the kthNN will
                new_kNNs = o.__dict__["__kNNs__"] + [obj]
                new_rds = o.__dict__["__rds__"] + [max(obj.__compute_dist__(o), obj.__dict__["__k_dist__"])]
                o_new_metrics = sorted([(o.__compute_dist__(p), p, new_rds[i]) for i, p in enumerate(new_kNNs)], key=lambda elt: elt[0])
                for old_kNN in [elt[1] for elt in o_new_metrics if elt[0] >= o.__dict__["__k_dist__"]]:
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
            o.__dict__["__lrd__"] = 1 / np.mean(o.__dict__["__rds__"])
        ### Compute lrd
        obj.__dict__["__lrd__"] = 1 / np.mean(obj.__dict__["__rds__"])
        ### Compute lof
        obj.__dict__["__lof__"] = np.mean([o.__dict__["__lrd__"] for o in kNNs]) / obj.__dict__["__lrd__"]

    def __update_metrics_deletion__(self, obj):
        obj.parent.__update_k_dist__(obj)
        S_update_k_distance = obj.__dict__["__RkNNs__"]
        ### Remove obj from the RkNNs list of its kNNs
        for o in obj.__dict__["__kNNs__"]:
            o.__dict__["__RkNNs__"].remove(obj)
        ### Update k-distances
        S_k_distance_updated = []
        for o in S_update_k_distance:
            if len(o.__dict__["__kNNs__"]) > self.k:
                ### We can remove obj without updating the k-distance
                o.__dict__["__rds__"].pop(o.__dict__["__kNNs__"].index(obj))
                o.__dict__["__kNNs__"].pop(obj)
                o.__dict__["__lrd__"] = 1 / np.mean(o.__dict__["__rds__"])
            else:
                ### We need to update the k-distance with a new kthNN
                o.__dict__["__kNNs__"] = self.rst.search_kNN(o)
                o.__dict__["__k_dist__"] = o.__compute_dist__(o.__dict__["__kNNs__"][-1])
                o.parent.__update_k_dist__(o)
                new_kNNs = o.__dict__["__kNNs__"][-1 - (len(o.__dict__["__kNNs__"]) - self.k):]
                for new_kNN in new_kNNs:
                    new_kNN.__dict__["__RkNNs__"].append(o)
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

    def __search_RkNNs__(self, obj):
        RkNNs = []
        for o in self.rst.objects:
            if o.__compute_dist__(obj) <= o.__dict__["__k_dist__"] and o != obj:
                RkNNs.append(o)
        return RkNNs

    def copy(self):
        pass
