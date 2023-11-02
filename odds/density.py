from .base import BaseDetector
from .utils import np, linalg
from .utils import RStarTree


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

    def __init__(self, k: int, threshold: float, win_size: int, min_size: int=3, max_size: int=12, p_reinsert_tol: int=4, reinsert_strategy: str="close"):
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
            rds = [max(linalg.norm(xx.reshape(1, -1) - o.low), o.__dict__["__k_dist__"]) for o in kNNs]
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
