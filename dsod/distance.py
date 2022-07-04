from typing import Union
import numpy as np

from .base import BaseDetector, NotFittedError
from .utils import MTree, MTreePoint, MTreeMicroCluster


class OSMCOD(BaseDetector):
    def __init__(self, k: int, R: Union[int, float], win_size: int = 2000, M: int = 5):
        self.p = None
        self.k = k
        self.R = R
        self.win_size = win_size
        self.M = M
        self.point_type = MTreePoint
        self.mcluster_type = MTreeMicroCluster
        self.mtp = MTree(self.point_type, M)
        self.mtmc = MTree(self.mcluster_type, M)
        self.points = []
        self.mcs = []
        self.__offset__ = 1 / (1 + k)

    def fit(self, x):
        self.p = x.shape[1]
        for xx in x[-self.win_size:]:
            self.__insert__(xx)
        return self

    def update(self, x):
        for xx in x[-self.win_size:]:
            # Removal
            old_point = self.points.pop(0)
            if old_point.mc is not None:
                old_point_mc = old_point.mc
                old_point_mc.remove(old_point)
                if len(old_point_mc.points) <= self.k+1:
                    self.mtmc.remove_point(old_point_mc)
                    self.mcs.remove(old_point_mc)
                    for p in old_point_mc.points:
                        self.mtp.insert_point(p)
            # Insertion
            self.__insert__(xx)
        return self

    def score_samples(self, x):
        res = np.zeros(x.shape[0])
        for i, xx in enumerate(x):
            point = self.point_type(xx)
            nb_points_without_clusters = self.mtp.score_point(point, self.R)
            clusters = self.mtmc.range_query(point, 3*self.R/2)
            points_with_clusters = np.concatenate([mc.points for mc in clusters]) if len(clusters) != 0 else np.array([])
            distances = np.array([point.dist(p) for p in points_with_clusters])
            nb_points_with_clusters = len(distances[distances <= self.R])
            res[i] = nb_points_with_clusters + nb_points_without_clusters
        return 1 / (1 + res)

    def decision_function(self, x):
        return self.__offset__ - self.score_samples(x)

    def predict(self, x):
        preds = np.zeros(len(x))
        for i, xx in enumerate(x):
            point = self.point_type(xx)
            clusters = self.mtmc.range_query(point, 3*self.R/2)
            distances = np.array([mc.dist(point) for mc in clusters])
            if len(distances[distances <= self.R / 2]) > 0:
                preds[i] = 1
            else:
                points_in_clusters = np.concatenate([mc.points for mc in clusters]) if len(clusters) != 0 else np.array([])
                if len(points_in_clusters) >= self.k:
                    preds[i] = 1
                else:
                    points = np.array(self.mtp.range_query(point, self.R))
                    if len(points) >= self.k - len(points_in_clusters):
                        preds[i] = 1
                    else:
                        preds[i] = -1
        return preds

    def predict_update(self, x):
        preds = np.zeros(len(x))
        for i, xx in enumerate(x):
            preds[i] = self.predict(xx.reshape(1, -1))
            self.update(xx.reshape(1, -1))
        return preds

    def eval_update(self, x):
        evals = np.zeros(len(x))
        for i, xx in enumerate(x):
            evals[i] = self.decision_function(xx.reshape(1, -1))
            self.update(xx.reshape(1, -1))
        return evals

    def __insert__(self, x):
        point = MTreePoint(x)
        closest_mclusters = self.mtmc.range_query(point, 3 * self.R / 2)
        if len(closest_mclusters) != 0:
            distances = np.array([mc.dist(point) for mc in closest_mclusters])
            closest_mcluster = closest_mclusters[np.argmin(distances)]
            closest_mcluster_dist = np.min(distances)
        else:
            closest_mcluster = None
            closest_mcluster_dist = np.infty
        if closest_mcluster_dist <= self.R / 2:
            point.set_mcluster(closest_mcluster)
        else:
            closest_points = np.array(self.mtp.range_query(point, self.R))
            points_distances = np.array([mc.dist(point) for mc in closest_points])
            if len(points_distances[points_distances <= self.R / 2]) > self.k:
                new_mc = self.mcluster_type(point, closest_points[points_distances <= self.R / 2], self.R / 2)
                point.set_mcluster(new_mc)
                self.mtmc = self.mtmc.insert_point(new_mc)
                self.mcs.append(new_mc)
            else:
                self.mtp = self.mtp.insert_point(point)
        self.points.append(point)

    def copy(self):
        pass


class OSCOD(BaseDetector):
    def __init__(self, k: int, R: Union[int, float], win_size: int = 2000, M: int = 5):
        self.p = None
        self.k = k
        self.R = R
        self.win_size = win_size
        self.M = M
        self.point_type = MTreePoint
        self.mt = MTree(self.point_type, M)
        self.points = []
        self.__offset__ = 1 / (1 + k)

    def fit(self, x):
        self.p = x.shape[1]
        for xx in x[-self.win_size:]:
            self.__insert__(xx)
        return self

    def update(self, x):
        for xx in x[-self.win_size:]:
            # Removal
            old_point = self.points.pop(0)
            self.mt = self.mt.remove_point(old_point)
            # Insertion
            self.__insert__(xx)
        return self

    def score_samples(self, x):
        res = np.zeros(x.shape[0])
        for i, xx in enumerate(x):
            res[i] = self.mt.score_point(self.point_type(xx), self.R)
        return 1 / (1 + res)

    def decision_function(self, x):
        return self.__offset__ - self.score_samples(x)

    def predict(self, x):
        return np.where(self.decision_function(x) < 0, -1, 1)

    def predict_update(self, x):
        preds = np.zeros(len(x))
        for i, xx in enumerate(x):
            preds[i] = self.predict(xx.reshape(1, -1))
            self.update(xx.reshape(1, -1))
        return preds

    def eval_update(self, x):
        evals = np.zeros(len(x))
        for i, xx in enumerate(x):
            evals[i] = self.decision_function(xx.reshape(1, -1))
            self.update(xx.reshape(1, -1))
        return evals

    def __insert__(self, x):
        point = self.point_type(x)
        self.mt = self.mt.insert_point(point)
        self.points.append(point)

    def copy(self):
        pass
