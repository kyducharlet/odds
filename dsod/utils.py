import numpy as np
from scipy import linalg
import itertools


""" Kernel functions for KDE """


def gaussian_kernel(x):
    if x.shape[0] == 1:
        return np.exp(-1 * np.dot(x, x.T)[0, 0] / 2) / np.power(np.sqrt(2 * np.pi), x.shape[1])
    else:
        return np.array([gaussian_kernel(point.reshape(1, -1)) for point in x])


IMPLEMENTED_KERNEL_FUNCTIONS = {
    "gaussian": gaussian_kernel,
}


""" Bandwidth estimators for KDE """


def scott_rule(x):
    ev = np.std(x, axis=0) / np.power(x.shape[0], 1 / (x.shape[1] + 4))
    return np.diag(1 / ev), np.product(1 / ev)


IMPLEMENTED_BANDWIDTH_ESTIMATORS = {
    "scott": scott_rule,
}


""" Methods for LOF computation """


def search_kNN(points, index, k):
    x = np.array(list(points.values()))
    distances = np.linalg.norm(x - points[index], axis=1)
    arg_sorted_distances = np.argsort(distances)
    sorted_distances = distances[arg_sorted_distances]
    kthNN_distance = sorted_distances[k]  # we take the (k+1)th since the point indexed by index is in the dataset
    kNN = np.array(list(points.keys()))[arg_sorted_distances[np.where(sorted_distances <= kthNN_distance)]]
    sorted_distances = sorted_distances[np.where(sorted_distances <= kthNN_distance)]
    sorted_distances = sorted_distances[np.where(kNN != index)]  # we remove the point indexed by index
    kNN = kNN[np.where(kNN != index)]
    return {index_: sorted_distances[i] for i, index_ in enumerate(kNN)}


def search_reverse_kNN(points, k_distances, index):
    x = np.array(list(points.values()))
    indices = np.array(list(points.keys()))
    distances = np.linalg.norm(x - points[index], axis=1)
    kRNN = indices[np.where(distances <= list(k_distances.values()))]
    return kRNN[kRNN != index]


def compute_k_distance(points, index, kNNs):
    kthNN = list(kNNs[index].keys())[np.argmax(list(kNNs[index].values()))]
    return np.linalg.norm(points[kthNN]- points[index])


def compute_rd(points, index_p, index_o, k_distances):
    return np.max([np.linalg.norm(points[index_p] - points[index_o]), k_distances[index_o]])


def compute_lrd(rds, index):
    return 1 / np.mean(list(rds[index].values()))


def compute_lof(lrds, kNNs, index):
    lrd_p = lrds[index]
    lrd_kNN_p = np.array([lrds[o] for o in list(kNNs[index].keys())])
    return np.mean(lrd_kNN_p) / lrd_p


def update_when_adding(points, index_new, x_new, kNNs, k_distances, rds, lrds, lofs, k):
    points[index_new] = x_new
    kNNs[index_new] = search_kNN(points, index_new, k)
    k_distances[index_new] = compute_k_distance(points, index_new, kNNs)
    rds[index_new] = {o: compute_rd(points, index_new, o, k_distances) for o in list(kNNs[index_new].keys())}
    kdist_to_update = search_reverse_kNN(points, k_distances, index_new)
    for index_p in kdist_to_update:
        lof_to_update = [index_p]
        new_distance = np.linalg.norm(points[index_p] - points[index_new])
        old_distances = np.array(list(kNNs[index_p].values()))
        """ Remove points that are not kNNs anymore and add the new point as a kNN to its kRNNs """
        if new_distance != np.max(old_distances):
            to_remove = np.array(list(kNNs[index_p].keys()))[np.where(old_distances == np.max(old_distances))]
            if len(old_distances) - len(to_remove) + 1 >= k:
                for id in to_remove:
                    del kNNs[index_p][id]
                    del rds[index_p][id]
        kNNs[index_p][index_new] = new_distance
        """ Update k_distance """
        k_distances[index_p] = np.max(np.array(list(kNNs[index_p].values())))
        rds[index_p][index_new] = compute_rd(points, index_p, index_new, k_distances)
        """ Update rd and lrd """
        for index_q in list(kNNs[index_p].keys()):
            if rds[index_q].get(index_p) is not None:
                rds[index_q][index_p] = compute_rd(points, index_q, index_p, k_distances)
                lrds[index_q] = compute_lrd(rds, index_q)
                lof_to_update.append(index_q)
                lof_to_update.extend(search_reverse_kNN(points, k_distances, index_q).tolist())
        lrds[index_p] = compute_lrd(rds, index_p)
        lrds[index_new] = compute_lrd(rds, index_new)
        lof_to_update.extend(search_reverse_kNN(points, k_distances, index_p).tolist())
        """ Update lof """
        for index_q in np.unique(lof_to_update):
            lofs[index_q] = compute_lof(lrds, kNNs, index_q)
    if lrds.get(index_new) is None:
        lrds[index_new] = compute_lrd(rds, index_new)
    lofs[index_new] = compute_lof(lrds, kNNs, index_new)
    return kNNs, k_distances, rds, lrds, lofs


def update_when_removing(points, index_dead, kNNs, k_distances, rds, lrds, lofs, k):
    kdist_to_update = search_reverse_kNN(points, k_distances, index_dead)  # Update k_distance of the kNN of the deleted point
    del points[index_dead]
    del k_distances[index_dead]
    del kNNs[index_dead]
    del rds[index_dead]
    del lrds[index_dead]
    del lofs[index_dead]
    lrd_to_update = kdist_to_update.tolist()
    for index_p in kdist_to_update:
        if len(list(kNNs[index_p].keys())) == k:  # the k_distance changes only if the point has exactly k kNN
            prev_kNN = set(kNNs[index_p].keys())
            kNNs[index_p] = search_kNN(points, index_p, k)
            kNN_to_add = list(set(kNNs[index_p].keys()).difference(prev_kNN))
            k_distances[index_p] = np.linalg.norm(points[index_p] - points[kNN_to_add[0]])
            del rds[index_p][index_dead]  # we only keep rd of p to its kNN
            for index_q in kNN_to_add:
                rds[index_p][index_q] = compute_rd(points, index_p, index_q, k_distances)  # we have to add rd of p to its new kNN
            rd_to_update = np.array(list(kNNs[index_p].keys()))[  # from all kNN...
                    np.array(list(kNNs[index_p].values())) < k_distances[index_p]  # ...we select only the (k-1)NN
                ]
            for index_q in rd_to_update :  # Update rd of (k-1)NN of points we changed the k_distance
                if index_p in list(kNNs[index_q].keys()):
                    rds[index_q][index_p] = k_distances[index_p]
                    lrd_to_update.append(index_q)
        else:
            del kNNs[index_p][index_dead]
            del rds[index_p][index_dead]
    lof_to_update = 1 * lrd_to_update
    for index_p in lrd_to_update:
        lrds[index_p] = compute_lrd(rds, index_p)
        lof_to_update.extend(search_reverse_kNN(points, k_distances, index_p).tolist())
    for index_p in lof_to_update:
        lofs[index_p] = compute_lof(lrds, kNNs, index_p)
    return kNNs, k_distances, rds, lrds, lofs


""" M-tree used by MCOD for range queries """


class MTree:
    def __init__(self, point_type, M, node_type="leaf", parent=None, center=None, radius=None, children=None):
        self.node_type = node_type  # One of "internal" or "leaf"
        self.point_type = point_type
        self.M = M  # max size for a node
        self.parent = parent
        self.center = center
        self.radius = radius
        self.children = children
        if children is not None:
            for child in children.keys():
                child.parent = self

    def insert_point(self, point):
        assert type(point) == self.point_type
        if self.node_type == "leaf":
            if self.center is None:
                self.center = point
                self.radius = 0
                self.children = dict()
            dist = self.dist(point)
            self.children[point] = dist
            point.parent = self
            self.radius = max(self.radius, dist)
            if len(self.children) > self.M:
                return self.split()
            else:
                return self.__get_root__()
        else:
            best_child = self.__search_best_child__(point)
            return best_child.insert_point(point)

    def remove_point(self, point):
        assert type(point) == self.point_type
        assert point.parent is not None
        del point.parent.children[point]  # Remove point address in the leaf node
        current_node = point
        if len(current_node.parent.children) == 0:  # If the leaf has no children :
            current_node = current_node.parent
            parent = current_node.parent
            if parent is not None:  # If it is not the root of the tree :
                del parent.children[current_node]  # A leaf node that has no children should be deleted
                if len(parent.children) == 1:  # If the parent node has only one child left, it should absorb this child
                    child = list(parent.children.keys())[0]
                    parent.node_type = child.node_type
                    parent.center = child.center
                    parent.radius = child.radius
                    parent.children = child.children
                    for n_child in parent.children:
                        n_child.parent = parent
                current_node = parent
        while current_node.parent is not None:  # Checking all the radiuses
            parent = current_node.parent
            parent.__recompute_radius__()
            current_node = parent
        return self

    def split(self):
        center_1, center_2 = self.__promote_centers__()
        children_1, radius_1, children_2, radius_2 = self.__split_children__(center_1, center_2)
        if len(children_1) >= len(children_2):
            center = center_1
            children = children_1
            radius = radius_1
            new_center = center_2
            new_children = children_2
            new_radius = radius_2
        else:
            center = center_2
            children = children_2
            radius = radius_2
            new_center = center_1
            new_children = children_1
            new_radius = radius_1
        if self.node_type == "leaf" or len(new_children) > 1:
            new_Mtree = MTree(node_type=self.node_type, point_type=self.point_type, M=self.M, parent=self.parent, center=new_center, children=new_children, radius=new_radius)
        else:  # An internal node should not have one single child, or it absorbs it
            new_Mtree = list(new_children.keys())[0]
            new_Mtree.parent = self.parent
        self.center = center
        self.radius = radius
        self.children = children
        for child in self.children:
            child.parent = self
        if self.parent is not None:
            self.parent.children[self] = self.parent.dist(self.center)
            dist = self.parent.center.dist(center_2)
            self.parent.children[new_Mtree] = dist
            self.parent.radius = max(self.parent.radius, dist + radius_2)
            if len(self.parent.children.keys()) > self.M:
                return self.parent.split()
            else:
                return self.__get_root__()
        else:
            dist = center.dist(new_center)
            r = max(radius, dist + new_radius)
            new_root = MTree(node_type="internal", point_type=self.point_type, M=self.M, parent=None, center=center, children={self: 0, new_Mtree: dist}, radius=r)
            return new_root

    def dist(self, elt):
        return self.center.dist(elt)

    def score_point(self, point, R):
        return len(self.range_query(point, R))

    def range_query(self, point, R):
        res = []
        if self.center is not None:
            self.__range_query__(point, R, self.dist(point), res)
        return res

    def __range_query__(self, point, R, dist, res):  # res is a list where the results will be contained
        if self.node_type == "leaf":
            for child in self.children.keys():
                if abs(dist - self.children[child]) <= R:
                    n_dist = child.dist(point)
                    if n_dist <= R:
                        res.append(child)
        else:
            for child in self.children.keys():
                if abs(dist - self.children[child]) <= R + child.radius:
                    n_dist = child.center.dist(point)
                    if n_dist <= R + child.radius:
                        child.__range_query__(point, R, n_dist, res)

    def __search_best_child__(self, point):
        dists = np.array([child.dist(point) for child in self.children.keys()])
        rs = np.array([child.radius for child in self.children.keys()])
        dists_diff = dists - rs
        matching_children = np.where(dists_diff <= 0)[0]
        if len(matching_children) >= 1:
            indexes = np.linspace(0, len(dists) - 1, len(dists)).astype(int)
            indexes = indexes[matching_children]
            return list(self.children.keys())[indexes[np.argmin(dists[matching_children])]]
        else:
            chosen_child_index = np.argmin(dists_diff)
            chosen_child = list(self.children.keys())[chosen_child_index]
            chosen_child.radius += dists_diff[chosen_child_index]
            return chosen_child

    def __promote_centers__(self):
        c1 = self.center
        children = list(self.children.keys())
        if self.node_type != "leaf":
            children_ = [c for c in children if c.center != c1]
            c2 = children_[np.argmax([self.children[c] for c in children_])].center
            if len(children_) == len(children):  # c1 is not a child anymore
                children_ = [c for c in children if c.center != c2]
                c1 = children_[np.argmax([c2.dist(c) for c in children_])].center
        else:
            children_ = [c for c in children if c != c1]
            c2 = children_[np.argmax([self.children[c] for c in children_])]
            if len(children_) == len(children):  # c1 is not a child anymore
                children_ = [c for c in children if c != c2]
                c1 = children_[np.argmax([c2.dist(c) for c in children_])]
        return c1, c2

    def __split_children__(self, c1, c2):
        ch1 = dict()
        r1 = 0
        ch2 = dict()
        r2 = 0
        if c1.dist(c2) != 0:
            for child in self.children:
                dist1 = c1.dist(child)
                dist2 = c2.dist(child)
                if dist1 <= dist2:
                    ch1[child] = dist1
                    r1 = max(r1, dist1) if type(child) == self.point_type else max(r1, dist1 + child.radius)
                else:
                    ch2[child] = dist2
                    r2 = max(r2, dist2) if type(child) == self.point_type else max(r2, dist2 + child.radius)
        else:  # all points are the same
            for child in self.children:
                c = child.children if type(child) == MTree else child
                if c == c1:
                    ch1[c] = 0
                elif c == c2 or len(ch2) < len(ch1):
                    ch2[c] = 0
                else:
                    ch1[c] = 0
        return ch1, r1, ch2, r2

    def __get_root__(self):
        root = self
        while root.parent is not None:
            root = root.parent
        return root

    def __recompute_radius__(self):
        if self.node_type == "leaf":
            self.radius = np.max(list(self.children.values()))
        else:
            distances = np.array(list(self.children.values()))
            radiuses = np.array([c.radius for c in self.children.keys()])
            self.radius = np.max(distances + radiuses)
        return self

    """def __check_count__(self):
        if self.node_type != "leaf":
            if len(self.children) == 0:
                raise ValueError
            for child in self.children:
                child.__check_count__()
        elif self.parent is not None:
            if len(self.children) == 0:
                raise ValueError"""


class MTreePoint:
    def __init__(self, values):
        self.values = values
        self.parent = None
        self.mc = None

    def dist(self, elt):
        return np.linalg.norm(self.values - elt.values) if type(elt) == MTreePoint else np.linalg.norm(self.values - elt.center.values)

    def set_mcluster(self, mcluster):
        self.mc = mcluster
        mcluster.points.append(self)


class MTreeMicroCluster:
    def __init__(self, center, points, R):
        self.values = center.values
        self.points = [point for point in points]
        self.R = R

    def dist(self, elt):
        return np.linalg.norm(self.values - elt.values) if type(elt) != MTree else np.linalg.norm(self.values - elt.center.values)

    def remove(self, elt):
        self.points.remove(elt)
        return self


""" Methods and classes used by SimpleChristoffel and DyCG """


class MomentsMatrix:
    def __init__(self, d, forget_factor=None, incr_opt="inv"):
        self.d = d
        self.forget_factor = forget_factor
        self.incr_opt = incr_opt
        if forget_factor is None:
            self.fit = self.__fit__
            self.update = self.__update__
        else:
            self.fit = self.__fit_forget__
            self.update = self.__update_forget__

    def __fit__(self, x):
        monomials = Monomials.generate_combinations(self.d, x.shape[1])
        self.__dict__["__monomials"] = monomials
        len_m = len(monomials)
        moments_matrix = np.zeros((len_m, len_m), dtype=x.dtype)
        for xx in x:
            v = np.power(xx, monomials)
            v = np.product(v, axis=1).reshape((-1, 1))
            moments_matrix += np.dot(v, v.T)
        moments_matrix /= len(x)
        self.__dict__["__moments_matrix"] = moments_matrix
        self.__dict__["__inverse_moments_matrix"] = linalg.inv(moments_matrix)
        return self

    def __fit_forget__(self, x):
        monomials = Monomials.generate_combinations(self.d, x.shape[1])
        self.__dict__["__monomials"] = monomials
        len_m = len(monomials)
        moments_matrix = np.zeros((len_m, len_m), dtype=x.dtype)
        sum = 0
        for xx in x:
            v = np.power(xx, monomials)
            v = np.product(v, axis=1).reshape((-1, 1))
            moments_matrix = self.forget_factor * moments_matrix + np.dot(v, v.T)
            sum = self.forget_factor * sum + 1
        moments_matrix /= sum
        self.__dict__["__moments_matrix"] = moments_matrix
        self.__dict__["__inverse_moments_matrix"] = linalg.inv(moments_matrix)
        return self

    def score_samples(self, x):
        res = []
        for xx in x:
            v = np.power(xx, self.__dict__["__monomials"])
            v = np.product(v, axis=1).reshape((-1, 1))
            res.append(np.dot(np.dot(v.T, self.__dict__["__inverse_moments_matrix"]), v))
        return np.array(res).reshape(-1)

    def __update__(self, x, n):  # Deux options : 1) incrémentation de la matrice et inversion, 2) incrémentation de l'inverse de la matrice
        if self.incr_opt != "inv":
            # OPT #1:
            moments_matrix = n * self.__dict__["__moments_matrix"]
            for xx in x:
                v = np.power(xx, self.__dict__["__monomials"])
                v = np.product(v, axis=1).reshape((-1, 1))
                moments_matrix += np.dot(v, v.T)
            moments_matrix /= (n + x.shape[0])
            self.__dict__["__moments_matrix"] = moments_matrix
            self.__dict__["__inverse_moments_matrix"] = linalg.inv(moments_matrix)
        else:
            # OPT #2:
            inv_moments_matrix = self.__dict__["__inverse_moments_matrix"] / n
            for xx in x:
                v = np.power(xx, self.__dict__["__monomials"])
                v = np.product(v, axis=1).reshape((-1, 1))
                a = np.matmul(np.matmul(inv_moments_matrix, np.dot(v, v.T)), inv_moments_matrix)
                b = np.dot(np.dot(v.T, inv_moments_matrix), v)
                inv_moments_matrix -= a / (1 + b)
            self.__dict__["__inverse_moments_matrix"] = (n + x.shape[0]) * inv_moments_matrix
        return self

    def __update_forget__(self, x, n):  # Comme pour __update__
        if self.incr_opt != "inv":
            # OPT #1:
            sum = (1 - np.power(self.forget_factor, n)) / (1 - self.forget_factor)
            moments_matrix = sum * self.__dict__["__moments_matrix"]
            for xx in x:
                v = np.power(xx, self.__dict__["__monomials"])
                v = np.product(v, axis=1).reshape((-1, 1))
                moments_matrix = self.forget_factor * moments_matrix + np.dot(v, v.T)
                sum = self.forget_factor * sum + 1
            moments_matrix /= sum
            self.__dict__["__moments_matrix"] = moments_matrix
            self.__dict__["__inverse_moments_matrix"] = linalg.inv(moments_matrix)
        else:
            # OPT #2:
            sum = (1 - np.power(self.forget_factor, n)) / (1 - self.forget_factor)
            inv_moments_matrix = self.__dict__["__inverse_moments_matrix"] / sum
            for xx in x:
                v = np.power(xx, self.__dict__["__monomials"])
                v = np.product(v, axis=1).reshape((-1, 1))
                inv_moments_matrix = inv_moments_matrix / self.forget_factor
                a = np.matmul(np.matmul(inv_moments_matrix, np.dot(v, v.T)), inv_moments_matrix)
                b = np.dot(np.dot(v.T, inv_moments_matrix), v)
                inv_moments_matrix = inv_moments_matrix - (a / (1 + b))
                sum = 1 + self.forget_factor * sum
            self.__dict__["__inverse_moments_matrix"] = sum * inv_moments_matrix
        return self

    def learned(self):
        return self.__dict__.get("__inverse_moments_matrix") is not None

    def copy(self):
        mm_bis = MomentsMatrix(d=self.d, forget_factor=self.forget_factor, incr_opt=self.incr_opt)
        mm_bis.__dict__["__monomials"] = self.__dict__["__monomials"]
        mm_bis.__dict__["__moments_matrix"] = self.__dict__["__moments_matrix"]
        mm_bis.__dict__["__inverse_moments_matrix"] = self.__dict__["__inverse_moments_matrix"]
        return mm_bis


class Monomials:
    @staticmethod
    def generate_combinations(n, p):
        it = itertools.product(range(n + 1), repeat=p)
        mono = [i for i in it if np.sum(list(i)) <= n]
        return sorted(mono, key=lambda e: (np.sum(list(e)), list(-1 * np.array(list(e)))))

    @staticmethod
    def apply_combinations(x, m):
        if type(m) == tuple:
            assert x.shape[1] == len(m)
            result = np.power(x, list(m))
            result = np.product(result, axis=1).reshape(-1, 1)
            return result
        elif type(m) == list:
            results = np.zeros((x.shape[0], len(m)))
            if x.dtype == np.dtype(object):
                results = results.astype(object)
            for i, mm in enumerate(m):
                assert x.shape[1] == len(mm)
                result = np.power(x, list(mm))
                result = np.product(result, axis=1)
                results[:, i] = result
            return results
