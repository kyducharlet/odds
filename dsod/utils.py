import numpy as np
from scipy import linalg
from scipy import optimize
from hilbertcurve import hilbertcurve
import itertools
import heapq


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
    ev = np.maximum(np.std(x, axis=0), 1e-32 * np.ones(x.shape[1])) / np.power(x.shape[0], 1 / (x.shape[1] + 4))
    return np.diag(1 / ev), np.product(1 / ev)


IMPLEMENTED_BANDWIDTH_ESTIMATORS = {
    "scott": scott_rule,
}


""" R*-tree and methods for ILOF optimization """


class RStarTree:
    def __init__(self, k, min_size=3, max_size=12, p_reinsert_tol=4, reinsert_strategy="close", n_trim_iterations=3, p_hilbert_curve=4):
        self.k = k
        self.levels = [RStarTreeLevel(0)]
        self.root = RStarTreeNode(min_size, max_size, p_reinsert_tol, level=self.levels[0], leaf_level=self.levels[-1], reinsert_strategy=reinsert_strategy, tree=self)
        self.I = n_trim_iterations
        self.p_hc = p_hilbert_curve
        self.objects = []

    def insert_data(self, x):
        obj = RStarTreeObject(x, x)
        self.root.insert_data(obj)
        self.objects.append(obj)
        return obj

    def remove_oldest(self, n):
        objects = []
        for i in range(n):
            obj = self.objects.pop(0)
            obj.__remove__()
            objects.append(obj)
        return objects

    def remove_data(self, obj):
        self.objects.remove(obj)
        obj.__remove__()

    def search_kNN(self, obj):
        if type(obj) == np.ndarray:
            obj_ = RStarTreeObject(obj, obj)
        else:
            obj_ = obj
            obj.parent.children.remove(obj)
            obj.parent.__adjust_mbr__()
        res = self.k * [(np.infty, None)]
        res = self.__search_kNN__(self.root, obj_, res)
        if type(obj) != np.ndarray:
            obj.parent.children.append(obj)
            obj.parent.__adjust_mbr__()
        return [o[1] for o in res]

    def __search_kNN__(self, node, obj, res):
        if node.level == node.leaf_level:
            for c in node.children:
                dist = obj.__compute_dist__(c)
                if dist <= res[-1][0]:
                    res.append((dist, c))
                    res = sorted(res, key=lambda elt: elt[0])
                    if dist < res[-1][0]:
                        new_res = [(d, o) for (d, o) in res if d < res[-1][0]]
                        if len(new_res) >= self.k:
                            res = new_res
            return res
        else:
            branch_list = sorted([(obj.__compute_mindist__(r), obj.__compute_minmaxdist__(r), r) for r in node.children], key=lambda elt: elt[0])
            # max_possible_dist = min(np.min([elt[1] for elt in branch_list]), res[-1][0])  # do not work with k>1
            max_possible_dist = res[-1][0]
            branch_list = [elt for elt in branch_list if elt[0] <= max_possible_dist]
            for elt in branch_list:
                res = self.__search_kNN__(elt[2], obj, res)
                while len(branch_list) > 0 and branch_list[-1][0] > res[-1][0]:
                    branch_list.pop()
            return res

    """def search_RkNN(self, obj):
        if type(obj) == np.ndarray:
            obj_ = RStarTreeObject(obj, obj)
        else:
            obj_ = obj
            obj.parent.children.remove(obj)
            obj.parent.__adjust_mbr__()
        Scnd, Prfn, Nrfn = self.__search_RkNN_filter__(obj_)
        Srnn = self.__search_RkNN_refinement__(obj_, Scnd, Prfn, Nrfn)
        if type(obj) != np.ndarray:
            obj.parent.children.append(obj)
            obj.parent.__adjust_mbr__()
        return Srnn

    def __search_RkNN_filter__(self, obj):
        h = []
        heapq.heappush(h, (0, self.root))
        Scnd = []
        Srfn = []
        while len(h) != 0:
            dist, elt = heapq.heappop(h)
            if self.__search_RkNN_trim__(obj, Scnd, elt) == np.infty:
                Srfn.append(elt)
            else:
                if type(elt) != RStarTreeNode:
                    Scnd.append([elt, self.k])
                elif elt.level == elt.leaf_level:
                    for dist, point in sorted(zip([p.__compute_dist__(obj) for p in elt.children], elt.children), key=lambda p: p[0]):
                        if self.__search_RkNN_trim__(obj, Scnd, point) == np.infty:
                            Srfn.append(point)
                        else:
                            heapq.heappush(h, (dist, point))
                else:
                    for node in elt.children:
                        dist = self.__search_RkNN_trim__(obj, Scnd, node)
                        if dist == np.infty:
                            Srfn.append(node)
                        else:
                            heapq.heappush(h, (dist, node))
        Prfn = [elt for elt in Srfn if type(elt) != RStarTreeNode]
        Nrfn = [elt for elt in Srfn if type(elt) == RStarTreeNode]
        return Scnd, Prfn, Nrfn

    def __search_RkNN_trim__(self, obj, Scnd, node):
        trimmed_square = RStarTreeObject(node.low.copy(), node.high.copy())
        if len(Scnd) >= self.k:
            Scnd_ = [elt[0] for elt in Scnd]
            hc = hilbertcurve.HilbertCurve(self.p_hc, obj.high.shape[1])
            coord = np.array([cnd.low for cnd in Scnd_]).reshape(len(Scnd_), obj.low.shape[1])
            min_coord = np.min(coord, axis=0)
            div_coord = np.max(coord, axis=0) - min_coord
            div_coord[div_coord == 0] = 1
            coord = hc.max_x * (coord - min_coord) / div_coord
            dists = hc.distances_from_points(np.round(coord))
            Scnd_ = sorted(zip(Scnd_, dists), key=lambda elt: elt[1])
            Scnd_ = [elt[0] for elt in Scnd_]
            Scnd_ring = Scnd_ + Scnd_[:-1]
            # for cand_comb in itertools.combinations([o[0] for o in Scnd], self.k):
            # if True: pass
            for cand_comb in [Scnd_ring[i:self.k+i] for i in range(len(Scnd))]:
                trimmed_square = trimmed_square.__clip__(obj, cand_comb, self.I)
                if trimmed_square.contains_nan():
                    return np.infty
        return obj.__compute_mindist__(trimmed_square)

    def __search_RkNN_refinement__(self, obj, Scnd, Prfn, Nrfn):
        Srnn = []
        Scnd_ = 1 * Scnd
        for p in Scnd:
            for p_ in Scnd_:
                if p_[0] != p[0]:
                    if p[0].__compute_dist__(p_[0]) < p[0].__compute_dist__(obj):
                        p[1] -= 1
                        if p[1] == 0:
                            Scnd_.remove(p)
                            break
        Scnd = Scnd_
        while len(Scnd) != 0:
            Scnd, to_visit, Srnn = self.__search_RkNN_refinement_round__(obj, Scnd, Prfn, Nrfn, Srnn)
            if len(Scnd) == 0:
                return Srnn
            Prfn = []
            Nrfn = []
            count_nodes = {}
            for node_list in to_visit:
                for node in node_list:
                    if count_nodes.get(node.level.level) is None:
                        count_nodes[node.level.level] = {node: 1}
                    elif count_nodes[node.level.level].get(node) is None:
                        count_nodes[node.level.level][node] = 1
                    else:
                        count_nodes[node.level.level][node] += 1
            min_level = np.min(list(count_nodes.keys()))
            best_node = sorted(count_nodes[min_level].items(), key=lambda elt: -1 * elt[1])[0][0]
            if best_node.level == best_node.leaf_level:
                Prfn.extend(best_node.children)
            else:
                Nrfn.extend(best_node.children)
        return Srnn

    def __search_RkNN_refinement_round__(self, obj, Scnd, Prfn, Nrfn, Srnn):
        to_visit = []
        Scnd_ = 1 * Scnd
        for p in Scnd:
            to_visit_p = []
            valid = True
            for p_ in Prfn:
                if p[0].__compute_dist__(p_) < p[0].__compute_dist__(obj):
                    p[1] -= 1
                    if p[1] == 0:
                        Scnd_.remove(p)
                        valid = False
                        break
            if valid:
                for node in Nrfn:
                    if p[0].__compute_minmaxdist__(node) < p[0].__compute_dist__(obj):
                        p[1] -= 1
                        if p[1] == 0:
                            Scnd_.remove(p)
                            valid = False
                            break
                if valid:
                    for node in Nrfn:
                        if p[0].__compute_mindist__(node) < p[0].__compute_dist__(obj):
                            to_visit_p.append(node)
                    if len(to_visit_p) == 0:
                        Srnn.append(p[0])
                        Scnd_.remove(p)
                    else:
                        to_visit.append(to_visit_p)
        return Scnd_, to_visit, Srnn"""

    def search_RkNN(self, obj):
        if type(obj) == np.ndarray:
            obj_ = RStarTreeObject(obj, obj)
        else:
            obj_ = obj
            obj.parent.children.remove(obj)
            obj.parent.__adjust_mbr__()

        RkNN = []
        self.root.__find_reachable__(obj_, RkNN)

        if type(obj) != np.ndarray:
            obj.parent.children.append(obj)
            obj.parent.__adjust_mbr__()
        return RkNN

    def __create_new_root__(self):
        for level in self.levels:
            level.increment()
        self.levels.append(RStarTreeLevel(0))
        self.root = RStarTreeNode(self.root.min_size, self.root.max_size, self.root.p, level=self.levels[-1], leaf_level=self.root.leaf_level,
                                  reinsert_strategy=self.root.reinsert_strategy, tree=self)
        return self.root


class RStarTreeNode:
    def __init__(self, min_size, max_size, p_reinsert_tol, level, leaf_level, parent=None, reinsert_strategy="close", tree=None):
        self.parent = parent
        assert 2 <= min_size <= max_size / 2, "It is required that 2 <= min_size <= max_size / 2."
        self.min_size = min_size
        self.max_size = max_size
        self.p = p_reinsert_tol
        assert reinsert_strategy in ["close", "far"], "'reinsert_strategy' should be either 'close' or 'far'."
        self.reinsert_strategy = reinsert_strategy
        self.level = level
        self.leaf_level = leaf_level
        self.tree = tree
        self.high = None
        self.low = None
        self.children = []

        self.max_k_dist = [0, None]

    def insert_data(self, obj):
        self.__insert__(obj, self.leaf_level)

    def remove_data(self, obj):
        self.children.remove(obj)
        self.__adjust_mbr__()
        if len(self.children) < self.min_size:
            self.__underflow_treatment__()

    def __insert__(self, obj, level):
        chosen_node = self.__chose_subtree__(obj, level)
        chosen_node.children.append(obj)
        # chosen_node.__adjust_k_dist__()
        obj.parent = chosen_node
        if len(chosen_node.children) > self.max_size:
            chosen_node.__overflow_treatment__(chosen_node.level)
        chosen_node.__adjust_mbr__()

    def __chose_subtree__(self, obj, level):
        if self.level == level:
            return self
        else:
            if self.level == self.leaf_level:
                min_enlargement = np.infty
                selected_nodes = []
                for child in self.children:
                    enlargement = child.__compute_volume_enlargement__(obj)
                    if enlargement < min_enlargement:
                        min_enlargement = enlargement
                        selected_nodes.clear()
                        selected_nodes.append(child)
                    elif enlargement == min_enlargement:
                        selected_nodes.append(child)
                if len(selected_nodes) == 1:
                    return selected_nodes[0].__chose_subtree__(obj, level)
                else:
                    selected_nodes_volume = [c.__compute_volume__() for c in selected_nodes]
                    return selected_nodes[np.argmin(selected_nodes_volume)].__chose_subtree__(obj, level)
            else:
                min_enlargement = np.infty
                selected_nodes = []
                for child in self.children:
                    enlargement = child.__compute_overlap_enlargement__([c for c in self.children if c != child], obj)
                    if enlargement < min_enlargement:
                        min_enlargement = enlargement
                        selected_nodes.clear()
                        selected_nodes.append(child)
                    elif enlargement == min_enlargement:
                        selected_nodes.append(child)
                if len(selected_nodes) == 1:
                    return selected_nodes[0].__chose_subtree__(obj, level)
                else:
                    min_enlargement = np.infty
                    selected_nodes_2 = []
                    for child in selected_nodes:
                        enlargement = child.__compute_volume_enlargement__(obj)
                        if enlargement < min_enlargement:
                            min_enlargement = enlargement
                            selected_nodes_2.clear()
                            selected_nodes_2.append(child)
                        elif enlargement == min_enlargement:
                            selected_nodes_2.append(child)
                    if len(selected_nodes_2) == 1:
                        return selected_nodes_2[0].__chose_subtree__(obj, level)
                    else:
                        selected_nodes_volume = [c.__compute_volume__() for c in selected_nodes_2]
                        return selected_nodes_2[np.argmin(selected_nodes_volume)].__chose_subtree__(obj, level)

    def __overflow_treatment__(self, level):
        if level.level != 0 and not level.overflow_treated:
            level.overflow_treated = True
            self.__reinsert__()
        else:
            self.__split__()
            level.overflow_treated = False

    def __underflow_treatment__(self):
        if self.level.level != 0:
            self.parent.children.remove(self)
            self.parent.__adjust_mbr__()
            root = self.__get_root__()
            mbr_center = (root.high + root.low) / 2
            distances_to_mbr = [np.linalg.norm(mbr_center - ((r.high + r.low) / 2)) for r in self.children]
            closest_rects_indices = np.argsort(distances_to_mbr)
            to_reinsert = [self.children[i] for i in closest_rects_indices]
            for r in to_reinsert:
                root.__insert__(r, self.level)
            if len(self.parent.children) < self.min_size:
                self.parent.__underflow_treatment__()
        elif self.leaf_level.level != 0:
            grandchildren = []
            children_level = self.children[0].level
            for c in self.children:
                grandchildren.extend(c.children)
            self.children = []
            self.tree.levels.remove(children_level)
            for level in self.tree.levels[:-1]:
                level.decrement()
            mbr_center = (self.high + self.low) / 2
            distances_to_mbr = [np.linalg.norm(mbr_center - ((r.high + r.low) / 2)) for r in grandchildren]
            closest_rects_indices = np.argsort(distances_to_mbr)
            to_reinsert = [grandchildren[i] for i in closest_rects_indices]
            for r in to_reinsert:
                self.__insert__(r, self.level)

    def __reinsert__(self):
        mbr_center = (self.high + self.low) / 2
        distances_to_mbr = [np.linalg.norm(mbr_center - ((r.high + r.low) / 2)) for r in self.children]
        furthest_rects_indices = np.argsort(distances_to_mbr)[:self.p:-1]
        to_reinsert = [self.children[i] for i in furthest_rects_indices]
        for r in to_reinsert:
            self.children.remove(r)
        self.__adjust_mbr__()
        if self.reinsert_strategy == "close":
            to_reinsert.reverse()
        root = self.__get_root__()
        for r in to_reinsert:
            root.__insert__(r, self.level)
            r.parent.__adjust_k_dist__()

    def __split__(self):
        split_axis = self.__chose_split_axis__()
        split_index, first_group, second_group = self.__chose_split_index__(split_axis)
        if self.parent is None:
            new_root = self.tree.__create_new_root__()
            new_root.__insert__(self, new_root.level)
            self.tree = None
        new_node = RStarTreeNode(min_size=self.min_size, max_size=self.max_size, p_reinsert_tol=self.p, level=self.level, leaf_level=self.leaf_level, parent=self.parent, reinsert_strategy=self.reinsert_strategy)
        for r in second_group:
            self.children.remove(r)
            new_node.__insert__(r, level=new_node.level)
        self.__adjust_mbr__()
        self.parent.__insert__(new_node, level=self.parent.level)
        self.__adjust_k_dist__()
        new_node.__adjust_k_dist__()

    def __chose_split_axis__(self):
        best_axis = (-1, np.infty)
        for i in range(self.low.shape[1]):
            sorted_entries = sorted(self.children, key=lambda elt: [elt.low[0, i], elt.high[0, i]])
            sum_margin_values = 0
            for j in range(self.max_size - 2 * self.min_size + 2):
                first_group = sorted_entries[:self.min_size + j]
                fg_margin = np.product(np.max([r.high for r in first_group], axis=0) - np.min([r.low for r in first_group], axis=0)) - np.sum([r.__compute_volume__() for r in first_group])
                second_group = sorted_entries[self.min_size + j:]
                sg_margin = np.product(np.max([r.high for r in second_group], axis=0) - np.min([r.low for r in second_group], axis=0)) - np.sum([r.__compute_volume__() for r in second_group])
                sum_margin_values += fg_margin + sg_margin
            if sum_margin_values < best_axis[1]:
                best_axis = (i, sum_margin_values)
        return best_axis[0]

    def __chose_split_index__(self, split_axis):
        sorted_entries = sorted(self.children, key=lambda elt: [elt.low[0, split_axis], elt.high[0, split_axis]])
        best_index = (-1, np.infty, np.infty, None, None)
        for j in range(self.max_size - 2 * self.min_size + 2):
            first_group = sorted_entries[:self.min_size + j]
            fg_low = np.min([r.low for r in first_group], axis=0)
            fg_high = np.max([r.high for r in first_group], axis=0)
            second_group = sorted_entries[self.min_size + j:]
            sg_low = np.min([r.low for r in second_group], axis=0)
            sg_high = np.max([r.high for r in second_group], axis=0)
            overlap_volume = np.product(np.maximum(np.zeros(fg_low.shape), np.minimum(fg_high, sg_high) - np.maximum(fg_low, sg_low)))
            if overlap_volume <= best_index[1]:
                total_volume = np.product(fg_high - fg_low) + np.product(sg_high - sg_low)
                if overlap_volume < best_index[1] or total_volume < best_index[2]:
                    best_index = (j, overlap_volume, total_volume, first_group, second_group)
        return best_index[0], best_index[3], best_index[4]

    def __compute_volume__(self):
        return np.product(self.high - self.low)

    def __compute_volume_enlargement__(self, obj):
        new_low = np.minimum(self.low, obj.low)
        new_high = np.maximum(self.high, obj.high)
        return np.product(new_high - new_low) - self.__compute_volume__()

    def __compute_overlap_enlargement__(self, compared_nodes, obj):
        new_low = np.minimum(self.low, obj.low)
        new_high = np.maximum(self.high, obj.high)
        old_overlap = 0
        new_overlap = 0
        for node in compared_nodes:
            min_high = np.minimum(self.high, node.high)
            max_low = np.maximum(self.low, node.low)
            old_overlap += np.product(np.maximum(np.zeros(min_high.shape), min_high - max_low))
            min_high = np.minimum(new_high, node.high)
            max_low = np.maximum(new_low, node.low)
            new_overlap += np.product(np.maximum(np.zeros(min_high.shape), min_high - max_low))
        return new_overlap - old_overlap

    def __adjust_mbr__(self):
        old_low = self.low.copy() if self.low is not None else None
        old_high = self.high.copy() if self.high is not None else None
        self.low = np.min([r.low for r in self.children], axis=0)
        self.high = np.max([r.high for r in self.children], axis=0)
        if self.parent is not None and not ((self.low == old_low).all() and (self.high == old_high).all()):
            self.parent.__adjust_mbr__()

    def __get_root__(self):
        return self if self.parent is None else self.parent.__get_root__()

    def __update_k_dist__(self, obj):
        if obj == self.max_k_dist[1]:
            if self.max_k_dist[0] < obj.__dict__["__k_dist__"]:  # The max k-distance has increased and needs to be updated
                self.max_k_dist[0] = obj.__dict__["__k_dist__"]
            elif self.max_k_dist[0] > obj.__dict__["__k_dist__"]:  # The max k-distance has decreased and needs to be chosen again and updated
                if self.level == self.leaf_level:
                    self.max_k_dist = sorted([[o.__dict__["__k_dist__"], o] for o in self.children if o.__dict__.get("__k_dist__") is not None], key=lambda elt: -1 * elt[0])[0]
                else:
                    self.max_k_dist = sorted([c.max_k_dist for c in self.children], key=lambda elt: -1 * elt[0])[0]
            if self.parent is not None:
                self.parent.__update_k_dist__(obj)
        else:
            if self.max_k_dist[0] < obj.__dict__["__k_dist__"]:  # The obj k-distance needs to replace the current max k-distance
                self.max_k_dist = [obj.__dict__["__k_dist__"], obj]
                if self.parent is not None:
                    self.parent.__update_k_dist__(obj)

    def __adjust_k_dist__(self):
        old_mkd = self.max_k_dist
        if self.level == self.leaf_level:
            res = sorted([[o.__dict__["__k_dist__"], o] for o in self.children if o.__dict__.get("__k_dist__") is not None], key=lambda elt: -1 * elt[0])
            if len(res) != 0:
                self.max_k_dist = res[0]
        else:
            self.max_k_dist = sorted([c.max_k_dist for c in self.children], key=lambda elt: -1 * elt[0])[0]
        if self.max_k_dist != old_mkd and self.parent is not None:
            self.parent.__adjust_k_dist__()

    def __find_reachable__(self, obj, list):
        if self.level != self.leaf_level:
            for c in self.children:
                if obj.__compute_mindist__(c) <= c.max_k_dist[0]:
                    c.__find_reachable__(obj, list)
        else:
            for c in self.children:
                if obj.__compute_dist__(c) <= c.__dict__["__k_dist__"]:
                    list.append(c)

    def __lt__(self, other):
        return np.product(self.high - self.low) < np.product(other.high - other.low)


class RStarTreeObject:
    def __init__(self, low: np.ndarray, high: np.ndarray):
        self.low = low
        self.high = high
        self.parent = None

    def __compute_volume__(self):
        return np.product(self.high - self.low)

    def __compute_dist__(self, obj):
        return np.linalg.norm(obj.low - self.low)

    def __compute_mindist__(self, rect):
        p_before = np.maximum(np.zeros(self.low.shape), rect.low - self.low)
        p_after = np.maximum(np.zeros(self.high.shape), self.high - rect.high)
        return np.sum(np.square(np.maximum(p_before, p_after)))

    def __compute_minmaxdist__(self, rect):
        mbr_center = (rect.high + rect.low) / 2
        where_rm = np.where(self.low <= mbr_center)
        rm = rect.high.copy()
        rm[where_rm] = rect.low[where_rm]
        where_rM = np.where(self.low >= mbr_center)
        rM = rect.high.copy()
        rM[where_rM] = rect.low[where_rM]
        return np.min([np.square(self.low[0, i] - rm[0, i]) + np.sum([
            np.square([self.low[0, j] - rM[0, j]]) for j in range(self.low.shape[1]) if j != i
        ]) for i in range(self.low.shape[1])])

    def __clip__(self, ref_point, clipping_points, I):
        a = np.zeros((len(clipping_points), self.low.shape[1]))
        b = np.zeros((len(clipping_points), 1))
        z = a.copy()
        d = b.copy()
        cpt = 0
        while(cpt <= I):
            old_low = self.low.copy()
            old_high = self.high.copy()
            for i in range(len(clipping_points)):
                a[i] = ref_point.low - clipping_points[i].low
                b[i] = (np.square(np.linalg.norm(ref_point.low)) - np.square(np.linalg.norm(clipping_points[i].low))) / 2
                z[i] = self.high.copy()
                z[i][np.where(a[i] <= 0)] = self.low[0][np.where(a[i] <= 0)]
                d[i] = b[i] - np.dot(a[i].reshape(-1), z[i].reshape(-1))
            if (d > 0).all():
                self.low = np.nan * self.low
                self.high = np.nan * self.high
                return self
            elif self.__is_point__():
                return self
            else:
                a_red = a[np.where(d <= 0)[0]]
                d_red = d[np.where(d <= 0)[0]]
                for j in range(a_red.shape[1]):
                    if (a_red[:, j] > 0).all():
                        # self.low[0, j] = np.min([np.maximum(self.low[0, j], self.high[0, j] + d_arr[i, 0] / a_arr[i, j]) for i in range(a_arr.shape[0])])
                        possibilities = [np.maximum(self.low[0, j], self.high[0, j] + d_red[i, 0] / a_red[i, j]) for i in range(a_red.shape[0])]
                        self.low[0, j] = np.min(possibilities)
                    elif (a_red[:, j] < 0).all():
                        # self.high[0, j] = np.max([np.minimum(self.high[0, j], self.low[0, j] + d_arr[i, 0] / a_arr[i, j]) for i in range(a_arr.shape[0])])
                        possibilities = [np.minimum(self.high[0, j], self.low[0, j] + d_red[i, 0] / a_red[i, j]) for i in range(a_red.shape[0])]
                        self.high[0, j] = np.max(possibilities)
                if (self.low == old_low).all() and (self.high == old_high).all():
                    return self
                else:
                    cpt += 1
        return self

    def __is_point__(self):
        return (self.high - self.low == 0).all()

    def __remove__(self):
        self.parent.remove_data(self)

    def __lt__(self, other):
        return np.product(self.high - self.low) < other.product(self.high - self.low)

    def contains_nan(self):
        return np.isnan(self.low).any() or np.isnan(self.high).any()

    def copy(self):
        r_ = RStarTreeObject(self.low.copy(), self.high.copy())
        r_.parent = self.parent
        return r_


class RStarTreeLevel:
    def __init__(self, level):
        self.level = level
        self.overflow_treated = False

    def increment(self):
        self.level += 1

    def decrement(self):
        self.level -= 1


""" Methods for LOF computation in DILOF 
(kNN and RkNN searches are done naively since maintaining an R*-tree would be difficult, 
with a large number of removals at once and computation of kdist among a reduced dataset) """


def sigmoid(x):
    return 1 / (1 + np.exp(x))


class DILOFPoint:
    def __init__(self, values):
        self.values = values
        self.kNNs = []
        self.kdist = None
        self.rds = []
        self.lrd = None
        self.lof = None

    def compute_with_updates(self, points, k):  # used when the point is added to the model
        dists = [np.linalg.norm(self.values - p.values) for p in points]
        self.kdist = dists[np.argsort(dists)[k-1]]
        self.kNNs = list(np.array(points)[np.where(dists <= self.kdist)])
        RkNNs = [(i, p) for (i,p) in enumerate(points) if dists[i] <= p.kdist]
        lrd_updates = {p for (i, p) in RkNNs}  # We use a Python set here to avoid repetitions
        for (i, p) in RkNNs:
            lrd_updates = lrd_updates.union(p.update_dists(self, points, k))
        self.compute_rds_lrd()
        for p in lrd_updates:
            p.update_lrd()
        self.compute_lof()

    def compute_without_updates(self, points, k):  # used when scoring a point without update of the whole model
        self.compute_kNNs_kdist(points, k)
        self.compute_rds_lrd()
        self.compute_lof()

    def compute_kNNs_kdist(self, points, k):
        dists = [np.linalg.norm(self.values - p.values) for p in points if p != self]
        self.kdist = dists[np.argsort(dists)[k - 1]]
        self.kNNs = list(np.array(points)[np.where(dists <= self.kdist)])

    def compute_rds_lrd(self):
        self.rds = [max(q.kdist, np.linalg.norm(self.values - q.values)) for q in self.kNNs]
        self.lrd = 1 / np.mean(self.rds)

    def compute_lof(self):
        self.lof = np.mean([p.lrd for p in self.kNNs]) / self.lrd

    def update_dists(self, new_kNN, points, k):  # used to update kNNs, kdist and rds and to know what points need their lrd to be updated
        self.compute_kNNs_kdist(points + [new_kNN], k)  # Recompute entirely kNNs and kdist since it can have change with the removal of points
        self.rds = [max(q.kdist, np.linalg.norm(self.values - q.values)) for q in self.kNNs]
        lrd_updates = set()
        for q in self.kNNs:
            if self in q.kNNs:
                lrd_updates.add(q)
        return lrd_updates

    def update_lrd(self):  # update lrd based on already updated rds
        self.lrd = 1 / np.mean(self.rds)

    def get_local_kdist(self, points, k):  # used to compute a point kdist with a restricted dataset (or local dataset)
        dists = [np.linalg.norm(self.values - p.values) for p in points if p != self]
        return dists[np.argsort(dists)[k - 1]]


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
                c = child.center if type(child) == MTree else child
                if c == c1:
                    ch1[child] = 0
                elif c == c2 or len(ch2) < len(ch1):
                    ch2[child] = 0
                else:
                    ch1[child] = 0
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


def monomials(x, n):
    return np.power(x, n)


def tchebychev(x, n):
    if x < -1:
        return (-1)**n * np.cosh(n * np.arccosh(-x)) / np.sqrt((np.pi if n == 0 else np.pi / 2))
    elif x > 1:
        return np.cosh(n * np.arccosh(x)) / np.sqrt((np.pi if n == 0 else np.pi / 2))
    else:
        return np.cos(n * np.arccos(x)) / np.sqrt((np.pi if n == 0 else np.pi / 2))


IMPLEMENTED_POLYNOMIAL_BASIS = {
    "monomials": monomials,
    "tchebychev": tchebychev,
}


class MomentsMatrix:
    def __init__(self, d, forget_factor=None, incr_opt="wood"):
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
    def apply_combinations(x, m, basis="tchebychev"):
        if type(m) == tuple:
            assert x.shape[1] == len(m)
            result = np.array([IMPLEMENTED_POLYNOMIAL_BASIS[basis](x, n) for n in m])
            result = np.product(result, axis=1).reshape(-1, 1)
            return result
        elif type(m) == list:
            results = np.zeros((x.shape[0], len(m)))
            if x.dtype == np.dtype(object):
                results = results.astype(object)
            for i, mm in enumerate(m):
                assert x.shape[1] == len(mm)
                result = np.array([IMPLEMENTED_POLYNOMIAL_BASIS[basis](x, n) for n in mm])
                result = np.product(result, axis=1)
                results[:, i] = result
            return results
