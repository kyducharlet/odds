import numpy as np


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
    kNN = kNN[np.where(kNN != index)]  # we remove the point indexed by index
    sorted_distances = sorted_distances[np.where(kNN != index)]
    return {index_: sorted_distances[i] for i, index_ in enumerate(kNN)}


def search_reverse_kNN(points, k_distances, index):
    x = np.array(list(points.values()))
    indices = np.array(list(points.keys()))
    distances = np.linalg.norm(x - points[index], axis=1)
    kRNN = indices[np.where(distances <= list(k_distances.values))]
    return kRNN[kRNN != index]


def compute_k_distance(points, index, kNNs):
    kthNN = list(kNNs[index].keys())[np.argmax(list(kNNs[index].values()))]
    return np.linalg.norm(points[kthNN]- points[index])


def compute_rd(points, index_p, index_o, k_distances):
    return np.max(np.linalg.norm(points[index_p] - points[index_o]), k_distances[index_o])


def compute_lrd(rds, index):
    return 1 / np.mean(list(rds[index].values()))


def compute_lof(lrds, kNNs, index):
    lrd_p = lrds[index]
    lrd_kNN_p = np.array([lrds[o] for o in list(kNNs[index].keys())])
    return np.mean(lrd_kNN_p) / lrd_p


def update_when_adding(points, index_new, x_new, kNNs, k_distances, rds, lrds, lofs, k):
    points[index_new] = x_new
    kNNs[index_new] = search_kNN(points, index_new, k)
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
        lof_to_update = [index_p]
        lof_to_update.extend(search_reverse_kNN(points, k_distances, index_p).tolist())
        """ Update lof """
        for index_q in np.unique(lof_to_update):
            lofs[index_q] = compute_lof(lrds, kNNs, index_q)
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
