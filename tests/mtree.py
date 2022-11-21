from dsod.utils import MTree, MTreePoint
import numpy as np


def range_query(point, r, data):
    dists = np.linalg.norm(data - point, axis=1)
    return len(dists[dists - r <= 0])


if __name__ == "__main__":
    mt = MTree(MTreePoint, 5)
    data = np.linspace(1, 100, 1000)
    np.random.shuffle(data)
    data = data.reshape(-1, 1)
    points = [MTreePoint(p) for p in data]
    for point in points:
        mt = mt.insert_point(point)

    test_point = np.array([57])
    mtp_test_point = MTreePoint(test_point)
    R = 9.6
    dist = mt.dist(mtp_test_point)

    expected = range_query(test_point, R, data)

    res = []
    mt.__range_query__(mtp_test_point, R, dist, res)

    obtained = len(res)

    print(expected, obtained)

    points_parents = np.array([point.parent for point in points])
    count = [len(np.where(points_parents == parent)[0]) for parent in points_parents]

    """internal_nodes = []

    def fill_internal_nodes(mt, internal_nodes):
        if mt.node_type == "internal":
            internal_nodes.append(mt)
            for child in mt.children.keys():
                fill_internal_nodes(child, internal_nodes)

    fill_internal_nodes(mt, internal_nodes)

    children_count = [len(node.children) for node in internal_nodes]

    nodes_of_1_child = np.array(internal_nodes)[np.array(children_count) == 1]"""

    all_nodes_desc = []

    def fill_all_nodes_desc(mt, all_nodes_desc):
        if type(mt) == MTree:
            all_nodes_desc.append(mt)
            for child in mt.children.keys():
                fill_all_nodes_desc(child, all_nodes_desc)

    fill_all_nodes_desc(mt, all_nodes_desc)

    all_nodes_asc = []

    def fill_all_nodes_asc(point, all_nodes_asc):
        if point.parent not in all_nodes_asc and point.parent is not None:
            all_nodes_asc.append(point.parent)
            fill_all_nodes_asc(point.parent, all_nodes_asc)

    for point in points:
        fill_all_nodes_asc(point, all_nodes_asc)

    same = []
    for node in all_nodes_asc:
        same.append(node in all_nodes_desc)

    for i in range(len(points)):
        mt = mt.remove_point(points[-(i+1)])
        if len(mt.children) == 2:
            pass