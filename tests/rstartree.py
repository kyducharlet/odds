from dsod.utils import RStarTree
from dsod.plotter import RStarTreePlotter
import numpy as np
import time

if __name__ == "__main__":
    data_1 = np.random.uniform(-1, 1, (5000, 2))
    data_2 = np.random.uniform(0, 2, (5000, 2))
    rst = RStarTree(20)
    start = time.time()
    for point in data_1:
        rst.insert_data(point.reshape(1, -1))
    for point in data_2:
        rst.remove_data(1)
        rst.insert_data(point.reshape(1, -1))
    ellapsed_time = time.time() - start
    print("Ellapsed time (building with 5k then remove/add 5k): {}s\n".format(ellapsed_time))

    rst_test_req = RStarTree(5)
    start = time.time()
    data = np.random.uniform(-1, 1, (20, 2))
    for point in data[:-1]:
        rst_test_req.insert_data(point.reshape(1, -1))
    kNNs_last = rst_test_req.search_kNN(data[-1].reshape(1, -1))
    rst_test_req.insert_data(data[-1].reshape(1, -1))
    rst_test_req.k += 1
    RkNNs_kNNs_last = {o: rst_test_req.search_RkNN(o.low) for o in kNNs_last}
    max_RkNNs = sorted([item for item in RkNNs_kNNs_last.items()], key=lambda elt: len(elt[1]))[0][0]
    kNNs_max_RkNNs = {o: rst_test_req.search_kNN(o.low) for o in RkNNs_kNNs_last[max_RkNNs]}
    ellapsed_time = time.time() - start
    for o in kNNs_last:
        print("{} is a kNN of {} with the following RkNNs:\n".format(o.low, data[-1].reshape(1, -1)))
        for RkNN in RkNNs_kNNs_last[o]:
            print(RkNN.low)
        print("\n")
    for o in RkNNs_kNNs_last[max_RkNNs]:
        print("{} is a RkNN of {} with the following kNNs:\n".format(o.low, max_RkNNs.low))
        for kNN in kNNs_max_RkNNs[o]:
            print(kNN.low)
        print("\n")
    print("Ellapsed time checking for kNNs and RkNNs of kNNs: {}s".format(ellapsed_time))

    rstp = RStarTreePlotter(rst)
    rstp.plot(data_2, show=True)
