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
    print("Ellapsed time: {}s".format(ellapsed_time))
    rstp = RStarTreePlotter(rst)
    rstp.plot(data_2, show=True)
