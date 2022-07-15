from dsod.utils import RStarTree
from dsod.plotter import RStarTreePlotter
import numpy as np


if __name__ == "__main__":
    data = np.random.uniform(-1, 1, (500, 2))
    rst = RStarTree(5)
    rst.insert_data(data)
    rstp = RStarTreePlotter(rst)
    rstp.plot(data, show=True)
