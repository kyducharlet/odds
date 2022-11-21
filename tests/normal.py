import numpy as np
from dsod.statistics import SlidingMKDE
from dsod.density import NaiveILOF
from dsod.distance import OSMCOD, OSCOD
from dsod.plotter import LevelsetPlotter, MTreePlotter


if __name__ == "__main__":
    model = OSMCOD(k=20, R=0.5, win_size=1000, M=5)
    # model = ILOF(k=20, win_size=1000)
    data1 = np.random.multivariate_normal(np.array([0, 0]), np.array([[1, 0], [0, 1]]), 1000)
    model.fit(data1)
    data2 = np.random.multivariate_normal(np.array([0, 0]), np.array([[1, 0], [0, 1]]), 1000)
    model.update(data2)
    data3 = np.random.multivariate_normal(np.array([0, 0]), np.array([[1, 0], [0, 1]]), 1000)
    model.update(data3)
    lp = LevelsetPlotter(model)
    lp.plot(np.concatenate([data1, data2, data3]), show=True)
    mtp = MTreePlotter(model)
    mtp.plot(data3, show=True)