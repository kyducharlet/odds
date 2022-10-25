from dsod.statistics import SlidingMKDE, DyCF
from dsod.plotter import LevelsetPlotter
from tests.methods_comparison import load_dataset

import warnings
import matplotlib.pyplot as plt
import time


if __name__=="__main__":
    warnings.filterwarnings("error")
    """ First dataset: Two Moons """
    data = load_dataset("../res/two_moons.csv")[:,:-1]
    # Christoffel Function
    cf = DyCF(d=6)
    start = time.time()
    cf.fit(data)
    print(f"CF fit: {time.time() - start}s")
    cf_lp = LevelsetPlotter(cf)
    # Multivariate KDE
    mkde = SlidingMKDE(win_size=data.shape[0])
    start = time.time()
    mkde.fit(data)
    print(f"MKDE fit: {time.time() - start}s")
    mkde_lp = LevelsetPlotter(mkde)
    # Visual comparison
    fig, axes = plt.subplots(1, 2)
    axes[0].set_title("(a) CF levelsets", fontsize='x-large')
    axes[1].set_title("(b) MKDE levelsets", fontsize='x-large')
    fig.set_figheight(6)
    fig.set_figwidth(16)
    start = time.time()
    cf_lp.plot_in_ax(data, axes[0], percentiles=[10, 20, 30, 50, 75])
    print(f"CF plot: {time.time() - start}s")
    start = time.time()
    mkde_lp.plot_in_ax(data, axes[1], percentiles=[10, 20, 30, 50, 75])
    print(f"MKDE plot: {time.time() - start}s")
    plt.savefig("../tests/cf_vs_kde_1.png", bbox_inches='tight')
    # plt.savefig("../tests/cf_vs_kde_1.eps", format='eps', bbox_inches='tight')
    plt.close()

    """ Second dataset: Two Circles """
    data = load_dataset("../res/two_circles.csv")[:, :-1]
    # Christoffel Function
    cf = DyCF(d=6)
    start = time.time()
    cf.fit(data)
    print(f"CF fit: {time.time() - start}s")
    cf_lp = LevelsetPlotter(cf)
    # Multivariate KDE
    mkde = SlidingMKDE(win_size=data.shape[0])
    start = time.time()
    mkde.fit(data)
    print(f"MKDE fit: {time.time() - start}s")
    mkde_lp = LevelsetPlotter(mkde)
    # Visual comparison
    fig, axes = plt.subplots(1, 2)
    axes[0].set_title("(a) CF levelsets", fontsize='x-large')
    axes[1].set_title("(b) MKDE levelsets", fontsize='x-large')
    fig.set_figheight(6)
    fig.set_figwidth(16)
    start = time.time()
    cf_lp.plot_in_ax(data, axes[0], percentiles=[10, 20, 30, 50, 75])
    print(f"CF plot: {time.time() - start}s")
    start = time.time()
    mkde_lp.plot_in_ax(data, axes[1], percentiles=[10, 20, 30, 50, 75])
    print(f"MKDE plot: {time.time() - start}s")
    plt.savefig("../tests/cf_vs_kde_2.png", bbox_inches='tight')
    # plt.savefig("../tests/cf_vs_kde_2.eps", format='eps', bbox_inches='tight')
    plt.close()
