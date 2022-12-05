from dsod.statistics import SlidingMKDE, DyCF
from dsod.plotter import LevelsetPlotter
from tests.methods_comparison import load_dataset
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score, precision_score
from methods_comparison import average_precision_score

import warnings
import pandas as pd
import matplotlib.pyplot as plt
import time


if __name__=="__main__":
    warnings.filterwarnings("error")
    cols = ["AUROC (CF)", "AP (CF)", "AUROC (MKDE)", "AP (MKDE)"]
    table = []
    """ First dataset: Two Moons """
    data = load_dataset("../res/two_moons.csv")
    labels = data[:, -1]
    data = data[:,:-1]
    # Christoffel Function
    cf = DyCF(d=6, reg='none')
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
    # AUROC & AP
    cf_scores = cf.decision_function(data)
    cf_auroc = roc_auc_score(labels, cf_scores)
    cf_ap = average_precision_score(labels, cf_scores, pos_label=-1)
    mkde_scores = mkde.decision_function(data)
    mkde_auroc = roc_auc_score(labels, mkde_scores)
    mkde_ap = average_precision_score(labels, mkde_scores, pos_label=-1)
    res = [cf_auroc, cf_ap, mkde_auroc, mkde_ap]
    table.append(['%1.9f' % v for v in res])
    # Visual comparison
    fig, axes = plt.subplots(1, 2)
    axes[0].set_title("(a) CF levelsets", fontsize='x-large')
    axes[1].set_title("(b) MKDE levelsets", fontsize='x-large')
    fig.set_figheight(6)
    fig.set_figwidth(16)
    start = time.time()
    cf_lp.plot_in_ax(data, axes[0], percentiles=[10, 20, 30, 50, 75])
    cf_lp.plot_in_ax(data, axes[0], levels=[cf.d ** (3 * cf.p / 2)], colors="green")
    print(f"CF plot: {time.time() - start}s")
    start = time.time()
    mkde_lp.plot_in_ax(data, axes[1], percentiles=[10, 20, 30, 50, 75])
    print(f"MKDE plot: {time.time() - start}s")
    plt.savefig("../tests/cf_vs_kde_1.png", bbox_inches='tight')
    # plt.savefig("../tests/cf_vs_kde_1.eps", format='eps', bbox_inches='tight')
    plt.close()

    """ Second dataset: Two Circles """
    data = load_dataset("../res/two_circles.csv")
    labels = data[:, -1]
    data = data[:,:-1]
    # Christoffel Function
    cf = DyCF(d=6, reg='none')
    start = time.time()
    cf.fit(data)
    print(f"CF fit: {time.time() - start}s")
    cf_scores = cf.score_samples(data)
    cf_lp = LevelsetPlotter(cf)
    # Multivariate KDE
    mkde = SlidingMKDE(win_size=data.shape[0])
    start = time.time()
    mkde.fit(data)
    print(f"MKDE fit: {time.time() - start}s")
    mkde_scores = mkde.score_samples(data)
    mkde_lp = LevelsetPlotter(mkde)
    # AUROC & AP
    cf_scores = cf.decision_function(data)
    cf_auroc = roc_auc_score(labels, cf_scores)
    cf_ap = average_precision_score(labels, cf_scores, pos_label=-1)
    mkde_scores = mkde.decision_function(data)
    mkde_auroc = roc_auc_score(labels, mkde_scores)
    mkde_ap = average_precision_score(labels, mkde_scores, pos_label=-1)
    res = [cf_auroc, cf_ap, mkde_auroc, mkde_ap]
    table.append(['%1.9f' % v for v in res])
    # Visual comparison
    fig, axes = plt.subplots(1, 2)
    axes[0].set_title("(a) CF levelsets", fontsize='x-large')
    axes[1].set_title("(b) MKDE levelsets", fontsize='x-large')
    fig.set_figheight(6)
    fig.set_figwidth(16)
    start = time.time()
    cf_lp.plot_in_ax(data, axes[0], percentiles=[10, 20, 30, 50, 75])
    cf_lp.plot_in_ax(data, axes[0], levels=[cf.d ** (3 * cf.p / 2)], colors="green")
    print(f"CF plot: {time.time() - start}s")
    start = time.time()
    mkde_lp.plot_in_ax(data, axes[1], percentiles=[10, 20, 30, 50, 75])
    print(f"MKDE plot: {time.time() - start}s")
    plt.savefig("../tests/cf_vs_kde_2.png", bbox_inches='tight')
    # plt.savefig("../tests/cf_vs_kde_2.eps", format='eps', bbox_inches='tight')
    plt.close()

    """ Save AUROC & AP results """
    df = pd.DataFrame(data=table, columns=cols, index=["two_moons", "two_circles"])
    df.to_csv(f"cf_vs_kde.csv")
