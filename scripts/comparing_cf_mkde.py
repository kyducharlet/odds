from odds.statistics import DyCF, KDE
from odds.utils import load_dataset, roc_auc_score, average_precision_score
from odds.plotter import LevelsetPlotter
import time
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    cols = ["AUROC", "AP"]
    table = []

    data = load_dataset("../res/two_disks.csv")
    labels = data[:, -1]
    data = data[:, :-1]

    # Christoffel Function
    cf = DyCF(d=6, regularization="none")
    start = time.time()
    cf.fit(data)
    print(f"CF fit: {time.time() - start}s")
    cf_lp = LevelsetPlotter(cf)

    # Multivariate KDE
    mkde = KDE(threshold=0.1, win_size=data.shape[0])
    start = time.time()
    mkde.fit(data)
    print(f"MKDE fit: {time.time() - start}s")
    mkde_lp = LevelsetPlotter(mkde)

    # AUROC & AP
    start = time.time()
    cf_scores = cf.decision_function(data)
    cf_auroc = roc_auc_score(labels, cf_scores)
    cf_ap = average_precision_score(labels, cf_scores, pos_label=-1)
    print(f"CF score & metrics: {time.time() - start}s")
    res = [cf_auroc, cf_ap]
    table.append(['%1.9f' % v for v in res])
    start = time.time()
    mkde_scores = mkde.decision_function(data)
    mkde_auroc = roc_auc_score(labels, mkde_scores)
    mkde_ap = average_precision_score(labels, mkde_scores, pos_label=-1)
    print(f"MKDE score & metrics: {time.time() - start}s")
    res = [mkde_auroc, mkde_ap]
    table.append(['%1.9f' % v for v in res])

    # Visual comparison
    fig, axes = plt.subplots(1, 2)
    axes[0].set_title("(a) CF levelsets", fontsize='x-large')
    axes[1].set_title("(b) MKDE levelsets", fontsize='x-large')
    fig.set_figheight(6)
    fig.set_figwidth(16)
    start = time.time()
    cf_lp.plot_in_ax(data, axes[0], percentiles=[10, 20, 30, 50, 75], colors=["gold", "green", "aqua", "fuchsia", "indigo"], linewidths=1.8)
    cf_lp.plot_in_ax(data, axes[0], levels=[cf.d ** (3 * cf.p / 2)], colors="red", linewidths=3.2)
    print(f"CF plot: {time.time() - start}s")
    start = time.time()
    mkde_lp.plot_in_ax(data, axes[1], percentiles=[10, 20, 30, 50, 75], colors=["gold", "green", "aqua", "fuchsia", "indigo"], linewidths=1.8)
    print(f"MKDE plot: {time.time() - start}s")
    plt.savefig("../tests/cf_vs_kde.png", bbox_inches='tight')
    plt.close()

    """ Save AUROC & AP results """
    df = pd.DataFrame(data=table, columns=cols, index=["CF", "MKDE"])
    df.to_csv(f"cf_vs_kde.csv")
