from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.exceptions import UndefinedMetricWarning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import pickle
from tqdm import tqdm

from dsod.distance import OSCOD, OSMCOD
from dsod.density import ILOF
from dsod.statistics import SlidingMKDE, SimpleChristoffel, DyCG
from dsod.plotter import LevelsetPlotter


METHODS = [
        {
            "name": "OSCOD k=10",
            "method": OSCOD,
            "params": {
                "k": 10,
                "R": 0.5,
                "win_size": 2000,
                "M": 5,
            },
            "linestyle": (0, (1, 0))
        },
        {
            "name": "OSCOD k=20",
            "method": OSCOD,
            "params": {
                "k": 20,
                "R": 0.5,
                "win_size": 2000,
                "M": 5,
            },
            "linestyle": (0, (1, 1))
        }, {
            "name": "OSMCOD k=10",
            "method": OSMCOD,
            "params": {
                "k": 10,
                "R": 0.5,
                "win_size": 2000,
                "M": 5,
            },
            "linestyle": (0, (2, 1))
        }, {
            "name": "OSMCOD k=20",
            "method": OSMCOD,
            "params": {
                "k": 20,
                "R": 0.5,
                "win_size": 2000,
                "M": 5,
            },
            "linestyle": (0, (1, 2))
        }, """{
            "name": "ILOF k=10",
            "method": ILOF,
            "params": {
                "k": 10,
                "win_size": 2000,
            },
            "linestyle": (0, (1, 3))
        }, {
            "name": "ILOF k=20",
            "method": ILOF,
            "params": {
                "k": 20,
                "win_size": 2000,
            },
            "linestyle": (0, (3, 1))
        }""", {
            "name": "SlidingMKDE W=1000",
            "method": SlidingMKDE,
            "params": {
                "win_size": 1000,
            },
            "linestyle": (0, (.5, 1))
        }, {
            "name": "SlidingMKDE W=2000",
            "method": SlidingMKDE,
            "params": {
                "win_size": 2000,
            },
            "linestyle": (0, (1, .5))
        }, {
            "name": "Simple d=2",
            "method": SimpleChristoffel,
            "params": {
                "d": 2,
            },
            "linestyle": (0, (3, 2))
        }, {
            "name": "SimpleChristoffel d=6",
            "method": SimpleChristoffel,
            "params": {
                "d": 6,
            },
            "linestyle": (0, (2, 3))
        }, {
            "name": "DyCG mean_diff reg 1",
            "method": DyCG,
            "params": {"decision": "mean_growth_rate", "reg": "1"},
            "linestyle": (0, (2, 2))
        }, {
            "name": "DyCG sign_reg reg 1",
            "method": DyCG,
            "params": {"decision": "sign_poly_2_reg", "reg": "1"},
            "linestyle": (0, (2, 5))
        }, {
            "name": "DyCG mean_diff reg 2",
            "method": DyCG,
            "params": {"decision": "mean_growth_rate", "reg": "2"},
            "linestyle": (0, (2, 2))
        }, {
            "name": "DyCG sign_reg reg 2",
            "method": DyCG,
            "params": {"decision": "sign_poly_2_reg", "reg": "2"},
            "linestyle": (0, (2, 5))
        }, {
            "name": "DyCG mean_diff reg 3",
            "method": DyCG,
            "params": {"decision": "mean_growth_rate", "reg": "3"},
            "linestyle": (0, (2, 2))
        }, {
            "name": "DyCG sign_reg reg 3",
            "method": DyCG,
            "params": {"decision": "sign_poly_2_reg", "reg": "3"},
            "linestyle": (0, (2, 5))
        },
    ]


def load_dataset(filename):
    data = pd.read_csv(filename, index_col=0).values
    min = np.min(data, axis=0)[:-1]
    max = np.max(data, axis=0)[:-1]
    data[:, :-1] = 2 * (((data[:, :-1] - min) / (max - min)) - .5)
    return data


def split_data(data, split_pos):
    train = data[:split_pos]
    test = data[split_pos:]
    return train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]


def plot(methods, x_train, x_test, y_test, savename):
    fig, ax = plt.subplots(4, figsize=(32, 18))
    y_decisions = dict()
    y_preds = dict()
    for method in methods:
        if type(method) == dict:
            filename = savename + "__" + method["name"] + ".pickle"
            try:
                with open(filename, "rb") as f:
                    results = pickle.load(f)
                y_decisions[method["name"]] = results["y_decision"]
                y_preds[method["name"]] = results["y_pred"]
            except FileNotFoundError:
                model = method["method"](**method["params"])
                model.fit(x_train)
                y_decision = np.zeros(len(y_test))
                y_pred = np.zeros(len(y_test))
                for i in tqdm(range(len(x_test)), desc=method["name"]):
                    y_decision[i] = model.eval_update(x_test[i].reshape(1, -1))
                    y_pred[i] = -1 if y_decision[i] < 0 else 1
                results = {
                    "y_decision": y_decision,
                    "y_pred": y_pred,
                }
                y_decisions[method["name"]] = y_decision
                y_preds[method["name"]] = y_pred
                with open(filename, "wb") as f:
                    pickle.dump(results, f)
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        for method in methods:
            if type(method) == dict:
                filename = "metric_" + savename + "__" + method["name"] + ".pickle"
                try:
                    with open(filename, "rb") as f:
                        results = pickle.load(f)
                    f1_out = results["f1_out"]
                    f1_in = results["f1_in"]
                    roc_auc = results["roc_auc"]
                    balanced_accuracy = results["balanced_accuracy"]
                except FileNotFoundError:
                    f1_out = np.zeros(len(y_test))
                    f1_in = np.zeros(len(y_test))
                    roc_auc = np.zeros(len(y_test))
                    balanced_accuracy = np.zeros(len(y_test))
                    for i in tqdm(range(len(x_test)), desc=method["name"]):
                        try:
                            f1_out[i] = f1_score(y_test[:i+1], y_preds[method["name"]][:i+1], pos_label=-1)
                        except (UndefinedMetricWarning, RuntimeWarning):
                            f1_out[i] = np.NaN
                        try:
                            f1_in[i] = f1_score(y_test[:i+1], y_preds[method["name"]][:i+1], pos_label=1)
                        except (UndefinedMetricWarning, RuntimeWarning):
                            f1_in[i] = np.NaN
                        try:
                            roc_auc[i] = roc_auc_score(y_test[:i+1], y_decisions[method["name"]][:i+1])  # score and decision (-1*score) give the same results
                        except (ValueError, RuntimeWarning):
                            roc_auc[i] = np.NaN
                        try:
                            balanced_accuracy[i] = balanced_accuracy_score(y_test[:i+1], y_preds[method["name"]][:i+1])
                        except (UserWarning, RuntimeWarning):
                            balanced_accuracy[i] = np.NaN
                    results = {
                        "f1_out": f1_out,
                        "f1_in": f1_in,
                        "roc_auc": roc_auc,
                        "balanced_accuracy": balanced_accuracy,
                    }
                    with open(filename, "wb") as f:
                        pickle.dump(results, f)
                ax[0].plot(np.arange(0, len(x_test)), f1_out, ls=method["linestyle"], label=method["name"])
                ax[1].plot(np.arange(0, len(x_test)), f1_in, ls=method["linestyle"], label=method["name"])
                ax[2].plot(np.arange(0, len(x_test)), roc_auc, ls=method["linestyle"], label=method["name"])
                ax[3].plot(np.arange(0, len(x_test)), balanced_accuracy, ls=method["linestyle"], label=method["name"])
    ax[0].title.set_text("f1 score on outliers")
    ax[0].legend()
    ax[1].title.set_text("f1 score on inliers")
    ax[1].legend()
    ax[2].title.set_text("roc auc score")
    ax[2].legend()
    ax[3].title.set_text("balanced accuracy score")
    ax[3].legend()
    plt.savefig(savename + ".png")
    plt.show()


def compute(methods, x_train, x_test, y_test, savename):
    y_decisions = dict()
    y_preds = dict()
    for method in methods:
        if type(method) == dict:
            filename = savename + "__" + method["name"] + ".pickle"
            try:
                with open(filename, "rb") as f:
                    results = pickle.load(f)
                y_decisions[method["name"]] = results["y_decision"]
                y_preds[method["name"]] = results["y_pred"]
            except FileNotFoundError:
                model = method["method"](**method["params"])
                model.fit(x_train)
                y_decision = np.zeros(len(y_test))
                y_pred = np.zeros(len(y_test))
                for i in tqdm(range(len(x_test)), desc=method["name"]):
                    y_decision[i] = model.eval_update(x_test[i].reshape(1, -1))
                    y_pred[i] = -1 if y_decision[i] < 0 else 1
                results = {
                    "y_decision": y_decision,
                    "y_pred": y_pred,
                }
                y_decisions[method["name"]] = y_decision
                y_preds[method["name"]] = y_pred
                with open(filename, "wb") as f:
                    pickle.dump(results, f)
    cols = ["f1_out", "f1_in", "roc_auc", "balanced_acc"]
    rows = []
    table = []
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        for method in methods:
            if type(method) == dict:
                rows.append(method["name"])
                f1_out = f1_score(y_test, y_preds[method["name"]], pos_label=-1)
                f1_in = f1_score(y_test, y_preds[method["name"]], pos_label=1)
                roc_auc = roc_auc_score(y_test, y_decisions[method["name"]])
                balanced_accuracy = balanced_accuracy_score(y_test, y_preds[method["name"]])
                res = [f1_out, f1_in, roc_auc, balanced_accuracy]
                table.append(['%1.3f' % v for v in res])
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=table, rowLabels=rows, colLabels=cols, loc="center")
    fig.tight_layout()
    plt.savefig("final_metrics_" + savename + ".png")
    plt.show()


if __name__ == "__main__":
    split_pos = 1000
    data = load_dataset("../res/conveyor.csv")
    x_train, y_train, x_test, y_test = split_data(data, split_pos)
    compute(METHODS, x_train, x_test, y_test, "results_conveyor_final_2")
