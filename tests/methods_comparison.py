from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score, precision_score
from sklearn.exceptions import UndefinedMetricWarning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import pickle
from tqdm import tqdm
from multiprocessing import Pool
from pympler import asizeof
import time

from dsod.distance import OSCOD, OSMCOD
from dsod.density import ILOF
from dsod.statistics import SlidingMKDE, DyCF, DyCG


METHODS = [
    {
        "name": "ILOF k=10 W=1000 tau=1.1",
        "method": ILOF,
        "params": {
            "k": 10,
            "threshold": 1.1,
            "win_size": 1000,
        }
    }, {
        "name": "ILOF k=20 W=1000 tau=1.1",
        "method": ILOF,
        "params": {
            "k": 20,
            "threshold": 1.1,
            "win_size": 1000,
        }
    }, {
        "name": "ILOF k=10 W=1000 tau=1.05",
        "method": ILOF,
        "params": {
            "k": 10,
            "threshold": 1.05,
            "win_size": 1000,
        }
    }, {
        "name": "ILOF k=20 W=1000 tau=1.05",
        "method": ILOF,
        "params": {
            "k": 20,
            "threshold": 1.05,
            "win_size": 1000,
        }
    },
]


def average_precision_score(y_true, y_score, pos_label):
    y_score_ = y_score[y_true == pos_label]
    y_preds = [np.where(y_score - threshold < 0, -1, 1) for threshold in y_score_]
    precision_scores = [precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0) for y_pred in y_preds]
    return np.mean(precision_scores)


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


def multiprocessable_method(method, savename, x_train, x_test, y_test):
    if type(method) == dict:
        filename = savename + "__" + method["name"] + ".pickle"
        try:
            with open(filename, "rb") as f:
                pickle.load(f)
        except FileNotFoundError:
            start = time.time()
            model = method["method"](**method["params"])
            model.fit(x_train)
            fit_time = time.time() - start
            y_decision = np.zeros(len(y_test))
            y_pred = np.zeros(len(y_test))
            start = time.time()
            """for i in tqdm(range(len(x_test)), desc=method["name"]):"""
            for i in range(len(x_test)):
                y_decision[i] = model.eval_update(x_test[i].reshape(1, -1))
                y_pred[i] = -1 if y_decision[i] < 0 else 1
            eval_time = time.time() - start
            """print(f"{method['name']} size : {asizeof.asizeof(model)}")
            print(f"{method['name']} fit time : {fit_time}")
            print(f"{method['name']} eval time : {eval_time}")"""
            results = {
                "y_decision": y_decision,
                "y_pred": y_pred,
                "model_size": asizeof.asizeof(model),
                "model_fit_time": fit_time,
                "model_eval_time": eval_time,
            }
            print(f"Method {method['name']} over.")
            with open(filename, "wb") as f:
                pickle.dump(results, f)


def compute_vm(methods, data, savename, multi_processing):
    if not multi_processing:
        for method in methods:
            for k, v in data.items():
                multiprocessable_method(method, savename + '_' + k, **v)
    else:
        params = []
        for k, v in data.items():
            params.extend([(method, savename + '_' + k, v["x_train"], v["x_test"], v["y_test"]) for method in methods])
        pool = Pool()
        pool.starmap(multiprocessable_method, params)
        pool.close()


def compute(methods, x_train, x_test, y_test, savename, multi_processing=False, show=True, close=False):
    if not multi_processing:
        for method in methods:
            multiprocessable_method(method, savename, x_train, x_test, y_test)
    else:
        params = [(method, savename, x_train, x_test, y_test) for method in methods]
        pool = Pool()
        pool.starmap(multiprocessable_method, params)
        pool.close()
    y_decisions = dict()
    y_preds = dict()
    for method in methods:
        filename = savename + "__" + method["name"] + ".pickle"
        with open(filename, "rb") as f:
            results = pickle.load(f)
        y_decisions[method["name"]] = results["y_decision"]
        y_preds[method["name"]] = results["y_pred"]
    cols = ["AUROC", "AP"]
    # cols = ["f1_out", "f1_in", "f1_mean", "roc_auc", "balanced_acc"]
    rows = []
    table = []
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        for method in methods:
            if type(method) == dict:
                rows.append(method["name"])
                # y_pred = y_preds[method["name"]]
                y_deci = y_decisions[method["name"]]
                # f1_out = f1_score(y_test[~np.isnan(y_deci)], y_pred[~np.isnan(y_deci)], labels=[-1, 1], pos_label=-1)
                # f1_in = f1_score(y_test[~np.isnan(y_deci)], y_pred[~np.isnan(y_deci)], labels=[-1, 1], pos_label=1)
                # f1_mean = f1_score(y_test[~np.isnan(y_deci)], y_pred[~np.isnan(y_deci)], labels=[-1, 1], average="macro")
                roc_auc = roc_auc_score(y_test[~np.isnan(y_deci)], y_deci[~np.isnan(y_deci)])
                average_p = average_precision_score(y_test[~np.isnan(y_deci)], y_deci[~np.isnan(y_deci)], pos_label=-1)
                # balanced_accuracy = balanced_accuracy_score(y_test[~np.isnan(y_deci)], y_pred[~np.isnan(y_deci)])
                res = [roc_auc, average_p]
                # res = [f1_out, f1_in, f1_mean, roc_auc, average_p, balanced_accuracy]
                table.append(['%1.9f' % v for v in res])
    """fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=table, rowLabels=rows, colLabels=cols, loc="center")"""
    df = pd.DataFrame(data=table, columns=cols, index=rows)
    df.to_csv(f"final_metrics_{savename}.csv")
    # df.to_excel(f"final_metrics_{savename}.xlsx")
    """fig.tight_layout()
    plt.savefig("final_metrics_" + savename + ".png")
    if show:
        plt.show()
    if close:
        plt.close()"""


if __name__ == "__main__":
    warnings.filterwarnings("error")
    data = load_dataset("../res/two_moons.csv")
    split_pos = 1000
    x_train, y_train, x_test, y_test = split_data(data, split_pos)
    del y_train
    compute(METHODS, x_train, x_test, y_test, "comparison_2moons", show=False, close=True)
    data = load_dataset("../res/conveyor.csv")
    split_pos = 5000
    x_train, y_train, x_test, y_test = split_data(data, split_pos)
    del y_train
    compute(METHODS, x_train, x_test, y_test, "comparison_convey", show=False, close=True)
    data = load_dataset("../res/http.csv")
    split_pos = 10000
    x_train, y_train, x_test, y_test = split_data(data, split_pos)
    del y_train
    compute(METHODS, x_train, x_test, y_test, "comparison_http", show=False, close=True)
    data = load_dataset("../res/four_modes.csv")
    split_pos = 5000
    x_train, y_train, x_test, y_test = split_data(data, split_pos)
    del y_train
    compute(METHODS, x_train, x_test, y_test, "comparison_4modes", show=False, close=True)
