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


"""METHODS = [
    {
        "name": "DyCG",
        "method": DyCG,
        "params": {}
    }, {
        "name": "DyCF d=6 (tm)",
        "method": DyCF,
        "params": {
            "d": 6,
        }
    }, {
        "name": "DyCF d=8 (lc)",
        "method": DyCF,
        "params": {
            "d": 8,
        }
    }, {
        "name": "DyCF d=2 (http)",
        "method": DyCF,
        "params": {
            "d": 2
        }
    }, {
        "name": "SlidingMKDE W=200 (tm)",
        "method": SlidingMKDE,
        "params": {
            "win_size": 200,
        }
    }, {
        "name": "SlidingMKDE W=5000 (lc & http)",
        "method": SlidingMKDE,
        "params": {
            "win_size": 5000,
        }
    }, {
        "name": "OSCOD R=1.2 W=200 (tm)",
        "method": OSCOD,
        "params": {
            "k": 10,  # need to be fixed but not used for evaluation
            "R": 1.2,
            "win_size": 200,
        }
    }, {
        "name": "OSCOD R=0.2 W=100 (lc)",
        "method": OSCOD,
        "params": {
            "k": 10,  # need to be fixed but not used for evaluation
            "R": 0.2,
            "win_size": 100,
        }
    }, {
        "name": "OSCOD R=0.5 W=5000 (http)",
        "method": OSCOD,
        "params": {
            "k": 10,  # need to be fixed but not used for evaluation
            "R": 0.5,
            "win_size": 5000,
        }
    },
]"""


METHODS = [
    {
        "name": "DyCG reg=vu",
        "method": DyCG,
        "params": {
            "reg": "vu"
        }
    }, {
        "name": "DyCG reg=alpha",
        "method": DyCG,
        "params": {
            "reg": "alpha"
        }
    }, {
        "name": "DyCG reg=d_alpha",
        "method": DyCG,
        "params": {
            "reg": "d_alpha"
        }
    }, {
        "name": "DyCF d=6 (tm)",
        "method": DyCF,
        "params": {
            "d": 6,
        }
    }, {
        "name": "DyCF d=8 (lc)",
        "method": DyCF,
        "params": {
            "d": 8,
        }
    }, {
        "name": "DyCF d=2 (http)",
        "method": DyCF,
        "params": {
            "d": 2
        }
    }, {
        "name": "SlidingMKDE W=200 (tm)",
        "method": SlidingMKDE,
        "params": {
            "win_size": 200,
        }
    }, {
        "name": "SlidingMKDE W=5000 (lc & http)",
        "method": SlidingMKDE,
        "params": {
            "win_size": 5000,
        }
    }, {
        "name": "OSCOD R=1.2 W=200 (tm)",
        "method": OSCOD,
        "params": {
            "k": 10,  # need to be fixed but not used for evaluation
            "R": 1.2,
            "win_size": 200,
        }
    }, {
        "name": "OSCOD R=0.2 W=100 (lc)",
        "method": OSCOD,
        "params": {
            "k": 10,  # need to be fixed but not used for evaluation
            "R": 0.2,
            "win_size": 100,
        }
    }, {
        "name": "OSCOD R=0.5 W=5000 (http)",
        "method": OSCOD,
        "params": {
            "k": 10,  # need to be fixed but not used for evaluation
            "R": 0.5,
            "win_size": 5000,
        }
    }, {
        "name": "iLOF k=5 W=200",
        "method": ILOF,
        "params": {
            "k": 5,
            "win_size": 200,
        }
    }
]


"""METHODS = [
    {
        "name": "DyCG reg=vu",
        "method": DyCG,
        "params": {
            "reg": "vu"
        }
    }, {
        "name": "DyCG reg=alpha",
        "method": DyCG,
        "params": {
            "reg": "alpha"
        }
    }, {
        "name": "DyCG reg=d_alpha",
        "method": DyCG,
        "params": {
            "reg": "d_alpha"
        }
    },
]
    }
]"""


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
            for i in tqdm(range(len(x_test)), desc=method["name"]):
                """for i in range(len(x_test)):"""
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
            """print(f"Method {method['name']} over.")"""
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


def process_vm_results(methods, data, savename):
    for k, v in data.items():
        cols = ["AUROC", "AP", "Duration (fit)", "Duration (eval)", "Memory size"]
        rows = []
        table = []
        for method in methods:
            filename = savename + f"_{k}__{method['name']}.pickle"
            with open(filename, "rb") as f:
                results = pickle.load(f)
            y_decision = results["y_decision"]
            y_test = v["y_test"]
            rows.append(method["name"])
            roc_auc = roc_auc_score(y_test[~np.isnan(y_decision)], y_decision[~np.isnan(y_decision)])
            average_p = average_precision_score(y_test[~np.isnan(y_decision)], y_decision[~np.isnan(y_decision)], pos_label=-1)
            fit_time = results["model_fit_time"]
            eval_time = results["model_eval_time"]
            model_size = results["model_size"]
            res_metric = [roc_auc, average_p]
            res_int = [fit_time, eval_time, model_size]
            table.append(['%1.9f' % v for v in res_metric] + [str(v) for v in res_int])
        df = pd.DataFrame(data=table, columns=cols, index=rows)
        df.to_csv(f"final_metrics_{savename}_{k}.csv")


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

    data_dict = dict()

    """ Dataset #1 """
    split_pos = 500
    data = load_dataset("../res/two_moons.csv")
    x_train, y_train, x_test, y_test = split_data(data, split_pos)
    del y_train
    data_dict["tm"] = {
        "x_test": x_test,
        "x_train": x_train,
        "y_test": y_test,
    }

    """ Dataset #2 """
    split_pos = 15000
    data = load_dataset("../res/conveyor.csv")
    x_train, y_train, x_test, y_test = split_data(data, split_pos)
    del y_train
    data_dict["lc"] = {
        "x_test": x_test,
        "x_train": x_train,
        "y_test": y_test,
    }

    """ Dataset #3 """
    split_pos = 50000
    data = load_dataset("../res/http.csv")
    x_train, y_train, x_test, y_test = split_data(data, split_pos)
    del y_train
    data_dict["http"] = {
        "x_test": x_test,
        "x_train": x_train,
        "y_test": y_test,
    }

    compute_vm(METHODS, data_dict, "final_results", multi_processing=False)
    process_vm_results(METHODS, data_dict, "final_results")
