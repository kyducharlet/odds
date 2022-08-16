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


METHODS = [
    {
        "name": "SimpleChristoffel (d=2)",
        "method": SimpleChristoffel,
        "params": {
            "d": 2,
        }
    }, {
        "name": "SimpleChristoffel (d=4)",
        "method": SimpleChristoffel,
        "params": {
            "d": 4,
        }
    }, {
        "name": "SimpleChristoffel (d=6)",
        "method": SimpleChristoffel,
        "params": {
            "d": 6,
        }
    }, {
        "name": "SimpleChristoffel (d=8)",
        "method": SimpleChristoffel,
        "params": {
            "d": 8,
        }
    }, {
        "name": "DyCG 1 - 1",
        "method": DyCG,
        "params": {
            "decision": "mean_growth",
            "reg": "1",
        }
    }, {
        "name": "DyCG 1 - 2",
        "method": DyCG,
        "params": {
            "decision": "mean_growth",
            "reg": "2",
        }
    }, {
        "name": "DyCG 1 - 3",
        "method": DyCG,
        "params": {
            "decision": "mean_growth",
            "reg": "3",
        }
    }, {
        "name": "DyCG 1 - 4",
        "method": DyCG,
        "params": {
            "decision": "mean_growth",
            "reg": "4",
        }
    }, {
        "name": "DyCG 1 - 5",
        "method": DyCG,
        "params": {
            "decision": "mean_growth",
            "reg": "5",
        }
    }, {
        "name": "DyCG 1 - 6",
        "method": DyCG,
        "params": {
            "decision": "mean_growth",
            "reg": "6",
        }
    }, {
        "name": "DyCG 2 - 1",
        "method": DyCG,
        "params": {
            "decision": "mean_growth_rate",
            "reg": "1",
        }
    }, {
        "name": "DyCG 2 - 2",
        "method": DyCG,
        "params": {
            "decision": "mean_growth_rate",
            "reg": "2",
        }
    }, {
        "name": "DyCG 2 - 3",
        "method": DyCG,
        "params": {
            "decision": "mean_growth_rate",
            "reg": "3",
        }
    }, {
        "name": "DyCG 2 - 4",
        "method": DyCG,
        "params": {
            "decision": "mean_growth_rate",
            "reg": "4",
        }
    }, {
        "name": "DyCG 2 - 5",
        "method": DyCG,
        "params": {
            "decision": "mean_growth_rate",
            "reg": "5",
        }
    }, {
        "name": "DyCG 2 - 6",
        "method": DyCG,
        "params": {
            "decision": "mean_growth_rate",
            "reg": "6",
        }
    }, {
        "name": "DyCG 3 - 1",
        "method": DyCG,
        "params": {
            "decision": "mean_score",
            "reg": "1",
        }
    }, {
        "name": "DyCG 3 - 2",
        "method": DyCG,
        "params": {
            "decision": "mean_score",
            "reg": "2",
        }
    }, {
        "name": "DyCG 3 - 3",
        "method": DyCG,
        "params": {
            "decision": "mean_score",
            "reg": "3",
        }
    }, {
        "name": "DyCG 3 - 4",
        "method": DyCG,
        "params": {
            "decision": "mean_score",
            "reg": "4",
        }
    }, {
        "name": "DyCG 3 - 5",
        "method": DyCG,
        "params": {
            "decision": "mean_score",
            "reg": "5",
        }
    }, {
        "name": "DyCG 3 - 6",
        "method": DyCG,
        "params": {
            "decision": "mean_score",
            "reg": "6",
        }
    }, {
        "name": "DyCG 4 - 1",
        "method": DyCG,
        "params": {
            "decision": "sign_poly_2_reg",
            "reg": "1",
        }
    }, {
        "name": "DyCG 4 - 2",
        "method": DyCG,
        "params": {
            "decision": "sign_poly_2_reg",
            "reg": "2",
        }
    }, {
        "name": "DyCG 4 - 3",
        "method": DyCG,
        "params": {
            "decision": "sign_poly_2_reg",
            "reg": "3",
        }
    }, {
        "name": "DyCG 4 - 4",
        "method": DyCG,
        "params": {
            "decision": "sign_poly_2_reg",
            "reg": "4",
        }
    }, {
        "name": "DyCG 4 - 5",
        "method": DyCG,
        "params": {
            "decision": "sign_poly_2_reg",
            "reg": "5",
        }
    }, {
        "name": "DyCG 4 - 6",
        "method": DyCG,
        "params": {
            "decision": "sign_poly_2_reg",
            "reg": "6",
        }
    }
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


def compute(methods, x_train, x_test, y_test, savename, close=False):
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
    cols = ["f1_out", "f1_in", "f1_mean", "roc_auc", "balanced_acc"]
    rows = []
    table = []
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        for method in methods:
            if type(method) == dict:
                rows.append(method["name"])
                y_pred = y_preds[method["name"]]
                y_deci = y_decisions[method["name"]]
                f1_out = f1_score(y_test[~np.isnan(y_deci)], y_pred[~np.isnan(y_deci)], labels=[-1, 1], pos_label=-1)
                f1_in = f1_score(y_test[~np.isnan(y_deci)], y_pred[~np.isnan(y_deci)], labels=[-1, 1], pos_label=1)
                f1_mean = f1_score(y_test[~np.isnan(y_deci)], y_pred[~np.isnan(y_deci)], labels=[-1, 1], average="macro")
                roc_auc = roc_auc_score(y_test[~np.isnan(y_deci)], y_deci[~np.isnan(y_deci)])
                balanced_accuracy = balanced_accuracy_score(y_test[~np.isnan(y_deci)], y_pred[~np.isnan(y_deci)])
                res = [f1_out, f1_in, f1_mean, roc_auc, balanced_accuracy]
                table.append(['%1.3f' % v for v in res])
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=table, rowLabels=rows, colLabels=cols, loc="center")
    df = pd.DataFrame(data=table, columns=cols, index=rows)
    df.to_excel(f"final_metrics_{savename}.xlsx")
    fig.tight_layout()
    plt.savefig("final_metrics_" + savename + ".png")
    plt.show()
    if close:
        plt.close()


if __name__ == "__main__":
    warnings.filterwarnings("error")
    data = load_dataset("../res/two_moons.csv")
    split_pos = 1000
    x_train, y_train, x_test, y_test = split_data(data, split_pos)
    del y_train
    compute(METHODS, x_train, x_test, y_test, "dycg_optim_2moons", close=True)
    data = load_dataset("../res/conveyor.csv")
    split_pos = 5000
    x_train, y_train, x_test, y_test = split_data(data, split_pos)
    del y_train
    compute(METHODS, x_train, x_test, y_test, "dycg_optim_convey", close=True)
    data = load_dataset("../res/http.csv")
    split_pos = 10000
    x_train, y_train, x_test, y_test = split_data(data, split_pos)
    del y_train
    compute(METHODS, x_train, x_test, y_test, "dycg_optim_http", close=True)
    data = load_dataset("../res/four_modes.csv")
    split_pos = 5000
    x_train, y_train, x_test, y_test = split_data(data, split_pos)
    del y_train
    compute(METHODS, x_train, x_test, y_test, "dycg_optim_4modes", close=True)
