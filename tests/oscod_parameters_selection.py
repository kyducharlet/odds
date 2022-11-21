from dsod.distance import OSCOD
import itertools as it
import warnings

from tests.methods_comparison import load_dataset, split_data, compute


PARAMS = {
    "k": [5],
    "R": [0.1, 0.2, 0.5, 1, 1.2, 1.5],
    "win_size": [100, 200, 500, 1000, 2000, 5000],
}


def generate_methods_from_params(params):
    methods = []
    keys = list(params.keys())
    comb = list(it.product(*list(params.values())))
    for c in comb:
        method = dict()
        params_ = dict()
        name = "OSCOD"
        for i, k in enumerate(keys):
            name += f" {k}={c[i]}"
            params_[k] = c[i]
        method["name"] = name
        method["method"] = OSCOD
        method["params"] = params_
        methods.append(method)
    return methods


if __name__ == "__main__":
    METHODS = generate_methods_from_params(PARAMS)
    warnings.filterwarnings("error")
    """split_pos = 500
    data = load_dataset("../res/two_moons.csv")
    x_train, y_train, x_test, y_test = split_data(data, split_pos)
    compute(METHODS, x_train, x_test, y_test, "choose_params_oscod_tm", multi_processing=True, show=False, close=True)"""
    """split_pos = 15000
    data = load_dataset("../res/conveyor.csv")
    x_train, y_train, x_test, y_test = split_data(data, split_pos)
    compute(METHODS, x_train, x_test, y_test, "choose_params_oscod_lc", show=False, close=True)"""
    split_pos = 50000
    data = load_dataset("../res/http.csv")
    x_train, y_train, x_test, y_test = split_data(data, split_pos)
    compute(METHODS, x_train, x_test, y_test, "choose_params_oscod_http", show=False, close=True)
