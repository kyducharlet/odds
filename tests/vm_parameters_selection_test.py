from dsod.density import ILOF, DILOF
import itertools as it
import warnings

from tests.methods_comparison import load_dataset, split_data, compute

PARAMS_ILOF = {
    "k": [5],
    "threshold": [0.8, 0.9, 0.95, 1, 1.05, 1.1, 1.2],
    "win_size": [100],
}

PARAMS_DILOF = {
    "k": [5],
    "threshold": [0.8, 0.9, 0.95, 1, 1.05, 1.1, 1.2],
    "win_size": [100],
    "step_size": [0.3],
    "reg_const": [1],
    "max_iter": [100],
}


def generate_methods_from_params(params, method_name, method_class):
    methods = []
    keys = list(params.keys())
    comb = list(it.product(*list(params.values())))
    for c in comb:
        method = dict()
        params_ = dict()
        name = method_name
        for i, k in enumerate(keys):
            name += f" {k}={c[i]}"
            params_[k] = c[i]
        method["name"] = name
        method["method"] = method_class
        method["params"] = params_
        methods.append(method)
    return methods


if __name__ == "__main__":
    warnings.filterwarnings("error")

    """ Dataset #1 """
    split_pos = 500
    data = load_dataset("../res/two_moons.csv")
    x_train, y_train, x_test, y_test = split_data(data, split_pos)
    del y_train
    METHODS_ILOF = generate_methods_from_params(PARAMS_ILOF, "ILOF", ILOF)
    compute(METHODS_ILOF, x_train, x_test, y_test, "choose_params_ilof_tm", multi_processing=True, show=False, close=True)
    METHODS_DILOF = generate_methods_from_params(PARAMS_DILOF, "DILOF", DILOF)
    compute(METHODS_DILOF, x_train, x_test, y_test, "choose_params_dilof_tm", multi_processing=True, show=False, close=True)
