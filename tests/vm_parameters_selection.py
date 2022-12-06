from dsod.density import ILOF, DILOF
import itertools as it
import warnings

from tests.methods_comparison import load_dataset, split_data, compute_vm


PARAMS_ILOF = {
    "k": [5, 10, 15, 20, 25, 30],
    "threshold": [1],
    "win_size": [100, 200, 500, 1000, 2000, 5000],
}


PARAMS_DILOF = {
    "k": [5, 10, 15, 20, 25, 30],
    "threshold": [1],
    "win_size": [100, 200, 500, 1000, 2000, 5000],
    "step_size": [0.1, 0.2, 0.3, 0.5, 0.7],
    "reg_const": [1],
    "max_iter": [100, 200, 500, 1000],
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

    METHODS_ILOF = generate_methods_from_params(PARAMS_ILOF, "ILOF", ILOF)
    compute_vm(METHODS_ILOF, data_dict, "choose_params_ilof", multi_processing=True)
    """METHODS_DILOF = generate_methods_from_params(PARAMS_DILOF, "DILOF", DILOF)
    compute_vm(METHODS_DILOF, data_dict, "choose_params_dilof", multi_processing=True)"""
