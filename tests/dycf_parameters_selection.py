from dsod.statistics import DyCF
import warnings

from tests.methods_comparison import load_dataset, split_data, compute

METHODS = [
    {
        "name": "DyCF d=2",
        "method": DyCF,
        "params": {
            "d": 2,
        }
    }, {
        "name": "DyCF d=4",
        "method": DyCF,
        "params": {
            "d": 4,
        }
    }, {
        "name": "DyCF d=6",
        "method": DyCF,
        "params": {
            "d": 6,
        }
    }, {
        "name": "DyCF d=8",
        "method": DyCF,
        "params": {
            "d": 8,
        }
    },
]


if __name__ == "__main__":
    warnings.filterwarnings("error")
    split_pos = 500
    data = load_dataset("../res/two_moons.csv")
    x_train, y_train, x_test, y_test = split_data(data, split_pos)
    compute(METHODS, x_train, x_test, y_test, "choose_params_dycf_tm", show=False, close=True)
    split_pos = 15000
    data = load_dataset("../res/conveyor.csv")
    x_train, y_train, x_test, y_test = split_data(data, split_pos)
    compute(METHODS, x_train, x_test, y_test, "choose_params_dycf_lc", show=False, close=True)
    split_pos = 50000
    data = load_dataset("../res/http.csv")
    x_train, y_train, x_test, y_test = split_data(data, split_pos)
    compute(METHODS, x_train, x_test, y_test, "choose_params_dycf_http", show=False, close=True)
