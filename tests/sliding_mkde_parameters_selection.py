from dsod.statistics import SlidingMKDE
import warnings

from tests.methods_comparison import load_dataset, split_data, compute

METHODS = [
    {
        "name": "Sliding MKDE W=100",
        "method": SlidingMKDE,
        "params": {
            "win_size": 100,
        }
    }, {
        "name": "Sliding MKDE W=200",
        "method": SlidingMKDE,
        "params": {
            "win_size": 200,
        }
    }, {
        "name": "Sliding MKDE W=500",
        "method": SlidingMKDE,
        "params": {
            "win_size": 500,
        }
    }, {
        "name": "Sliding MKDE W=1000",
        "method": SlidingMKDE,
        "params": {
            "win_size": 1000,
        }
    }, {
        "name": "Sliding MKDE W=2000",
        "method": SlidingMKDE,
        "params": {
            "win_size": 2000,
        }
    }, {
        "name": "Sliding MKDE W=5000",
        "method": SlidingMKDE,
        "params": {
            "win_size": 5000,
        }
    }
]


if __name__ == "__main__":
    warnings.filterwarnings("error")
    split_pos = 500
    data = load_dataset("../res/two_moons.csv")
    x_train, y_train, x_test, y_test = split_data(data, split_pos)
    compute(METHODS, x_train, x_test, y_test, "choose_params_sliding_mkde_tm", show=False, close=True)
    split_pos = 15000
    data = load_dataset("../res/conveyor.csv")
    x_train, y_train, x_test, y_test = split_data(data, split_pos)
    compute(METHODS, x_train, x_test, y_test, "choose_params_sliding_mkde_lc", show=False, close=True)
    split_pos = 50000
    data = load_dataset("../res/http.csv")
    x_train, y_train, x_test, y_test = split_data(data, split_pos)
    compute(METHODS, x_train, x_test, y_test, "choose_params_sliding_mkde_http", show=False, close=True)
