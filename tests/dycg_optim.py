from dsod.statistics import SimpleChristoffel, DyCG
import warnings
from tests.conveyor import load_dataset, split_data, compute


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
