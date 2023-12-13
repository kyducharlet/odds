from odds.statistics import DyCF, DyCG, KDE, SmartSifter
from odds.distance import OSCOD
from odds.density import ILOF
from odds.utils import load_dataset, split_data, compute_and_save, plot_and_save_results
import warnings


METHODS = [
    {
        "name": "DyCG reg=vu",
        "method": DyCG,
        "params": {},
        "short_name": "DyCG"
    }, {
        "name": "DyCF d=6 (tm)",
        "method": DyCF,
        "params": {
            "d": 6,
        },
        "short_name": "DyCF (tm)"
    }, {
        "name": "DyCF d=8 (lc)",
        "method": DyCF,
        "params": {
            "d": 8,
        },
        "short_name": "DyCF (lc)"
    }, {
        "name": "DyCF d=2 (http)",
        "method": DyCF,
        "params": {
            "d": 2
        },
        "short_name": "DyCF (http)"
    }, {
        "name": "SlidingMKDE W=200 (tm)",
        "method": KDE,
        "params": {
            "threshold": 0.1,  # need to be fixed but not used for evaluation
            "win_size": 200,
        },
        "short_name": "S. MKDE (tm)"
    }, {
        "name": "SlidingMKDE W=5000 (lc & http)",
        "method": KDE,
        "params": {
            "threshold": 0.1,  # need to be fixed but not used for evaluation
            "win_size": 5000,
        },
        "short_name": "S. MKDE (lc & http)"
    }, {
        "name": "SmartSifter (tm)",
        "method": SmartSifter,
        "params": {
            "threshold": 0.5,  # need to be fixed but not used for evaluation
            "k": 15,
            "r": 0.01,
            "alpha": 1.5,
        },
        "short_name": "SmartSifter (tm)"
    }, {
        "name": "SmartSifter (lc)",
        "method": SmartSifter,
        "params": {
            "threshold": 0.5,  # need to be fixed but not used for evaluation
            "k": 10,
            "r": 0.01,
            "alpha": 1.0,
        },
        "short_name": "SmartSifter (lc)"
    }, {
        "name": "SmartSifter (http)",
        "method": SmartSifter,
        "params": {
            "threshold": 0.5,  # need to be fixed but not used for evaluation
            "k": 2,
            "r": 0.001,
            "alpha": 2.0,
        },
        "short_name": "SmartSifter (http)"
    }, {
        "name": "OSCOD R=1.2 W=200 (tm)",
        "method": OSCOD,
        "params": {
            "k": 10,  # need to be fixed but not used for evaluation
            "R": 1.2,
            "win_size": 200,
        },
        "short_name": "OSCOD (tm)"
    }, {
        "name": "OSCOD R=0.2 W=100 (lc)",
        "method": OSCOD,
        "params": {
            "k": 10,  # need to be fixed but not used for evaluation
            "R": 0.2,
            "win_size": 100,
        },
        "short_name": "OSCOD (lc)"
    }, {
        "name": "OSCOD R=0.5 W=5000 (http)",
        "method": OSCOD,
        "params": {
            "k": 10,  # need to be fixed but not used for evaluation
            "R": 0.5,
            "win_size": 5000,
        },
        "short_name": "OSCOD (http)"
    }, {
        "name": "iLOF k=5 W=200 (tm)",
        "method": ILOF,
        "params": {
            "k": 5,
            "threshold": 1,  # need to be fixed but not used for evaluation
            "win_size": 200,
        },
        "short_name": "iLOF (tm)"
    }, {
        "name": "iLOF k=25 W=100 (lc)",
        "method": ILOF,
        "params": {
            "k": 25,
            "threshold": 1,  # need to be fixed but not used for evaluation
            "win_size": 100,
        },
        "short_name": "iLOF (lc)"
    }, {
        "name": "iLOF k=30 W=500 (http)",
        "method": ILOF,
        "params": {
            "k": 30,
            "threshold": 1,  # need to be fixed but not used for evaluation
            "win_size": 500,
        },
        "short_name": "iLOF (http)"
    }
]


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

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
    data = load_dataset("../res/luggage_conveyor.csv")
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

    compute_and_save(METHODS, data_dict, "evaluating_methods")
    plot_and_save_results(METHODS, data_dict, "evaluating_methods")
