from odds import *
from odds.utils import load_dataset, split_data, compute_and_save, plot_and_save_results
import warnings


""" THIS SCRIPT IS AN EXAMPLE AND SHOULD NOT BE EXECUTED """


METHODS = [
    {
        "name": "DyCG",  # appears in the resulting dataset
        "method": DyCG,
        "params": {},
        "short_name": "DyCG"  # appears on graphs
    }, {
        "name": "DyCF d=2",
        "method": DyCF,
        "params": {
            "d": 2
        },
        "short_name": "DyCF (2)"
    }, {
        "name": "DyCF d=6",
        "method": DyCF,
        "params": {
            "d": 6,
        },
        "short_name": "DyCF (6)"
    }, {
        "name": "KDE W=200",
        "method": KDE,
        "params": {
            "threshold": 0.1,  # need to be fixed but not used for evaluation
            "win_size": 200,
        },
        "short_name": "KDE (200)"
    }, {
        "name": "SlidingMKDE W=2000",
        "method": KDE,
        "params": {
            "threshold": 0.1,  # need to be fixed but not used for evaluation
            "win_size": 2000,
        },
        "short_name": "KDE (2000)"
    }, {
        "name": "SmartSifter k=15, r=0.01, alpha=1.5",
        "method": SmartSifter,
        "params": {
            "threshold": 0.5,  # need to be fixed but not used for evaluation
            "k": 15,
            "r": 0.01,
            "alpha": 1.5,
        },
        "short_name": "SmartSifter (15,0.01,1.5)"
    }, {
        "name": "SmartSifter k=2, r=0.001, alpha=2.0",
        "method": SmartSifter,
        "params": {
            "threshold": 0.5,  # need to be fixed but not used for evaluation
            "k": 2,
            "r": 0.001,
            "alpha": 2.0,
        },
        "short_name": "SmartSifter (2,0.001,2)"
    }
]


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    data_dict = dict()

    """ Dataset #1 """
    split_pos = 500
    data = load_dataset("/path/to/dataset_1.csv")
    x_train, y_train, x_test, y_test = split_data(data, split_pos)
    del y_train
    data_dict["dataset_1"] = {
        "x_test": x_test,
        "x_train": x_train,
        "y_test": y_test,
    }

    """ Dataset #2 """
    split_pos = 15000
    data = load_dataset("/path/to/dataset_2.csv")
    x_train, y_train, x_test, y_test = split_data(data, split_pos)
    del y_train
    data_dict["dataset_2"] = {
        "x_test": x_test,
        "x_train": x_train,
        "y_test": y_test,
    }

    """ Dataset #3 """
    split_pos = 50000
    data = load_dataset("/path/to/dataset_3.csv")
    x_train, y_train, x_test, y_test = split_data(data, split_pos)
    del y_train
    data_dict["dataset_3"] = {
        "x_test": x_test,
        "x_train": x_train,
        "y_test": y_test,
    }

    compute_and_save(METHODS, data_dict, "/path/to/res")  # savename is a header for the files containing pickle models
    plot_and_save_results(METHODS, data_dict, "/path/to/res")  # savename is a header for the files containing graphs and metrics
