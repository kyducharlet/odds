import warnings

from tests.conveyor import load_dataset, split_data, compute, METHODS


if __name__ == "__main__":
    warnings.filterwarnings("error")
    split_pos = 1000
    data = load_dataset("../res/http.csv")
    x_train, y_train, x_test, y_test = split_data(data, split_pos)
    compute(METHODS, x_train, x_test, y_test, "results_http")
