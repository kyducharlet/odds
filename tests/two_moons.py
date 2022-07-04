from dsod.distance import OSCOD, OSMCOD
from dsod.density import ILOF
from dsod.statistics import SlidingMKDE, SimpleChristoffel, DyCG
import warnings

from tests.conveyor import load_dataset, split_data, plot, METHODS


if __name__ == "__main__":
    warnings.filterwarnings("error")
    split_pos = 1000
    data = load_dataset("../res/two_moons.csv")
    x_train, y_train, x_test, y_test = split_data(data, split_pos)
    plot(METHODS, x_train, x_test, y_test, "results_two_moons_final")
