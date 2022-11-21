from dsod.distance import OSCOD, OSMCOD
from dsod.density import ILOF, DILOF
from dsod.statistics import SlidingMKDE, DyCF, DyCG
import warnings

from tests.methods_comparison import load_dataset, split_data, compute

METHODS = [
    {
        "name": "ILOF optim k=10",
        "method": ILOF,
        "params": {
            "k": 10,
            "win_size": 1000,
        }
    }, {
        "name": "DILOF 1 k=10",
        "method": DILOF,
        "params": {
            "k": 10,
            "win_size": 1000,
            "threshold": 1.1,
            "use_Ckn": False,
        }
    }, {
        "name": "DILOF 2 k=10",
        "method": DILOF,
        "params": {
            "k": 10,
            "win_size": 1000,
            "threshold": 1.1,
            "use_Ckn": True,
        }
    }
]


if __name__ == "__main__":
    warnings.filterwarnings("error")
    split_pos = 1000
    data = load_dataset("../res/two_moons.csv")
    x_train, y_train, x_test, y_test = split_data(data, split_pos)
    compute(METHODS, x_train, x_test, y_test, "compare_ilof_methods_two_moons_3k")
