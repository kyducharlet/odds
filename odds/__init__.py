from .statistics import KDE, SmartSifter, DyCF, DyCG
from .statistics import ImprovedKDE, ImprovedDyCF, ImprovedDyCG
from .distance import DBOKDE, DBOECF
from .density import MDEFKDE, MDEFECF, ILOF
from .plotter import LevelsetPlotter

__all__ = [
    "KDE",
    "SmartSifter",
    "DyCF",
    "DyCG",
    "ImprovedKDE",
    "ImprovedDyCF",
    "ImprovedDyCG",
    "DBOKDE",
    "DBOECF",
    "MDEFKDE",
    "MDEFECF",
    "ILOF",
    "LevelsetPlotter",
]

DICT_OF_METHODS = {
    "kde": KDE,
    "dycf": DyCF,
    "dycg": DyCG,
    "ikde": ImprovedKDE,
    "idycf": ImprovedDyCG,
    "idycg": ImprovedDyCG,
}
