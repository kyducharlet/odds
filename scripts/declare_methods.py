import numpy as np
from odds import KDE, SmartSifter, DBOKDE, ILOF, DyCF, DyCG


experiment_method_params = {
    "KDE": KDE(0.1, 1000),
    "SmartSifter": SmartSifter(0.1, 12, 0.001, 1.5),
    "DBOKDE": DBOKDE(10, 0.1, 1000),
    "ILOF": ILOF(10, 1.1, 1000),
    "DyCF": DyCF(6),
    "DyCG": DyCG(),
}

kde_method_params = {
    "kde_200": KDE(0.1, 200),
    "kde_500": KDE(0.1, 500),
    "kde_1000": KDE(0.1, 1000),
    "kde_2000": KDE(0.1, 2000),
}

ss_method_params = {
    "ss_8_2": SmartSifter(threshold=0.1, k=8, r=0.01, alpha=1.2),
    "ss_8_5": SmartSifter(threshold=0.1, k=8, r=0.01, alpha=1.5),
    "ss_8_8": SmartSifter(threshold=0.1, k=8, r=0.01, alpha=1.8),
    "ss_12_2": SmartSifter(threshold=0.1, k=12, r=0.01, alpha=1.2),
    "ss_12_5": SmartSifter(threshold=0.1, k=12, r=0.01, alpha=1.5),
    "ss_12_8": SmartSifter(threshold=0.1, k=12, r=0.01, alpha=1.8),
    "ss_16_2": SmartSifter(threshold=0.1, k=16, r=0.01, alpha=1.2),
    "ss_16_5": SmartSifter(threshold=0.1, k=16, r=0.01, alpha=1.5),
    "ss_16_8": SmartSifter(threshold=0.1, k=16, r=0.01, alpha=1.8),
}

dbo_kde_method_params = {
    "dbo_kde_1_200": DBOKDE(k=10, R=0.1, win_size=200),
    "dbo_kde_1_500": DBOKDE(k=10, R=0.1, win_size=500),
    "dbo_kde_1_1000": DBOKDE(k=10, R=0.1, win_size=1000),
    "dbo_kde_5_200": DBOKDE(k=10, R=0.5, win_size=200),
    "dbo_kde_5_500": DBOKDE(k=10, R=0.5, win_size=500),
    "dbo_kde_5_1000": DBOKDE(k=10, R=0.5, win_size=1000),
}

ilof_method_params = {
    "ilof_10_200": ILOF(k=10, threshold=1.1, win_size=200),
    "ilof_10_500": ILOF(k=10, threshold=1.1, win_size=500),
    "ilof_10_1000": ILOF(k=10, threshold=1.1, win_size=1000),
    "ilof_20_200": ILOF(k=20, threshold=1.1, win_size=200),
    "ilof_20_500": ILOF(k=20, threshold=1.1, win_size=500),
    "ilof_20_1000": ILOF(k=20, threshold=1.1, win_size=1000),
}

dycf_method_params = {
    "dycf_2": DyCF(d=2),
    "dycf_4": DyCF(d=4),
    "dycf_6": DyCF(d=6),
    "dycf_8": DyCF(d=8),
}

dycg_method_params = {
    "dycg_3": DyCG(degrees=np.array([2, 3])),
    "dycg_4": DyCG(degrees=np.array([2, 4])),
    "dycg_4_6": DyCG(degrees=np.array([4, 6])),
    "dycg_6": DyCG(degrees=np.array([2, 6])),
    "dycg_8": DyCG(degrees=np.array([2, 8])),
}

METHOD_PARAMS = {
    "experiment": experiment_method_params,
    "kde_test": kde_method_params,
    "ss_test": ss_method_params,
    "dbo_kde_test": dbo_kde_method_params,
    "ilof_test": ilof_method_params,
    "dycf_test": dycf_method_params,
    "dycg_test": dycg_method_params,
}
