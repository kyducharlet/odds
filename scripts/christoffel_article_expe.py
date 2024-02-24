import pandas as pd
import numpy as np
import time
import os
import sys
import pickle

from odds.utils import roc_auc_score, average_precision_score
from odds.utils import em_goix, mv_goix
from odds.utils import asizeof
from odds import KDE, SmartSifter, DBOKDE, ILOF, DyCF, DyCG

from scripts.declare_methods import METHOD_PARAMS


def load_datasets(datasets):
    res_dict = dict()
    for (k, v) in datasets.items():
        data_files = [file for file in os.listdir(v) if file.endswith(".csv")]
        res_dict[k] = {f.split('_')[-1].split('.')[0]: pd.read_csv(v + f, index_col=0) for f in data_files}
    return res_dict


def raw_scoring(model):
    if type(model) is KDE:
        return lambda x: model.decision_function(x) + model.threshold
    elif type(model) is DyCF:
        return lambda x: model.decision_function(x) + 1
    elif type(model) is SmartSifter:
        return lambda x: model.decision_function(x) + model.threshold
    elif type(model) is DyCG:
        return lambda x: np.arctan(model.decision_function(x)) + (np.pi/2)
    elif type(model) is DBOKDE:
        return lambda x: np.log(1 / model.score_samples(x))
    elif type(model) is ILOF:
        return lambda x: 1 / (1 + model.score_samples(x))
    else:
        raise ValueError("Raw score is not implemented for this method.")


def em_auc_score(model, samples, random_generator, n_generated: int = 100000):
    t_max = 0.9
    lim_inf = samples.min(axis=0)
    lim_sup = samples.max(axis=0)
    volume_support = (lim_sup - lim_inf).prod()
    t = np.linspace(0, 100 / volume_support, n_generated)
    unif = random_generator.uniform(lim_inf, lim_sup,
                             size=(1000, 2))
    s_X = np.zeros(len(samples))
    for i, sample in enumerate(samples):
        s_X[i] = raw_scoring(model)(sample.reshape(1, -1)).item()
        model.update(sample.reshape(1, -1))
    s_unif = raw_scoring(model)(unif)
    res = em_goix(t, t_max, volume_support, s_unif, s_X, n_generated)
    return res[2]


def mv_auc_score(model, samples, random_generator, n_generated: int = 100000):
    alpha_min = 0.9
    alpha_max = 0.999
    axis_alpha = np.linspace(alpha_min, alpha_max, n_generated)
    lim_inf = samples.min(axis=0)
    lim_sup = samples.max(axis=0)
    volume_support = (lim_sup - lim_inf).prod()
    unif = random_generator.uniform(lim_inf, lim_sup,
                             size=(1000, 2))
    s_X = np.zeros(len(samples))
    for i, sample in enumerate(samples):
        s_X[i] = raw_scoring(model)(sample.reshape(1, -1)).item()
        model.update(sample.reshape(1, -1))
    s_unif = raw_scoring(model)(unif)
    res = mv_goix(axis_alpha, volume_support, s_unif, s_X, n_generated)
    return res[2]


def supervised_evaluation(methods, datasets, split_pos):
    res_dict = dict()
    cpt = 0
    for (k, v) in methods.items():
        print(f"Starting {k}...")
        for (k_, v_) in datasets.items():
            print(f">>> Starting {k_}...")
            for (i, df) in v_.items():
                if (type(v) not in [ILOF, DBOKDE]) or (int(i) <= 3):
                    print(f">>>>>> Sub dataset {i}.")
                    if os.path.exists(f"temp/supervised/{k}/{k_}/{i}"):
                        with open(f"temp/supervised/{k}/{k_}/{i}/temp.pkl", "rb") as f:
                            res_dict[cpt] = pickle.load(f)
                    else:
                        data = df.drop(columns=["Modes", "y"]).values
                        label = df["y"].values[split_pos:]
                        label = np.where(label < 0, -1, 1)
                        model = v.copy()
                        model.fit(data[:split_pos])
                        start = time.time()
                        scores = model.eval_update(data[split_pos:])
                        duration = time.time() - start
                        duration = duration / len(scores)
                        size = asizeof.asizeof(model)
                        auroc = roc_auc_score(label, scores)
                        ap = average_precision_score(label, scores)
                        res_dict[cpt] = {
                            "Method": k,
                            "Dataset": k_,
                            "Index": i,
                            "AUROC": auroc,
                            "AP": ap,
                            "Duration": duration,  # Average duration of the evaluation of a point and the update of the model with this point, in second
                            "Size": size,
                        }
                        os.makedirs(f"temp/supervised/{k}/{k_}/{i}")
                        with open(f"temp/supervised/{k}/{k_}/{i}/temp.pkl", "wb") as f:
                            pickle.dump(res_dict[cpt], f)
                    cpt += 1
    print("\n")
    return res_dict


def unsupervised_evaluation(methods, datasets, split_pos, rng):
    res_dict = {}
    cpt = 0
    for (k_, v_) in datasets.items():
        print(f"Starting {k_}...")
        for (i, df) in v_.items():
            print(f">>> Sub dataset {i}...")
            data = df[["speed", "intensity"]].dropna().values
            current_split_pos = split_pos
            valid_dataset = True
            while not (np.std(data[:current_split_pos], axis=0) != 0).all():
                current_split_pos += int(split_pos / 10)
                if current_split_pos > len(data):
                    valid_dataset = False
                    print(">>>>>> Failure.")
            if valid_dataset:
                print(f">>>>>> Starting from {current_split_pos}...")
                for (k, v) in methods.items():
                    if (type(v) not in [ILOF, DBOKDE]) or (int(i) <= 3):
                        print(f">>>>>> Method {k}.")
                        if os.path.exists(f"temp/unsupervised/{k}/{k_}/{i}"):
                            with open(f"temp/unsupervised/{k}/{k_}/{i}/temp.pkl", "rb") as f:
                                res_dict[cpt] = pickle.load(f)
                        else:
                            scores = np.zeros(len(data))
                            model = v.copy()
                            model.fit(data[:current_split_pos])
                            start = time.time()
                            scores[current_split_pos:] = model.eval_update(data[current_split_pos:])
                            em = em_auc_score(model, data[current_split_pos:], rng)
                            mv = mv_auc_score(model, data[current_split_pos:], rng)
                            duration = time.time() - start
                            duration = duration / len(scores)
                            size = asizeof.asizeof(model)
                            res_dict[cpt] = {
                                "Method": k,
                                "Dataset": k_,
                                "Index": i,
                                "EM": em,
                                "MV": mv,
                                "Duration": duration,  # Average duration of the evaluation of a point and the update of the model with this point, in second
                                "Size": size,
                            }
                            os.makedirs(f"temp/unsupervised/{k}/{k_}/{i}")
                            with open(f"temp/unsupervised/{k}/{k_}/{i}/temp.pkl", "wb") as f:
                                pickle.dump(res_dict[cpt], f)
                        cpt += 1
    print("\n")
    return res_dict


if __name__ == "__main__":
    methods = METHOD_PARAMS["experiment"]
    rng = np.random.RandomState(np.random.PCG64())

    src = "../data/synthetics/experiment/"
    supervised_split_pos = 1000
    supervised_datasets = load_datasets({
        name: src + name + "/" for name in os.listdir(src) if name.startswith("synthetic")
    })
    supervised_res = supervised_evaluation(methods, supervised_datasets, supervised_split_pos)
    supervised_df = pd.DataFrame.from_dict(supervised_res, orient="index")
    supervised_df.to_csv(f"results_synthetics_experiment.csv")

    src = "../data/conveyors/experiment/"
    unsupervised_initial_split_pos = 5000
    unsupervised_datasets = load_datasets({
        name: src + name + "/" for name in os.listdir(src) if name.startswith("MOTE")
    })
    unsupervised_res = unsupervised_evaluation(methods, unsupervised_datasets, unsupervised_initial_split_pos, rng)
    unsupervised_df = pd.DataFrame.from_dict(unsupervised_res, orient="index")
    unsupervised_df.to_csv(f"results_conveyors_experiment.csv")
