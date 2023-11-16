import numpy as np
import pandas as pd
from scipy.linalg import inv, pinv
from scipy.integrate import nquad, trapezoid
from math import comb, factorial
import itertools
import pickle
from tqdm import tqdm
from multiprocessing import Pool
from pympler import asizeof
import time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score
from sklearn.base import clone
from smartsifter import SDEM
import warnings


""" KDE: Kernel functions """


def gaussian_kernel(x):
    if x.shape[0] == 1:
        return np.exp(-1 * np.dot(x, x.T)[0, 0] / 2) / np.power(np.sqrt(2 * np.pi), x.shape[1])
    else:
        return np.array([gaussian_kernel(point.reshape(1, -1)) for point in x])


def epanechnikov_kernel(x):
    return np.array([np.product([0.75 * (1 - x[i, j] ** 2) for j in range(x.shape[1])]) if (x[i] ** 2 <= 1).all() else 0 for i in range(x.shape[0])])


IMPLEMENTED_KERNEL_FUNCTIONS = {
    "gaussian": gaussian_kernel,
    "epanechnikov": epanechnikov_kernel,
}


""" SlidingMKDE: Bandwidth estimators """


def scott_rule(x):
    ev = np.sqrt(5) * np.maximum(np.std(x, axis=0), 1e-32 * np.ones(x.shape[1])) / np.power(x.shape[0], 1 / (x.shape[1] + 4))
    return np.diag(1 / ev), np.product(1 / ev)


def scott_rule_with_R(x):
    ev = np.sqrt(5) * np.maximum(np.std(x, axis=0), 1e-32 * np.ones(x.shape[1])) / np.power(x.shape[0], 1 / (x.shape[1] + 4))
    R = 0.5 * np.linalg.norm(ev) / np.sqrt(x.shape[1])
    return R, np.diag(1 / ev), np.product(1 / ev)


IMPLEMENTED_BANDWIDTH_ESTIMATORS = {
    "scott": scott_rule,
    "scott_with_R": scott_rule_with_R,
}


""" DBOKDE / MDEFKDE : Estimation du nombre de voisins dans un certain rayon """


def neighbours_count(x, okc, bsi, bdsi, ws, ss, R):
    B = 1 / np.diagonal(bsi)
    return (ws / ss) * 0.75 ** len(x) * bdsi * np.sum([
        np.product((np.minimum(x + R, kc + B) - np.maximum(x - R, kc - B)) - (1 / 3) * np.dot(np.square(bsi), np.power(np.minimum(x + R, kc + B) - kc, 3) - np.power(np.maximum(x - R, kc - B) - kc, 3)))
        for kc in okc
    ])


""" MDEFKDE : Estimation du nombre de voisins dans les cellules d'une grille """


def neighbours_counts_in_grid(x, kcs, bsi, bdsi, ws, ss, m, r):
    cells = itertools.product(*[range(2 ** m) for i in range(len(x))])
    start = x - (2 ** m - 1) * r
    counts = []
    for cell in cells:
        center = start + 2 * np.array(cell) * r
        overlapping_kc = [kc for kc in kcs if (np.abs(kc - center) < (1 / np.diagonal(bsi)) + r).all()]
        counts.append(neighbours_count(center, overlapping_kc, bsi, bdsi, ws, ss, r))
    return np.array(counts)


""" SmartSifter: Scoring functions """


def likelihood(x, sdem):
    return np.exp(sdem.score_samples(x))


def logarithmic_loss(x, sdem: SDEM):
    return -1 * sdem.score_samples(x)


def hellinger_score(x, sdem: SDEM):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    """ We first need to clone the model and its parameters to be able to compute score even without updating """
    sdem_clone = clone(sdem)
    sdem_clone.covariances_ = sdem.covariances_
    sdem_clone.covariances_bar = sdem.covariances_bar
    sdem_clone.means_ = sdem.means_
    sdem_clone.means_bar = sdem.means_bar
    sdem_clone.precisions_ = sdem.precisions_
    sdem_clone.precisions_cholesky_ = sdem.precisions_cholesky_
    sdem_clone.weights_ = sdem.weights_
    old_covariances_inv = np.array([np.linalg.inv(cov) for cov in sdem.covariances_bar])
    """ Update of the clone """
    sdem_clone.update(x)
    new_covariances_inv = np.array([np.linalg.inv(cov) for cov in sdem_clone.covariances_bar])
    """ Score computation """
    res = np.square(np.sqrt(sdem_clone.weights_) - np.sqrt(sdem.weights_))
    mean_weights = np.mean(np.concatenate([sdem_clone.weights_.reshape(1, -1), sdem.weights_.reshape(1, -1)], axis=0), axis=0)
    dist_elts = []
    for i in range(len(sdem.weights_)):
        a = 2 / (np.sqrt(np.linalg.det((new_covariances_inv[i] + old_covariances_inv[i]) / 2)) * np.power(np.linalg.det(sdem_clone.covariances_bar[i]) * np.linalg.det(sdem.covariances_bar[i]), 1/4))
        b = np.dot(new_covariances_inv[i], sdem_clone.means_bar[i].reshape(-1, 1)) + np.dot(old_covariances_inv[i], sdem.means_bar[i].reshape(-1, 1))
        c = np.linalg.inv(new_covariances_inv[i] + old_covariances_inv[i])
        d = np.dot(np.dot(sdem_clone.means_bar[i].reshape(1, -1), new_covariances_inv[i]), sdem_clone.means_bar[i].reshape(-1, 1)) + np.dot(np.dot(sdem.means_bar[i].reshape(1, -1), old_covariances_inv[i]), sdem.means_bar[i].reshape(-1, 1))
        dist_elts.append(2 - a * np.exp(np.dot(np.dot(b.T, c), b) / 2) * np.exp(d / 2))
    res += np.dot(mean_weights.reshape(-1), np.array(dist_elts).reshape(-1))
    return np.sum(res), [sdem_clone.covariances_, sdem_clone.covariances_bar, sdem_clone.means_, sdem_clone.means_bar, sdem_clone.precisions_, sdem_clone.precisions_cholesky_, sdem_clone.weights_]


IMPLEMENTED_SS_SCORING_FUNCTIONS = {
    "likelihood": likelihood,
    "logloss": logarithmic_loss,
    "hellinger": hellinger_score,
}


""" DyCF: Polynomials basis functions """


def monomials(x, n):
    return np.power(x, n)


def chebyshev_t_1(x, n):  # Orthonormé sur [-1, 1] selon la mesure de Lebesgue avec 1 / sqrt(1 - x**2) comme poids
    if x < -1:
        return (-1)**n * np.cosh(n * np.arccosh(-x)) / np.sqrt((np.pi if n == 0 else np.pi / 2))
    elif x > 1:
        return np.cosh(n * np.arccosh(x)) / np.sqrt((np.pi if n == 0 else np.pi / 2))
    else:
        return np.cos(n * np.arccos(x)) / np.sqrt((np.pi if n == 0 else np.pi / 2))


def chebyshev_t_2(x, n):  # Orthonormé sur [-1, 1] selon la mesure de Lebesgue avec 1 / sqrt(1 - x**2) comme poids
    if n == 0:
        return 1 / np.sqrt(np.pi)
    else:
        return (n / np.sqrt(np.pi / 2)) * np.sum([(-2)**i * (factorial(n + i - 1) / (factorial(n - i) * factorial(2 * i))) * (1 - x)**i for i in range(n+1)])


def chebyshev_u(x, n):
    if n == 0:
        return np.sqrt(2 / np.pi)
    else:
        return np.sqrt(2 / np.pi) * np.sum([(-2)**i * (factorial(n + i - 1) / (factorial(n - i) * factorial(2 * i + 1))) * (1 - x)**i for i in range(n+1)])


def legendre(x, n):  # # Orthonormé sur [-1, 1] selon la mesure de Lebesgue
    return np.sqrt((2*n + 1) / 2) * np.sum([comb(n, i) * comb(n+i, i) * ((x - 1) / 2)**i for i in range(n+1)])


IMPLEMENTED_POLYNOMIAL_BASIS = {
    "monomials": monomials,
    "legendre": np.vectorize(legendre),
    "chebyshev": np.vectorize(chebyshev_u),
    "chebyshev_t_1": np.vectorize(chebyshev_t_1),
    "chebyshev_t_2": np.vectorize(chebyshev_t_2),
    "chebyshev_u": np.vectorize(chebyshev_u),
}


""" DyCF: Incrementation options """


def inverse_increment(mm, x, n, inv_opt):
    moments_matrix = n * mm.__dict__["__moments_matrix"]
    for xx in x:
        v = mm.polynomial_func(xx, mm.__dict__["__monomials"])
        moments_matrix += np.dot(v, v.T)
    moments_matrix /= (n + x.shape[0])
    mm.__dict__["__moments_matrix"] = moments_matrix
    mm.__dict__["__inverse_moments_matrix"] = inv_opt(moments_matrix)


def sherman_increment(mm, x, n, inv_opt):
    inv_moments_matrix = mm.__dict__["__inverse_moments_matrix"] / n
    for xx in x:
        v = mm.polynomial_func(xx, mm.__dict__["__monomials"])
        a = np.matmul(np.matmul(inv_moments_matrix, np.dot(v, v.T)), inv_moments_matrix)
        b = np.dot(np.dot(v.T, inv_moments_matrix), v)
        inv_moments_matrix -= a / (1 + b)
    mm.__dict__["__inverse_moments_matrix"] = (n + x.shape[0]) * inv_moments_matrix


IMPLEMENTED_INCREMENTATION_OPTIONS = {
    "inverse": inverse_increment,
    "sherman": sherman_increment,
}


""" DyCF: Moments matrix """


class MomentsMatrix:
    def __init__(self, d, incr_opt="inverse", polynomial_basis="monomials", inv_opt="inv"):
        assert polynomial_basis in IMPLEMENTED_POLYNOMIAL_BASIS.keys()
        assert incr_opt in IMPLEMENTED_INCREMENTATION_OPTIONS.keys()
        self.d = d
        self.polynomial_func = lambda x, m: PolynomialsBasis.apply_combinations(x, m, IMPLEMENTED_POLYNOMIAL_BASIS[polynomial_basis])
        self.incr_func = IMPLEMENTED_INCREMENTATION_OPTIONS[incr_opt]
        self.inv_opt = pinv if inv_opt == "pinv" else inv
        self.monomials = None
        self.moments_matrix = None
        self.inverse_moments_matrix = None

    def fit(self, x):
        monomials = PolynomialsBasis.generate_combinations(self.d, x.shape[1])
        self.monomials = monomials
        len_m = len(monomials)
        moments_matrix = np.zeros((len_m, len_m), dtype=x.dtype)
        for xx in x:
            v = self.polynomial_func(xx, monomials)
            moments_matrix += np.dot(v, v.T)
        moments_matrix /= len(x)
        self.moments_matrix = moments_matrix
        self.inverse_moments_matrix = self.inv_opt(moments_matrix)
        return self

    def score_samples(self, x):
        res = []
        for xx in x:
            v = self.polynomial_func(xx, self.monomials)
            res.append(np.dot(np.dot(v.T, self.inverse_moments_matrix), v))
        return np.array(res).reshape(-1)

    def __inv_score_samples_nquad__(self, *args):
        return 1 / self.score_samples(np.array([[*args]]))[0]

    def __inv_score_samples__(self, x):
        return 1 / self.score_samples(x)

    def estimate_neighbours_nquad(self, x, R):
        lims = [[x[i] - R, x[i] + R] for i in range(len(x))]
        return nquad(self.__inv_score_samples_nquad__, lims, opts={"epsabs": 1e-5})[0]

    def estimate_neighbours(self, x, R, N):
        estimation, err = montecarlo_integrate(self.__inv_score_samples__, x, N, len(x), R)
        return estimation, err

    def estimate_neighbours_in_grid_nquad(self, x, r, m):
        cells = itertools.product(*[range(2 ** m) for i in range(len(x))])
        start = x - (2 ** m - 1) * r
        counts = []
        for cell in cells:
            center = start + 2 * np.array(cell) * r
            lims = [[center[i] - r, center[i] + r] for i in range(len(center))]
            counts.append(nquad(self.__inv_score_samples_nquad__, lims)[0])
        return np.array(counts)

    def estimate_neighbours_in_grid(self, x, r, m, N):
        cells = itertools.product(*[range(2 ** m) for i in range(len(x))])
        start = x - (2 ** m - 1) * r
        counts = []
        for cell in cells:
            center = start + 2 * np.array(cell) * r
            estimation, err = montecarlo_integrate(self.__inv_score_samples__, center, N, len(center), r)
            counts.append(estimation)
        return np.array(counts)

    def update(self, x, n):
        self.incr_func(self, x, n, self.inv_opt)
        return self

    def save_model(self):
        serializable_monomials = None
        serializable_moments_matrix = self.moments_matrix.tolist()
        serializable_inverse_moments_matrix = self.inverse_moments_matrix.tolist()
        return {
            "monomials": serializable_monomials,
            "moments_matrix": serializable_moments_matrix,
            "inverse_moments_matrix": serializable_inverse_moments_matrix,
        }

    def load_model(self, model_dict):
        pass

    def learned(self):
        return self.inverse_moments_matrix is not None

    def copy(self):
        mm_bis = MomentsMatrix(d=self.d)
        mm_bis.polynomial_func = self.polynomial_func
        mm_bis.incr_func = self.incr_func
        mm_bis.monomials = self.monomials
        mm_bis.moments_matrix = self.moments_matrix
        mm_bis.inverse_moments_matrix = self.inverse_moments_matrix
        return mm_bis


""" DyCF: Polynomials basis """


class PolynomialsBasis:
    @staticmethod
    def generate_combinations(n, p):
        it = itertools.product(range(n + 1), repeat=p)
        mono = [i for i in it if np.sum(list(i)) <= n]
        return sorted(mono, key=lambda e: (np.sum(list(e)), list(-1 * np.array(list(e)))))

    @staticmethod
    def apply_combinations(x, m, basis_func):
        assert type(m) == list
        result = basis_func(x, m)
        return np.product(result, axis=1).reshape(-1, 1)


""" DBOECF & MDEFECF: Calcul dynamique de R """


def update_params(old_mean, old_std, new_point, N):
    new_mean = ((N - 1) / N) * old_mean + (1 / N) * new_point
    new_std = ((N - 1) / N) * (old_std + (1 / N) * np.square(new_point - old_mean))
    return new_mean, new_std


def compute_R(std, N, p):
    ev = np.sqrt(5) * np.maximum(std, 1e-32 * np.ones(p)) / np.power(N, 1 / (p + 4))
    R = 0.5 * np.linalg.norm(ev) / np.sqrt(p)
    return R


""" Toolbox: Useful methods for comparison """


def compute_and_save(methods, data, savename, multi_processing: bool = False, threads: int = None, inf_replacement: float = 1e9):
    if not multi_processing:
        for method in methods:
            for k, v in data.items():
                multiprocessable_compute_method(method, savename + '_' + k, inf_replacement=inf_replacement, **v)
    else:  # if time and memory performances are not important, multi_processing can be used to get the results faster
        params = []
        for k, v in data.items():
            params.extend([(method, savename + '_' + k, v["x_train"], v["x_test"], v["y_test"]) for method in methods])
        if threads is None:
            pool = Pool()
        else:
            pool = Pool(threads)
        pool.starmap(multiprocessable_compute_method, params)
        pool.close()


def multiprocessable_compute_method(method, savename, x_train, x_test, y_test, inf_replacement: float = 1e-9):
    if type(method) == dict:
        filename = f"{savename}__{method['short_name']}.pickle"
        try:
            with open(filename, "rb") as f:
                pickle.load(f)
        except FileNotFoundError:
            start = time.time()
            model = method["method"](**method["params"])
            model.fit(x_train)
            fit_time = time.time() - start
            y_decision = np.zeros(len(y_test))
            y_pred = np.zeros(len(y_test))
            start = time.time()
            for i in tqdm(range(len(x_test)), desc=method["name"]):
                y_decision[i] = model.eval_update(x_test[i].reshape(1, -1))
                y_pred[i] = -1 if y_decision[i] < 0 else 1
            eval_time = time.time() - start
            results = {
                "y_decision": y_decision,  # saved in order to be able to compute other metrics if needed
                "y_pred": y_pred,  # same here
                "model_size": asizeof.asizeof(model),
                "model_fit_time": fit_time,  # saved but not used
                "model_eval_time": eval_time,
            }
            y_decision[y_decision == -np.inf] = - inf_replacement  # the used metrics do not work with inf values that can be obtained with iLOF
            y_decision[y_decision == np.inf] = inf_replacement
            results["model_auroc"] = roc_auc_score(y_test[~np.isnan(y_decision)], y_decision[~np.isnan(y_decision)])
            results["model_ap"] = average_precision_score(y_test[~np.isnan(y_decision)], y_decision[~np.isnan(y_decision)])
            with open(filename, "wb") as f:
                pickle.dump(results, f)


def plot_and_save_results(methods, data, savename):
    fig, ax = plt.subplots(4, 1, figsize=(8, 12), sharex=True)
    ax_auroc = ax[0]
    ax_ap = ax[1]
    ax_duration = ax[2]
    ax_size = ax[3]

    for k, v in data.items():
        cols = ["AUROC", "AP", "Duration (s/it)", "Size (bytes)"]
        res = {
            "auroc": [],
            "ap": [],
            "duration": [],
            "size": [],
            "method": [],
            "dataset": [],
        }
        for method in methods:
            filename = f"{savename}_{k}__{method['short_name']}.pickle"
            with open(filename, "rb") as f:
                results = pickle.load(f)
            y_test = v["y_test"]
            res["auroc"].append(results["model_auroc"])
            res["ap"].append(results["model_ap"])
            res["duration"].append(results["model_eval_time"] / len(y_test))
            res["size"].append(results["model_size"])
            res["method"].append(method["short_name"])
            res["dataset"].append(k)
        table = np.array([res["auroc"], res["ap"], res["duration"], res["size"]]).T
        rows = res["method"]
        df = pd.DataFrame(data=table, columns=cols, index=rows)
        df.to_csv(f"{savename}_{k}.csv")
        print(f"Dataset {k} over.")

        ax_auroc.plot(res["auroc"], label=f"{k}")
        ax_auroc.scatter(np.argmax(res["auroc"]), np.max(res["auroc"]), marker='o', s=100, alpha=1)

        ax_ap.plot(res["ap"], label=f"{k}")
        ax_ap.scatter(np.argmax(res["ap"]), np.max(res["ap"]), marker='o', s=100, alpha=1)

        ax_duration.plot(res["duration"], label=f"{k}")
        ax_duration.scatter(np.argmin(res["duration"]), np.min(res["duration"]), marker='o', s=100, alpha=1)

        ax_size.plot(res["size"], label=f"{k}")
        ax_size.scatter(np.argmin(res["size"]), np.min(res["size"]), marker='o', s=100, alpha=1)

    names = [method["short_name"] for method in methods]

    ax_auroc.set_xticks(np.arange(len(methods)))
    ax_auroc.set_ylabel("(a) AUROC score")
    ax_auroc.tick_params(top=True, labeltop=True, axis="x")
    ax_auroc.set_xticklabels(names, rotation=60, ha="left", rotation_mode="anchor")
    ax_auroc_ = ax_auroc.twinx()
    ax_auroc_.set_yticks(np.arange(2))
    ax_auroc_.set_yticklabels(["worse", "better"])
    ax_auroc.legend()
    ax_auroc.grid()

    ax_ap.set_xticks(np.arange(len(methods)))
    ax_ap.set_ylabel("(b) AP score")
    ax_ap_ = ax_ap.twinx()
    ax_ap_.set_yticks(np.arange(2))
    ax_ap_.set_yticklabels(["worse", "better"])
    ax_ap.legend()
    ax_ap.grid()

    ax_duration.set_xticks(np.arange(len(methods)))
    ax_duration.set_ylabel("(c) Duration (seconds per iteration)")
    ax_duration.semilogy()
    ax_duration_ = ax_duration.twinx()
    ax_duration_.set_yticks(np.arange(2))
    ax_duration_.set_yticklabels(["better", "worse"])
    ax_duration.legend()
    ax_duration.grid()

    ax_size.set_xticks(np.arange(len(methods)))
    ax_size.set_ylabel("(d) Model size in memory (bytes)")
    ax_size.semilogy()
    ax_size.set_xticklabels(names, rotation=60, ha="right", rotation_mode="anchor")
    ax_size_ = ax_size.twinx()
    ax_size_.set_yticks(np.arange(2))
    ax_size_.set_yticklabels(["better", "worse"])
    ax_size.legend()
    ax_size.grid()

    fig.savefig(f"{savename}.png")


""" Toolbox: Other useful methods """


def load_dataset(filename):
    data = pd.read_csv(filename, index_col=0).values
    vmin = np.min(data, axis=0)[:-1]
    vmax = np.max(data, axis=0)[:-1]
    data[:, :-1] = 2 * (((data[:, :-1] - vmin) / (vmax - vmin)) - .5)
    return data


def split_data(data, split_pos):
    train = data[:split_pos]
    test = data[split_pos:]
    return train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]


def montecarlo_integrate(f, x, N, p, R):
    samples = x.reshape(1, p) + R * np.random.uniform(-1, 1, (N, p))
    estimates = f(samples)
    estimates_2 = np.square(estimates)
    volume = (2 * R) ** p
    mean = np.sum(estimates) * volume / N
    variance = (np.sum(estimates_2) * volume * volume / N) - mean ** 2
    if variance < 0:
        if abs(variance) < 1e-30:
            variance = 0
        else:
            raise ValueError("Variance is negative.")
    return mean, 1.96 * np.sqrt(variance / N)


""" Toolbox : Comparison metrics """


def average_precision_score(y_true, y_score):
    score_outliers = y_score[y_true == -1]
    precision_scores = [len(score_outliers[score_outliers < threshold]) / len(y_score[y_score < threshold]) if len(y_score[y_score < threshold]) != 0 else 1 for threshold in score_outliers]
    return np.mean(precision_scores)


def roc_auc_score(y_true, y_score):
    tpr, fpr, thresholds = roc_curve(y_true, y_score)
    return trapezoid(tpr, fpr)


def roc_curve(y_true, y_score):
    score_outliers = y_score[y_true == -1]
    score_inliers = y_score[y_true != -1]
    thresholds = np.concatenate(([np.min(y_score)], np.unique(score_outliers), [np.max(y_score)]))
    tpr = np.array([len(score_outliers[score_outliers < s]) / len(score_outliers) for s in thresholds])
    fpr = np.array([len(score_inliers[score_inliers < s]) / len(score_inliers) for s in thresholds])
    return tpr, fpr, thresholds


def roc_and_pr_curve(y_true, y_score):
    score_outliers = y_score[y_true == -1]
    score_inliers = y_score[y_true != -1]
    thresholds = np.concatenate(([np.min(y_score)], np.unique(score_outliers), [np.max(y_score)]))
    tpr = np.array([len(score_outliers[score_outliers < s]) / len(score_outliers) for s in thresholds])
    fpr = np.array([len(score_inliers[score_inliers < s]) / len(score_inliers) for s in thresholds])
    prec = np.array([len(score_outliers[score_outliers < s]) / len(y_score[y_score < s]) if len(y_score[y_score < s]) != 0 else np.nan for s in thresholds])
    return tpr, fpr, prec, thresholds


def supervised_metrics(y_true, y_pred):
    pred_outliers = y_pred[y_true == -1]
    pred_inliers = y_pred[y_true == 1]
    recall = len(pred_outliers[pred_outliers == -1]) / len(pred_outliers)  # TPR
    specificity = len(pred_inliers[pred_inliers == 1]) / len(pred_inliers)  # 1 - FPR
    precision = len(pred_outliers[pred_outliers == -1]) / len(y_pred[y_pred == -1]) if len(y_pred[y_pred == -1]) != 0 else np.nan
    accuracy = (recall + specificity) / 2
    f_score = 2 * recall * precision / (recall + precision) if recall + precision != 0 else 0
    return recall, specificity, precision, accuracy, f_score
