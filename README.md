# Outlier Detection for Data Streams (ODDS)

## Description

This is a Python framework for outlier detection in data streams.

## How to install

With Python 3.11 installed, you can either run the following command within the repository folder, which can be useful to modify some parts:

```shell 
pip install -e .
```

or install from git:

```shell
pip install git+https://gitlab.forge.berger-levrault.com/bl-drit/bl.drit.experiments/machine.learning/odds.git@bl-predict
```

The best way is to use personal access tokens (see: https://docs.readthedocs.io/en/stable/guides/private-python-packages.html#gitlab) and:

```shell
pip install git+https://${GITLAB_TOKEN_USER}:${GITLAB_TOKEN}@gitlab.forge.berger-levrault.com/bl-drit/bl.drit.experiments/machine.learning/odds.git@bl-predict
```

## How to use as a framework

If you want to use this repository as a framework, note that you can use newer version of Python or other libraries, changing the requirements in ```setup.py```.

### Implemented methods

Here we describe the different methods already implemented in this framework and their parameters.

#### Statistical methods

##### KDE

Kernel Density Estimation with sliding windows

    Attributes
    ----------
    threshold: float
        the threshold on the pdf, if the pdf computed for a point is greater than the threshold then the point is considered normal
    win_size: int
        size of the window of kernel centers to keep in memory
    kernel: str, optional
        the type of kernel to use, can be either "gaussian" or "epanechnikov" (default is "gaussian")
    bandwidth: str, optional
        rule of thumb to compute the bandwidth, "scott" is the only one implemented for now (default is "scott")

##### SmartSifter

Smart Sifter reduced to continuous domain only with its Sequentially Discounting Expectation and Maximizing (SDEM) algorithm
(see https://github.com/sk1010k/SmartSifter and https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)

    Attributes
    ----------
    threshold: float
        the threshold on the pdf, if the pdf computed for a point is greater than the threshold then the point is considered normal
    k: int
        number of gaussian mixture components ("n_components" from sklearn.mixture.GaussianMixture)
    r: float
        discounting parameter for the SDEM algorithm ("r" from smartsifter.SDEM)
    alpha: float
        stability parameter for the weights of gaussian mixture components ("alpha" from smartsifter.SDEM)
    scoring_function: str
        scoring function used, either "logloss" for logarithmic loss or "hellinger" for hellinger score, both proposed by the original article,
        or "likelihood" for the likelihood that a point is issued from the learned mixture (default is "likelihood")

##### DyCF

Dynamical Christoffel Function

    Attributes
    ----------
    d: int
        the degree of polynomials, usually set between 2 and 8
    incr_opt: str, optional
        can be either "inverse" to inverse the moments matrix each iteration or "sherman" to use the Sherman-Morrison formula (default is "inv")
    polynomial_basis: str, optional
        polynomial basis used to compute moment matrix, either "monomials", "chebyshev_t_1", "chebyshev_t_2", "chebyshev_u" or "legendre", 
        varying this parameter can bring stability to the score in some cases (default is "monomials")
    regularization: str, optional
        one of "vu" (score divided by d^{3p/2}), "vu_C" (score divided by d^{3p/2}/C), "comb" (score divided by comb(p+d, d)) or "none" (no regularization), "none" is used for cf vs mkde comparison (default is "vu_C")
    C: float, optional
        define a threshold on the score when used with regularization="vu_C", usually C<=1 (default is 1)
    inv: str, optional
        inversion method, one of "inv" for classical matrix inversion or "pinv" for Moore-Penrose pseudo-inversion (default is "inv")

##### DyCG

Dynamical Christoffel Growth

    Attributes
    ----------
    degrees: ndarray, optional
        the degrees of at least two DyCF models inside (default is np.array([2, 8]))
    dycf_kwargs:
        see DyCF args others than d

#### Distance-based methods

##### DBOKDE

Distance-Based Outliers by Kernel Density Estimation

    Attributes
    ----------
    k: int
        a threshold on the number of neighbours needed to consider the point as normal
    R: float or str
        the distance defining the neighborhood around a point, can be computed dynamically, in this case set R="dynamic"
    win_size: int
        the number of points in the sliding window used in neighbours count
    sample_size: int
        the number of points used as kernel centers for the KDE, if sample_size=-1 then sample_size is set to win_size (default is -1)

##### DBOECF

Distance-Based Outliers by Empirical Christoffel Function

    Attributes
    ----------
    threshold: float
        a threshold on the "unknownly ratioed" number of neighbours needed to consider the point as normal
    R: float
        the distance defining the neighborhood around a point
    d: int
        the degree for the ECF
    N_sample: int, optional
        number of samples used to estimate the Christoffel function's integral (default is 100)
    incr_opt: str, optional
        whether "inverse" to inverse the moments matrix each iteration or "sherman" to use the Sherman-Morrison formula (default is "inv")
    polynomial_basis: str, optional
        polynomial basis used to compute moment matrix, either "monomials", "chebyshev_t_1", "chebyshev_t_2", "chebyshev_u" or "legendre", 
        varying this parameter can bring stability to the score in some cases (default is "monomials")

#### Density-based methods

##### MDEFKDE

Multi-granularity DEviation Factor estimated with Kernel Density Estimation

    Attributes
    ----------
    k_sigma: float
        a threshold on the MDEF score: MDEF > k_sigma * sigmaMDEF (standard deviation of local MDEF) is an outlier (3 is often used)
    R: float
        the distance defining the neighborhood around a point
    alpha: int (alpha>1)
        a parameter for the second radius of neighborhood, defined as a 1/(2**alpha) * R neighborhood
    win_size: int
        the number of points in the sliding window used in neighbours count
    sample_size: int, optional
        the number of points used as kernel centers for the KDE, if sample_size=-1 then sample_size is set to win_size (default is -1)

##### MDEFECF

Multi-granularity DEviation Factor estimated with Empirical Christoffel Function

    Attributes
    ----------
    k_sigma: float
        a threshold on the MDEF score: MDEF > k_sigma * sigmaMDEF (standard deviation of local MDEF) is an outlier (3 is often used)
    R: float
        the distance defining the neighborhood around a point
    alpha: int (alpha>1)
        a parameter for the second radius of neighborhood, defined as a 1/(2**alpha) * R neighborhood
    d: int
        the degree for the ECF
    incr_opt: str, optional
        whether "inverse" to inverse the moments matrix each iteration or "sherman" to use the Sherman-Morrison formula (default is "inv")
    polynomial_basis: str, optional
        polynomial basis used to compute moment matrix, either "monomials", "chebyshev_t_1", "chebyshev_t_2", "chebyshev_u" or "legendre", 
        varying this parameter can bring stability to the score in some cases (default is "monomials")

### How to use outlier detection methods

The outlier detection methods implement the BaseDetector abstract class with the following methods:

    Methods
    -------
    fit(x)
        Generates a model that fit dataset x.
    update(x)
        Updates the current model with instances in x.
    score_samples(x)
        Makes the model compute the outlier score of samples in x (the higher the value, the more outlying the sample).
    decision_function(x)
        This is similar to score_samples(x) except outliers have negative score and inliers positive ones.
    predict(x)
        Returns the sign (-1 or 1) of decision_function(x).
    eval_update(x)
        Computes decision_function of each sample and then update the model with this sample.
    predict_update(x)
        Same as eval_update(x) but returns the prediction instead of the decision function.
    method_name()
        Returns the name of the method.
    save_model()
        Returns a dict of all the model attributes allowing to save the model in bases such as mongodb.
    load_model(model_dict)
        Reload a previously saved model based on the output of the save_model() method.
    copy()
        Returns a copy of the model.

### How to compare methods on labelled datasets

Datasets need to be csv files with a first line as header, a first column as indexes, following columns as variables and a last one as labels (1 for inliers and -1 for outliers).

For instance, a dataset containing two instances:
* the vector ```[0, 0]``` being an inlier,
* the vector ```[1, 1]``` being an outlier,

could be written as:
```csv
0,x1,x2,y
0,0,0,1
1,1,1,-1
```

#### Define methods and parameterizations

Methods used for evaluation should be defined following the example in ```scripts/evaluating_methods_example.py``` in a ```METHOD``` variable as a list of dictionaries.

The dictionaries elements contain the following fields:
* name: complete name to give the method,
* method: class name,
* params: another dictionary where the fields are the parameters names and the value are their values,
* short_name: a shorter name for the method to appear in the plot and as file names for saved results.

#### Set the datasets

Datasets should be set in a ```data_dict``` dictionary.

In order to do this, it is recommended to first define the split position as ```split_pos``` of the training and testing part of the dataset as an int.

Then, load the dataset into a ```data``` variable calling ```utils.load_dataset``` method with the path to the dataset (theoretically, it should be ```../res/dataset_name.csv```).

Finally, you just need to set the dataset into ```data_dict``` as in the following example (note that each dataset in data_dict need a different name):

```python
from odds.utils import load_dataset, split_data
data = load_dataset("path/to/file.csv")
split_pos = len(data) // 2  # for instance, if you want to split the dataset in two almost equal parts
x_train, y_train, x_test, y_test = split_data(data, split_pos)

data_dict = dict()
data_dict["dataset_name"] = {
    "x_test": x_test,
    "x_train": x_train,
    "y_test": y_test,
}
```

#### Run evaluation

You just have to call ```utils.compute_and_save``` and ```utils.plot_and_save_results``` with the variables defined earlier:

```python
from odds.utils import compute_and_save, plot_and_save_results
METHODS = [
    # list of methods dicts as explained in "Define methods and parameterizations" section
]
data_dict = {
    # datasets as explained in "Set the datasets" section
}
compute_and_save(METHODS, data_dict, "/path/to/res")  # here "/path/to/res" is used as a header for the files containing pickle models
plot_and_save_results(METHODS, data_dict, "/path/to/res")  # here "/path/to/res" is used as a header for the files containing graphs and metrics
```

This will generate the resulting files in the ```/path/to``` folder. The pickle files save the results in order to avoid computing the model again each time. If you want to recompute a model, the associated pickle file has to be deleted manually. One csv file is generated for each dataset, containing the results for each method. One png file is generated, showing the results in a plot.
