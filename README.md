# Outlier Detection for Data Streams (ODDS)

## Description

This repository is code material for the "Leveraging the Christoffel Function for Outlier Detection in Data Streams" article.

Its primary use is for reproducibility purposes. 
However, it can also be used as a Python framework for outlier detection in data streams and comparison purposes.

## How to install

With Python 3.10 installed, you just need to be in the repository and run:

```shell 
pip install -e .
```

## How to use for reproducibility

### Section 3.4

In order to get the results from section 3.4 of the article, you have to run ```scripts/comparing_cf_mkde.py```.

This will generate two files: 
* ```scripts/cf_vs_kde.csv``` which contains the AP and AUROC metrics,
* ```scripts/cf_vs_kde.png``` which contains the graph.

### Section 5.3

In order to get the results from section 5.3 of the article, you have to run ```scripts/evaluating_methods.py```.

Optional: Because this requires the training of all the models, we have put all the results in the ```res/results``` folder. Copy the content of this folder and paste it in the ```scripts``` folder to quickly generate the resulting files.

This will generate four files:
* ```scripts/evaluating_method_tm.csv``` which contains the AP, AUROC, duration and memory results for the two_moons dataset,
* ```scripts/evaluating_method_lc.csv``` which contains the AP, AUROC, duration and memory results for the luggage_conveyor dataset,
* ```scripts/evaluating_method_http.csv``` which contains the AP, AUROC, duration and memory results for the http dataset,
* ```scripts/evaluating_method.png``` which contains the graph.

## How to use as a framework

If you want to use this repository as a framework, note that you can use newer version of Python or other libraries, changing the requirements in ```setup.py```.

### Implemented methods

Here we describe the different methods already implemented in this framework and their parameters.

#### DyCF

Dynamical Christoffel Function

    Attributes
    ----------
    d: int
        the degree of polynomials
    incr_opt: str, optional
        whether "inverse" to inverse the moments matrix each iteration or "sherman" to use the Sherman-Morrison formula (default is "inv")
    polynomial_basis: str, optional
        whether "monomials" to use the monomials basis or "chebyshev" to use the Chebyshev polynomials (default is "monomials")
    regularization: str, optional
        one of "vu" (score divided by d^{3p/2}) or "none" (no regularization), "none" is used for cf vs mkde comparison (default is "vu")

#### DyCG

Dynamical Christoffel Function

    Attributes
    ----------
    degrees: ndarray, optional
        the degrees of two DyCF models inside (default is np.array([2, 8]))
    dycf_kwargs:
        see DyCF args others than d

#### Sliding MKDE

Multivariate Kernel Density Estimation with Sliding Windows

    Attributes
    ----------
    threshold: float
        the threshold on the pdf, if the pdf at a point is greater than the threshold then the point is considered normal
    win_size: int
        size of the window of kernel centers to keep in memory
    kernel: str, optional
        the type of kernel to use (default is "gaussian")
    bandwidth: str, optional
        rule of thumb to compute the bandwidth (default is "scott")

#### OSCOD

One-Shot COD

    Attributes
    ----------
    k: int
        a threshold on the number of neighbours needed to consider the point as normal
    R: float
        the distance defining the neighborhood around a point
    win_size: int
        number of points in the sliding window used in neighbours count
    M: int (optional)
        max size of a node in the M-tree containing all points

#### iLOF

Incremental Local Outlier Factor

    Attributes
    ----------
    k: int
        the number of neighbors to compute the LOF score on
    threshold: float
        a threshold on the LOF score to separate normal points and outliers
    win_size: int
        number of points in the sliding window used in kNN search
    min_size: int (optional)
        minimal number of points in a node, it is mandatory that 2 <= min_size <= max_size / 2 (default is 3)
    max_size: int (optional)
        maximal number of points in a node, it is mandatory that 2 <= min_size <= max_size / 2 (default is 12)
    p_reinsert_tol: int (optional)
        tolerance on reinsertion, used to deal with overflow in a node (default is 4)
    reinsert_strategy: str (optional)
        either "close" or "far", tells if we try to reinsert in the closest rectangles first on in the farthest (default is "close")


### How to implement new outlier detection methods

We strongly advise that new outlier detection methods implement the BaseDetector abstract class.

The different methods of the abstract class that need to be implemented are the following:

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
    method_name(x)
        Returns the name of the method.
    copy(x) (optional)
        Returns a copy of the model.

### How to compare on new datasets

New datasets can be added in the resource folder ```res```. 

Those datasets need to be csv files with a first line as header, a first column as indexes, following columns as variables and a last one as labels (1 for inliers and -1 for outliers).

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

Methods used for evaluation should be defined in ```scripts/evaluating_methods.py``` or in a similar script in a ```METHOD``` variable as a list of dictionaries.

The dictionaries elements contain the following fields:
* name: complete name to give the method,
* method: class name,
* params: another dictionary where the fields are the parameters names and the value are their values,
* short_name: a shorter name for the method to appear in the plot and as file names for saved results.

#### Set the datasets

Datasets should be set in a ```data_dict``` dictionary.

In order to do this, it is recommended to first define the split position as ```split_pos``` of the training and testing part of the dataset as an int.

Then, load the dataset into a ```data``` variable calling ```utils.load_dataset``` method with the path to the dataset (theoretically, it should be ```../res/dataset_name.csv```).

And call 
```python
x_train, y_train, x_test, y_test = split_data(data, split_pos)
``` 
with ```utils.split_data```.

Finally, you just need to set the dataset into ```data_dict``` according to the following example:

```python
data_dict["dataset_name"] = {
    "x_test": x_test,
    "x_train": x_train,
    "y_test": y_test,
}
```

#### Run evaluation

You just have to call ```utils.compute_and_save``` and ```utils.plot_and_save_results``` with the variables defined earlier:

```python
compute_and_save(METHODS, data_dict, "evaluating_methods")
plot_and_save_results(METHODS, data_dict, "evaluating_methods")
```

This will generate the resulting files in the ```scripts``` folder. The pickle files save the results in order to avoid computing the model again each time. If you want to recompute a model, the associated pickle file has to be deleted manually. One csv file is generated for each dataset, containing the results for each method. One png file is generated, showing the results in a plot.
