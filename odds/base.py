from abc import ABC, abstractmethod


class BaseDetector(ABC):
    """
    Base class for outlier detection methods

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
    """

    @abstractmethod
    def fit(self, x):
        pass

    @abstractmethod
    def update(self, x):
        pass

    @abstractmethod
    def score_samples(self, x):
        pass

    @abstractmethod
    def decision_function(self, x):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def eval_update(self, x):
        pass

    @abstractmethod
    def predict_update(self, x):
        pass

    @abstractmethod
    def copy(self):
        pass

    @abstractmethod
    def method_name(self):
        pass

    def assert_shape_unfitted(self, x):
        if len(x.shape) != 2:
            raise ValueError("The expected array shape: (N, p) do not match the given array shape: {}".format(x.shape))

    def assert_shape_fitted(self, x):
        if self.__dict__.get("p") is None:
            raise NotFittedError(f"This {self.method_name()} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        elif len(x.shape) != 2 or x.shape[1] != self.__dict__["p"]:
            raise ValueError("The expected array shape: (N, {}) do not match the given array shape: {}".format(self.__dict__["p"], x.shape))


class BasePlotter(ABC):
    """
        Base class for outlier detection results plotters. This requires that data is 2-dimensional.

        Methods
        -------
        plot(x, res)
            Generates the wanted plot and optionally show, save and close.
        """

    @abstractmethod
    def plot(self, x, n_x1=500, n_x2=500, show=False, save=False, save_title="fig.png", close=True):
        pass


class NotFittedError(Exception):
    pass
