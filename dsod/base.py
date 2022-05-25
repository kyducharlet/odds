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
        Makes the model compute the score of samples in x.
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
    def copy(self):
        pass


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
