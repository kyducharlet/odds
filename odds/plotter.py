import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from .base import BaseDetector, BasePlotter


class LevelsetPlotter(BasePlotter):
    """
    Plotter for levelsets associated with a model.

    Attributes
    ----------
    model: BaseDetector
        the model to study

    Methods
    -------
    See BasePlotter methods.

    """

    def __init__(self, model: BaseDetector):
        assert model.__dict__.get("p") is not None
        assert model.__dict__["p"] == 2
        self.model = model

    def plot(self, x, *args, n_x1=100, n_x2=100, show=False, save=False, save_title="fig.png", close=True, **kwargs):
        assert x.shape[1] == self.model.__dict__["p"]
        fig, ax = plt.subplots()
        ax.scatter(x[:, 0], x[:, 1], marker='x', s=20)
        x1_margin = (np.max(x[:, 0]) - np.min(x[:, 0])) / 10
        x2_margin = (np.max(x[:, 1]) - np.min(x[:, 1])) / 10
        ax.set_xlim([np.min(x[:, 0]) - x1_margin, np.max(x[:, 0]) + x1_margin])
        ax.set_ylim([np.min(x[:, 1]) - x2_margin, np.max(x[:, 1]) + x2_margin])
        X1 = np.linspace(np.min(x[:, 0] - x1_margin), np.max(x[:, 0] + x1_margin), n_x1)
        X2 = np.linspace(np.min(x[:, 1]) - x2_margin, np.max(x[:, 1] + x2_margin), n_x2)
        x1, x2 = np.meshgrid(X1, X2)
        x3 = np.c_[x1.ravel(), x2.ravel()]
        X3 = self.model.score_samples(x3).reshape(x1.shape)
        percentiles = [25, 50, 75]
        if kwargs.get("percentiles") is not None:
            percentiles = kwargs["percentiles"]
        levels = [np.percentile(X3, p) for p in percentiles]
        cs = ax.contour(X1, X2, X3, levels=levels)
        ax.clabel(cs, inline=1)
        if save:
            plt.savefig(save_title)
        if show:
            plt.show()
        if close:
            plt.close()
        else:
            return fig, ax

    def plot_in_ax(self, x, ax, n_x1=100, n_x2=100, **kwargs):
        assert x.shape[1] == self.model.__dict__["p"]
        contour_kwargs = kwargs.copy()
        ax.scatter(x[:, 0], x[:, 1], marker='x', s=20, c='b')
        if kwargs.get("lims") is not None and kwargs["lims"] == "set":
            x_lim = ax.get_xlim()
            y_lim = ax.get_ylim()
            X1 = np.linspace(x_lim[0], x_lim[1], n_x1)
            X2 = np.linspace(y_lim[0], y_lim[1], n_x2)
        else:
            x1_margin = (np.max(x[:, 0]) - np.min(x[:, 0])) / 10
            x2_margin = (np.max(x[:, 1]) - np.min(x[:, 1])) / 10
            ax.set_xlim([np.min(x[:, 0]) - x1_margin, np.max(x[:, 0]) + x1_margin])
            ax.set_ylim([np.min(x[:, 1]) - x2_margin, np.max(x[:, 1]) + x2_margin])
            X1 = np.linspace(np.min(x[:, 0] - x1_margin), np.max(x[:, 0] + x1_margin), n_x1)
            X2 = np.linspace(np.min(x[:, 1]) - x2_margin, np.max(x[:, 1] + x2_margin), n_x2)
        x1, x2 = np.meshgrid(X1, X2)
        x3 = np.c_[x1.ravel(), x2.ravel()]
        X3 = self.model.score_samples(x3).reshape(x1.shape)
        if kwargs.get("levels") is not None:
            levels = kwargs["levels"]
            del contour_kwargs["levels"]
        else:
            percentiles = [25, 50, 75]
            if kwargs.get("percentiles") is not None:
                percentiles = kwargs["percentiles"]
                del contour_kwargs["percentiles"]
            levels = [np.percentile(X3, p) for p in percentiles]
        cs = ax.contour(X1, X2, X3, levels=levels, **contour_kwargs)
        ax.clabel(cs, inline=1)
