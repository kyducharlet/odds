import matplotlib.pyplot as plt
import numpy as np
from .base import BaseDetector, BasePlotter
from .distance import OSMCOD


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
        assert model.p == 2
        self.model = model

    def plot(self, x, n_x1=100, n_x2=100, show=False, save=False, save_title="fig.png", close=True):
        assert x.shape[1] == self.model.p
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
        # levels = np.exp(np.linspace(np.log(np.min(X3)), np.log(np.max(X3)), 11))
        percentiles = [25, 50, 75, 90, 95, 98, 99, 99.9]
        levels = [np.percentile(X3, p) for p in percentiles]
        cs = ax.contour(X1, X2, X3, levels=list(levels[1:-1]))
        ax.clabel(cs, inline=1)
        if save:
            plt.savefig(save_title)
        if show:
            plt.show()
        if close:
            plt.close()
        else:
            return fig, ax


class MTreePlotter(BasePlotter):
    def __init__(self, model: OSMCOD):
        assert model.p == 2
        self.model = model

    def plot(self, x, n_x1=100, n_x2=100, show=False, save=False, save_title="fig.png", close=True):
        assert x.shape[1] == self.model.p
        fig, ax = plt.subplots()
        ax.scatter(x[:, 0], x[:, 1], marker='x', s=20)
        x1_margin = (np.max(x[:, 0]) - np.min(x[:, 0])) / 10
        x2_margin = (np.max(x[:, 1]) - np.min(x[:, 1])) / 10
        ax.set_xlim([np.min(x[:, 0]) - x1_margin, np.max(x[:, 0]) + x1_margin])
        ax.set_ylim([np.min(x[:, 1]) - x2_margin, np.max(x[:, 1]) + x2_margin])
        self.__plot_radius__(self.model.mtmc, ax)
        if save:
            plt.savefig(save_title)
        if show:
            plt.show()
        if close:
            plt.close()
        else:
            return fig, ax

    def __plot_radius__(self, node, ax):
        if node.node_type != "leaf":
            c = plt.Circle((node.center.values[0], node.center.values[1]), node.radius, color='blue', fill=False)
            ax.add_patch(c)
            for child in node.children.keys():
                self.__plot_radius__(child, ax)
        else:
            c = plt.Circle((node.center.values[0], node.center.values[1]), node.radius, color='blue', fill=False)
            ax.add_patch(c)
            for child in node.children.keys():
                c_ = plt.Circle((child.values[0], child.values[1]), self.model.R / 2, color='green', fill=False)
                ax.add_patch(c_)
