import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from .base import BaseDetector, BasePlotter
from .distance import OSMCOD
from .utils import RStarTree


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

    def plot(self, x, n_x1=100, n_x2=100, show=False, save=False, save_title="fig.png", close=True):
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


# TODO: Change MTreePlotter to take MTree in spite of OSMCOD
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


class RStarTreePlotter(BasePlotter):
    def __init__(self, structure: RStarTree):
        self.structure = structure

    def plot(self, x, n_x1=100, n_x2=100, show=False, save=False, save_title="fig.png", close=True):
        fig, ax = plt.subplots()
        cmap = cm.get_cmap('gist_rainbow')
        colors = cmap(np.linspace(0, 1, len(self.structure.levels)))
        self.__plot_rectangles__(self.structure.root, colors, ax)
        if save:
            plt.savefig(save_title)
        if show:
            plt.show()
        if close:
            plt.close()
        else:
            return fig, ax

    def __plot_rectangles__(self, node, colors, ax):
        r = plt.Rectangle((node.low[0, 0], node.low[0, 1]), node.high[0, 0] - node.low[0, 0], node.high[0, 1] - node.low[0, 1], color=colors[node.level.level], fill=False)
        ax.add_patch(r)
        if node.level != self.structure.levels[0]:
            for child in node.children:
                self.__plot_rectangles__(child, colors, ax)
        else:
            for child in node.children:
                ax.scatter(child.low[0, 0], child.low[0, 1], marker='x', s=20, c='k', alpha=0.3)
