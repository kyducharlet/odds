import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from .base import BaseDetector, BasePlotter
from .utils import RStarTree, MTree


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


class MTreePlotter(BasePlotter):
    """
    Plotter for an M-tree in space.

    Attributes
    ----------
    model: BaseDetector
        the model carrying the M-tree to study
    attr_name: str (optional)
        the name of the M-tree as attribute of the model (default is mt)

    Methods
    -------
    See BasePlotter methods.
    """

    def __init__(self, model: BaseDetector, attr_name: str="mt"):
        assert model.__dict__.get(attr_name) is not None and type(model.__dict__[attr_name]) == MTree  # assert that the pointed attribute is an M-tree
        self.mt = model.__dict__[attr_name]
        assert len(self.mt.points[0].values) == 2  # assert the bivariate case

    def plot(self, x, n_x1=100, n_x2=100, show=False, save=False, save_title="fig.png", close=True):
        assert x.shape[1] == 2
        fig, ax = plt.subplots()
        ax.scatter(x[:, 0], x[:, 1], marker='x', s=20)
        x1_margin = (np.max(x[:, 0]) - np.min(x[:, 0])) / 10
        x2_margin = (np.max(x[:, 1]) - np.min(x[:, 1])) / 10
        ax.set_xlim([np.min(x[:, 0]) - x1_margin, np.max(x[:, 0]) + x1_margin])
        ax.set_ylim([np.min(x[:, 1]) - x2_margin, np.max(x[:, 1]) + x2_margin])
        self.__plot_radius__(self.mt.root, ax)
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
                c_ = plt.Circle((child.values[0], child.values[1]), self.mt.R / 2, color='green', fill=False)
                ax.add_patch(c_)


class RStarTreePlotter(BasePlotter):
    """
    Plotter for an R*-tree in space.

    Attributes
    ----------
    rst: RStarTree
        the R*-tree to study

    Methods
    -------
    See BasePlotter methods.
    """

    def __init__(self, model: BaseDetector, attr_name: str="rst"):
        assert model.__dict__.get(attr_name) is not None and type(model.__dict__[attr_name]) == RStarTree  # assert that the pointed attribute is an R*-tree
        self.rst = model.__dict__[attr_name]
        assert self.rst.objects[0].high.shape[1] == 2  # assert the bivariate case

    def plot(self, x, n_x1=100, n_x2=100, show=False, save=False, save_title="fig.png", close=True):
        fig, ax = plt.subplots()
        cmap = cm.get_cmap('gist_rainbow')
        colors = cmap(np.linspace(0, 1, len(self.rst.levels)))
        self.__plot_rectangles__(self.rst.root, colors, ax)
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
        if node.level != self.rst.levels[0]:
            for child in node.children:
                self.__plot_rectangles__(child, colors, ax)
        else:
            for child in node.children:
                ax.scatter(child.low[0, 0], child.low[0, 1], marker='x', s=20, c='k', alpha=0.3)
