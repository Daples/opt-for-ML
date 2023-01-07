import os

import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    """A class to wrap the plotting functions.

    (Static) Attributes
    -------------------
    _levels: int
        The numer of levels for a countour plot.
    _folder: str
        The folder to store the output figures.
    """

    _levels: int = 25
    _folder: str = "figs"

    @staticmethod
    def clear() -> None:
        """It clears the graphic objects."""

        plt.cla()
        plt.clf()

    @staticmethod
    def __setup_config__() -> None:
        """It sets up the matplotlib configuration."""

    plt.rc("text", usetex=True)
    plt.rcParams.update({"font.size": 18})

    @classmethod
    def _add_folder(cls, path: str) -> str:
        """It adds the default folder to the input path.

        Parameters
        ----------
        path: str
            A path in string.

        Returns
        -------
        str
            The path with the added folder.
        """

        return os.path.join(cls._folder, path)

    @classmethod
    def store_iteration(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        matrix: np.ndarray,
        iteration_path: np.ndarray,
        figure_name: str,
        xlabel: str = "$\log_{10}C$",
        ylabel: str = "$\log_{10}\gamma$",
        xlim: list[float] = [0, 9],
        ylim: list[float] = [-10, 0],
    ) -> None:
        """It stores the contour of an acquisition function maximization.

        Parameters
        ----------
        x: numpy.ndarray
            The domain on the first dimension.
        y: numpy.ndarray
            The domain on the second dimesion.
        matrix: numpy.ndarray
            The surface values.
        iteration_path: numpy.ndarray
            The optimization path of points.
        figure_name: str
            The name label for the figure.
        xlabel: str, optional
            The label for the x-axis. Default: "$C$".
        ylabel: str, optional
            The label for the y-axis. Default: "$\gamma$"
        xlim: list[float], optional
            The x-axis limits. Default: [0, 9]
        ylim: list[float], optional
            The y-axis limits. Default: [-10, 0]
        """

        cls.clear()
        cls.__setup_config__()

        plt.contourf(x, y, matrix, cls._levels, zorder=-1)
        plt.colorbar()

        plt.plot(
            iteration_path[:, 0], iteration_path[:, 1], "r-", linewidth=1, zorder=1
        )
        plt.scatter(iteration_path[:, 0], iteration_path[:, 1], c="k", s=3, zorder=3)
        plt.scatter(
            iteration_path[-1, 0], iteration_path[-1, 1], c="magenta", s=4, zorder=3
        )
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(xlim)
        plt.ylim(ylim)

        path = f"bo_iter_{figure_name}.pdf"
        plt.savefig(cls._add_folder(path), bbox_inches="tight")
        cls.clear()

    @classmethod
    def get_heatmatp(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        matrix: np.ndarray,
        path: str,
        xlabel: str = "$\log_{10}C$",
        ylabel: str = "$\log_{10}\gamma$",
        xlim: list[float] = [0, 9],
        ylim: list[float] = [-10, 0],
    ) -> None:
        """It saves the heatmap of a matrix in the specified path.

        Parameters
        ----------
        x: numpy.ndarray
            The domain on the first dimension.
        y: numpy.ndarray
            The domain on the second dimension.
        matrix: numpy.ndarray
            The input matrix.
        path: str
            The filename to save the heatmap.
        xlabel: str, optional
            The label for the x-axis. Default: "$C$".
        ylabel: str, optional
            The label for the y-axis. Default: "$\gamma$"
        xlim: list[float], optional
            The x-axis limits. Default: [0, 9]
        ylim: list[float], optional
            The y-axis limits. Default: [-10, 0]
        """

        cls.clear()
        cls.__setup_config__()

        extent = [x[0], x[-1], y[0], y[-1]]
        fig, axs = plt.subplots(1, 1)
        heatmap = axs.imshow(
            matrix.T,
            cmap="viridis",
            interpolation="nearest",
            origin="lower",
            extent=extent,
        )
        fig.colorbar(heatmap)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.savefig(cls._add_folder(path), bbox_inches="tight")

    @classmethod
    def get_scatter(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        path: str,
        xlabel: str = "$\log_{10}C$",
        ylabel: str = "$\log_{10}\gamma$",
        xlim: list[float] = [0, 9],
        ylim: list[float] = [-10, 0],
    ) -> None:
        """It plots the scatter plot for the explored points.

        Parameters
        ----------
        x: numpy.ndarray
            The first dimension array.
        y: numpy.ndarray
            The second dimension array.
        path: str
            The path to save the figure.
        xlabel: str, optional
            The label for the x-axis. Default: "$C$".
        ylabel: str, optional
            The label for the y-axis. Default: "$\gamma$"
        xlim: list[float], optional
            The x-axis limits. Default: [0, 9]
        ylim: list[float], optional
            The y-axis limits. Default: [-10, 0]
        """

        cls.clear()
        cls.__setup_config__()

        plt.scatter(x, y, c="b", s=8)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.savefig(cls._add_folder(path), bbox_inches="tight")

    @classmethod
    def get_misclassification_plot(
        cls,
        performances: np.ndarray,
        performance_grid_search: float,
        path: str,
        xlabel: str = "Iteration",
        ylabel: str = "Missclassification error",
    ) -> None:
        """It plots the best-so-far misclassification error between iterations.

        Parameters
        ----------
        performances: numpy.ndarray
            The performances per iteration.
        performance_grid_search: float
            The best performance found from grid search.
        path: str
            The path to save the figure.
        xlabel: str, optional
            The label for the x-axis. Default: "Iteration".
        ylabel: str, optional
            The label for the y-axis. Default: "$Misclassfication error"
        """

        cls.clear()
        cls.__setup_config__()

        performances = np.maximum.accumulate(performances)
        plt.plot(1 - performances, "ko", label="BO performance", markersize=2)
        plt.axhline(
            y=1 - performance_grid_search,
            color="r",
            linestyle="--",
            label="Grid search best",
        )
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.ylim([0, 1])
        plt.legend()
        plt.grid()
        plt.savefig(cls._add_folder(path), bbox_inches="tight")

    @classmethod
    def get_times_plot(
        cls,
        times: list[float],
        path: str,
        xlabel: str = "Iteration",
        ylabel: str = "Execution time (s)",
    ) -> None:
        """It plots the best-so-far misclassification error between iterations.

        Parameters
        ----------
        times: list[float]
            The execution time per iteration.
        path: str
            The path to save the figure.
        xlabel: str, optional
            The label for the x-axis. Default: "Iteration".
        ylabel: str, optional
            The label for the y-axis. Default: "$Misclassfication error"
        """

        cls.clear()
        cls.__setup_config__()

        plt.plot(times, "k-o", markersize=2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.ylim([0, 1])
        plt.grid()
        plt.savefig(cls._add_folder(path), bbox_inches="tight")
