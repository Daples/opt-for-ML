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
        iteration: int,
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
        """

        plt.contourf(x, y, matrix, cls._levels)
        plt.colorbar()

        plt.plot(iteration_path[:, 0], iteration_path[:, 1], "r-", linewidth=1)
        plt.scatter(iteration_path[:, 0], iteration_path[:, 1], c="k", s=4)
        plt.scatter(iteration_path[-1, 0], iteration_path[-1, 1], c="w", s=4)

        path = f"bo_iter_{iteration}.pdf"
        plt.savefig(cls._add_folder(path), bbox_inches="tight")
        cls.clear()

    @classmethod
    def get_heatmatp(
        cls, x: np.ndarray, y: np.ndarray, matrix: np.ndarray, path: str
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
        """

        extent = [x[0], x[-1], y[0], y[-1]]
        fig, axs = plt.subplots(1, 1)
        heatmap = axs.imshow(
            matrix,
            cmap="winter",
            interpolation="nearest",
            origin="lower",
            extent=extent,
        )
        fig.colorbar(heatmap)
        plt.savefig(cls._add_folder(path), bbox_inches="tight")
        cls.clear()
