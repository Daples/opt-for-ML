from itertools import product
from typing import Any

import numpy as np
from numpy.random import Generator, default_rng
from sklearn.svm import SVC
from tqdm import tqdm

from utils.classifier import ClassificationModel


class GridSearchOptimizer:
    """A wrapper for the grid search optimizer.

    Attributes
    ----------
    model: utils.classifier.ClassificationModel
        The wrapping classfication model object.
    intervals: numpy.ndarray
        The array of dimensions `n_hyperparameters x 2` representing the lower and upper
        bounds for each hyperparameters.
    n_points: int
        The number of partitions per dimension.
    generator: numpy.random.Generator
        The numpy RNG.
    hyperparameter_names: list[str]
        The list of hyperparameter names.
    previous_hyperparameters: numpy.ndarray
        The array of previous hyperparameters explored.
    previous_performances: numpy.ndarray
        The previous model performances.
    best_hyperparameters: numpy.ndarray
        The best-so-far combination of hyperparameters.
    best_performance: float
        The best-so-far performance.
    performace_matrix: numpy.ndarray
        The matrix that stores the performances on the grid.
    """

    def __init__(
        self,
        intervals: np.ndarray,
        N: int,
        names: list[str],
        generator: Generator | int,
        k_cross_val: int = 5,
    ) -> None:
        self.model: ClassificationModel = ClassificationModel(
            SVC(kernel="rbf"), k_cross_val
        )
        self.intervals: np.ndarray = intervals
        self.n_points: int = N
        self.generator: Generator = default_rng(generator)
        self.hyperparameter_names: list[str] = names
        self.previous_hyperparameters: np.ndarray = np.zeros(
            (0, self.intervals.shape[0])
        )
        self.previous_performances: np.ndarray = np.zeros((0, 1))
        self.best_hyperparams: np.ndarray = np.zeros(1)
        self.best_performance: float = 0
        self.performance_matrix: np.ndarray = np.zeros(1)
        self.domain_matrix: np.ndarray = np.zeros(
            (self.intervals.shape[0], self.n_points)
        )

    def optimize(self, X: np.ndarray, y: np.ndarray) -> None:
        """Use grid search to evaluate the model performance.

        Parameters
        ----------
        X: numpy.ndarray
            The input dataset.
        y: numpy.ndarray
            The data labels.
        """

        # Get meshgrid
        n_hyperparams = len(self.hyperparameter_names)
        matrix = np.zeros((n_hyperparams, self.n_points))
        indices_matrix = np.zeros((n_hyperparams, self.n_points), dtype=int)
        for i in range(n_hyperparams):
            lower = self.intervals[i, 0]
            upper = self.intervals[i, 1]
            matrix[i, :] = np.linspace(lower, upper, num=self.n_points)
            indices_matrix[i, :] = np.linspace(
                0, self.n_points - 1, num=self.n_points, dtype=int
            )
        coordinates = list(product(*matrix.tolist()))
        indices = list(product(*indices_matrix.tolist()))

        self.previous_performances = np.zeros(len(coordinates))
        self.domain_matrix = matrix

        # Evaluate performances
        best_performance = 0
        best_hyperparameters = np.zeros(n_hyperparams)
        self.performance_matrix = np.zeros([self.n_points] * n_hyperparams)

        for i, hyperparameters in enumerate(tqdm(coordinates)):
            h = np.array(hyperparameters)
            performance = self.model.eval(X, y, h, self.hyperparameter_names)
            if performance > best_performance:
                best_performance = performance
                best_hyperparameters = h
            self.previous_performances[i] = performance
            self.performance_matrix[indices[i]] = performance

        self.previous_hyperparameters = np.array(coordinates)
        self.best_hyperparams = best_hyperparameters
        self.best_performance = best_performance

    def get_json(self) -> dict[str, Any]:
        """It returns a dictionary for JSON output formatting.

        Returns
        -------
        dict[str, Any]
            The dictionary for JSON outputs.
        """

        d = {}
        d["best_performance"] = self.best_performance
        d["best_hyperparameters"] = self.best_hyperparams.tolist()
        return d
