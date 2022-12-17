from itertools import product
from typing import Any, cast

import numpy as np
from numpy.random import Generator, default_rng
from sklearn.svm import SVC

from utils import gaussian_kernel, get_random_hyperparameters
from utils.classifier import ClassificationModel
from utils.optimize import golden_section_search, gradient, incremental_search, project
from utils.plotter import Plotter


class BayesianOptimizer:
    """A class that represents the Bayesian optimizer for hyperparameter tunning. (This
    class assumes that all hyperparameters are continuous.)

    Attributes
    ----------
    hyperparameter_names: list[str]
        The list of hyperparameter names.
    intervals: numpy.ndarray
        The array of dimensions `n_hyperparameters x 2` representing the lower and upper
        bounds for each hyperparameters.
    generator: numpy.random.Generator
        The numpy RNG.
    k_cross_val: int
        The number of folds for cross validation.
    model: utils.classifier.ClassificationModel
        The wrapping classfication model object.
    previous_hyperparameters: numpy.ndarray
        The array of previous hyperparameters explored.
    previous_performances: numpy.ndarray
        The previous model performances.
    best_hyperparameters: numpy.ndarray
        The best-so-far combination of hyperparameters.

    (Static) Attributes
    -------------------
    _tolerance: float
        The optimization improvement tolerance for stopping criterion.
    _n_points: int
        The number of points on each dimension to evaluate the acquisition function
        (only for plotting.)
    """

    _tolerance: float = 1e-11
    _n_points: int = 150

    def __init__(
        self,
        names: list[str],
        intervals: np.ndarray,
        generator: Generator | int,
        k_cross_val: int = 5,
    ) -> None:
        self.hyperparameter_names: list[str] = names
        self.intervals: np.ndarray = intervals
        self.generator: Generator = default_rng(generator)
        self.k_cross_val: int = k_cross_val

        self.model: ClassificationModel = ClassificationModel(
            SVC(kernel="rbf"), k_cross_val
        )

        n = self.intervals.shape[0]
        self.previous_hyperparameters: np.ndarray = np.zeros((0, n))
        self.previous_performances: np.ndarray = np.zeros((0, 1))
        self.best_hyperparameters: np.ndarray = np.zeros(n)
        self.best_performance: float = 0

        self._recalculate_parameters: bool = True
        self._matrix_sigma: np.ndarray = np.zeros((n, n))
        self._inv_matrix_sigma: np.ndarray = np.zeros((n, n))

    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_iter: int = 100,
        n_init: int = 5,
        n_contours: int = 5,
    ) -> None:
        """It performs Bayesian optimization to find the best hyperparameters.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The data labels.
        n_iter: int, optional
            The number of iterations to run the algorithm for. Default: 100.
        n_init: int, optional
            The initial number of sample. Default: 5.
        n_contour: int, optional
            The number of contours and paths to save. Default: 5.
        """

        # Get random hyperparameters
        for i in range(n_init):
            hyperparams = get_random_hyperparameters(self.intervals, self.generator)

            # Get and store initial performance
            performance = self.model.eval(X, y, hyperparams, self.hyperparameter_names)
            self._update_previous_info(hyperparams, performance)

        # Generate random indices
        chosen_iterations = []
        if n_contours > 0:
            iterations = np.arange(n_iter)
            chosen_iterations = self.generator.choice(
                iterations, size=n_contours, replace=False
            ).tolist()

        # Iterate Bayesian optimization
        i = 0
        while i < n_iter:
            print("B.O. iteration", i)
            # Maximize acquisition
            draw_iteration = i in chosen_iterations
            new_hyperparameters = self._maximize_acquisition(draw_iteration, i)

            # Evaluate performance
            new_performance = self.model.eval(
                X, y, new_hyperparameters, self.hyperparameter_names
            )

            # Update observations
            self._update_previous_info(new_hyperparameters, new_performance)
            i += 1

        idx = self.previous_performances.argmax()
        self.best_performance = self.previous_performances[idx, 0]
        self.best_hyperparameters = self.previous_hyperparameters[idx, :]

    def _maximize_acquisition(self, draw_iteration: bool, iteration: int) -> np.ndarray:
        """Returns the hyperparameters that (locally) maximize the acquisition function.

        Parameters
        ----------
        draw_iteration: bool
            If the maximization procedure should be stored.
        iteration: int
            The iteration of the BO process.

        Returns
        -------
        numpy.ndarray
            The new values for the hyperparameters.
        """

        # Declare gradient
        gradient_acq = lambda x: gradient(self.acquisition_function, x)

        # Randomly choose initial guess
        h = get_random_hyperparameters(self.intervals, self.generator)

        # Gradient descent on the acquisition function
        improvement = np.inf
        prev_val = self.acquisition_function(h)
        path = np.zeros((1, h.shape[0]))
        path[0, :] = h

        while improvement > type(self)._tolerance:
            # Objective function for line search
            grad = gradient_acq(h)
            norm = np.linalg.norm(grad)
            norm = norm if norm > 0 else 1
            update_h = lambda l: h + l * grad / norm
            auxiliary_of = lambda l: self.acquisition_function(update_h(l))

            # Find maximum possible step
            minimum_distance = np.inf
            for i in range(len(self.hyperparameter_names)):
                d = np.abs(self.intervals[i, :] - h[i]).min()
                if d < minimum_distance:
                    minimum_distance = d

            # Optimize for step size
            bounds = incremental_search(auxiliary_of, minimum_distance)
            bounds = golden_section_search(auxiliary_of, bounds)
            l = sum(bounds) / 2

            # Estimate improvement and project if necessary
            h = project(update_h(l), self.intervals)
            current_val = self.acquisition_function(h)
            improvement = np.abs(prev_val - current_val)

            path = np.vstack((path, h.reshape((1, h.shape[0]))))
            prev_val = current_val

        # Draw current iteration if needed
        if draw_iteration:
            # Get meshgrid
            n_hyperparams = len(self.hyperparameter_names)
            matrix = np.zeros((n_hyperparams, self._n_points))
            indices_matrix = np.zeros((n_hyperparams, self._n_points), dtype=int)
            for i in range(n_hyperparams):
                lower = self.intervals[i, 0]
                upper = self.intervals[i, 1]
                matrix[i, :] = np.linspace(lower, upper, num=self._n_points)
                indices_matrix[i, :] = np.linspace(
                    0, self._n_points - 1, num=self._n_points, dtype=int
                )
            coordinates = list(product(*matrix.tolist()))
            indices = list(product(*indices_matrix.tolist()))

            # Evaluate acquisition function
            acquisition = np.zeros([self._n_points] * n_hyperparams)
            for i, hyperparameters in enumerate((coordinates)):
                aux_h = np.array(hyperparameters)
                acquisition[indices[i]] = self.acquisition_function(aux_h)

            # Construct plot
            Plotter.store_iteration(
                matrix[0, :], matrix[1, :], acquisition, path, iteration
            )

        return h

    def _update_previous_info(
        self, hyperparameters: np.ndarray, performance: float
    ) -> None:
        """It updates the stored hyperparameters and values.

        Parameters
        ----------
        hyperparameters: numpy.ndarray
            The new hyperparameters to add.
        performance: numpy.ndarray
            The new performance achieved.
        """

        self.previous_hyperparameters = np.vstack(
            (self.previous_hyperparameters, hyperparameters)
        )
        self.previous_performances = np.vstack(
            (self.previous_performances, performance)
        )
        self._recalculate_parameters = True

    def acquisition_function(self, hyperparameters: np.ndarray) -> float:
        """It returns the lower confidence bound acquisition function.

        Parameters
        ----------
        hyperparameters
            The hyperparameters to evalaute the acquisition function.

        Returns
        -------
        float
            The acquisition function value.
        """

        mean, sigma = self.get_gaussian_params(hyperparameters)
        return mean + 2 * sigma

    def get_gaussian_params(self, hyperparameters: np.ndarray) -> tuple[float, float]:
        """It returns the parameters of the conditional Gaussian distribution for
        Bayesian optimization.

        Parameters
        ----------
        hyperparameters: numpy.ndarray
            The current hyperparameters.

        Returns
        -------
        float
            The mean of the Gaussian distribution.
        float
            The standard deviation of the Gaussian distribution.
        """

        if self._recalculate_parameters:
            n = self.previous_hyperparameters.shape[0]
            self._matrix_sigma = np.zeros((n, n))
            for i in range(n):
                self._matrix_sigma[i, :] = gaussian_kernel(
                    self.previous_hyperparameters[i, :], self.previous_hyperparameters
                )
            self._inv_matrix_sigma = np.linalg.inv(self._matrix_sigma)

        # Vector k and transform to column
        vector_k = gaussian_kernel(hyperparameters, self.previous_hyperparameters)[
            ..., None
        ]

        cov_k = gaussian_kernel(hyperparameters, hyperparameters[None, ...])
        aux_vector = vector_k.T @ self._inv_matrix_sigma

        self._mu = cast(float, aux_vector @ self.previous_performances)
        self._sigma = cast(float, np.sqrt(cov_k - aux_vector @ vector_k))
        self._recalculate_parameters = False

        return self._mu, self._sigma

    def get_json(self) -> dict[str, Any]:
        """It returns a dictionary for JSON output formatting.

        Returns
        -------
        dict[str, Any]
            The dictionary for JSON outputs.
        """

        d = {}
        d["best_performance"] = self.best_performance
        d["best_hyperparameters"] = self.best_hyperparameters.tolist()
        d["all_performance"] = self.previous_performances.tolist()
        d["all_hyperparameters"] = self.previous_hyperparameters.tolist()
        return d
