import scipy.stats as st
import numpy as np
from typing import Any, cast
from numpy.random import Generator, default_rng

from utils import gaussian_kernel
from utils.optimize import (
    incremental_search,
    golden_section_search,
    gradient,
    backtracking,
)
from utils.classifier import ClassificationModel
from sklearn.svm import SVC


class BayesianOptimizer:
    """A class that represents the Bayesian optimizer for hyperparameter tunning.

    Attributes
    ----------
    model:
    """

    _tolerance: float = 1e-7

    def __init__(
        self,
        names: list[str],
        intervals: np.ndarray,
        generator: Generator | int,
        k_cross_val: int = 5,
    ) -> None:
        self.intervals: np.ndarray = intervals
        self.generator: Generator = default_rng(generator)
        self.k_cross_val: int = k_cross_val

        self.model: ClassificationModel = ClassificationModel(
            SVC(kernel="rbf"), k_cross_val
        )

        self.hyperparameter_names: list[str] = names
        self.previous_hyperparameters: np.ndarray = np.zeros(
            (0, self.intervals.shape[0])
        )
        self.previous_performances: np.ndarray = np.zeros((0, 1))

        # Move to properties?
        self.best_performance: float = 0
        self.best_hyperparams: np.ndarray = np.zeros(2)

    def optimize(
        self, X: np.ndarray, y: np.ndarray, n_iter: int = 100, n_init: int = 5
    ) -> None:
        """It performs Bayesian optimization to find the best hyperparameters.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The data labels.
        n_iter: int
            The number of iterations to run the algorithm for.
        n_init: int
            The initial number of sample.
        """

        # Get random hyperparameters
        for i in range(n_init):
            hyperparams = self._get_random_hyperparameters()

            # Get and store initial performance
            performance = self.model.eval(X, y, hyperparams, self.hyperparameter_names)
            self._update_previous_info(hyperparams, performance)

        # Iterate BO
        i = 0
        while i < n_iter:
            print("iter BO:", i)
            # Maximize acquisition
            new_hyperparameters = self._maximize_acquisition()

            # Evaluate performance
            new_performance = self.model.eval(
                X, y, new_hyperparameters, self.hyperparameter_names
            )

            # Update observations
            self._update_previous_info(new_hyperparameters, new_performance)

            i += 1

        idx = self.previous_performances.argmax()
        self.best_performance = self.previous_performances[idx, 0]
        self.best_hyperparams = self.previous_hyperparameters[idx, :]

    def _maximize_acquisition(self) -> np.ndarray:
        """Returns the hyperparameters that (locally) maximize the acquisition function.

        Reutrns
        -------
        The
        """

        # Declare gradient
        gradient_acq = lambda x: gradient(self.acquisition_function, x)

        # Randomly choose initial guess
        h = self._get_random_hyperparameters()

        # Gradient descent on the acquisition function
        improvement = np.inf
        prev_val = self.acquisition_function(h)
        while improvement > type(self)._tolerance:
            # Objective function for line search
            update_h = lambda l: h + l * gradient_acq(h)
            auxiliary_of = lambda l: self.acquisition_function(update_h(l))

            # Optimize for step size
            # bounds = incremental_search(auxiliary_of)
            # bounds = golden_section_search(auxiliary_of, bounds)

            # Update hyperparameters
            # h = update_h(sum(bounds) / 2)
            l = backtracking(auxiliary_of, gradient_acq, 0.8, h)

            # Estimate improvement
            current_val = self.acquisition_function(update_h(l))
            improvement = np.abs(prev_val - current_val)

            prev_val = current_val

        return h

    def _get_random_hyperparameters(self) -> np.ndarray:
        """It returns a vector of random hyperparameters chosen within the defined
        intervals.

        Returns
        -------
        numpy.ndarray
            The vector of hyperparameters.
        """

        n = self.intervals.shape[0]
        hyperparams = np.zeros(n)
        for i in range(n):
            lower = self.intervals[i, 0]
            upper = self.intervals[i, 1]
            hyperparams[i] = self.generator.uniform(lower, upper)
        return hyperparams

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

    def acquisition_function(self, hyperparameters: np.ndarray) -> float:
        """It evaluates the "probability of improving" acquisition function.

        Parameters
        ----------
        hyperparameters: numpy.ndarray
            The hyperparameters to evaluate the function in.

        Returns
        -------
        float
            The acquisition function value at the input hyperparameters.
        """

        mean, std = self.get_gaussian_params(hyperparameters)
        return cast(
            float,
            st.norm.cdf((mean - self.best_performance) / std, loc=mean, scale=std),
        )

    def get_gaussian_params(self, hyperparameters: np.ndarray) -> tuple[float, float]:
        """It returns the parameters of the conditional Gaussian distribution for
        Bayesian optimization.

        Parameters
        ----------

        """

        # Vector k and transform to column
        vector_k = gaussian_kernel(hyperparameters, self.previous_hyperparameters)[
            ..., None
        ]

        n = self.previous_hyperparameters.shape[0]
        matrix_sigma = np.zeros((n, n))
        for i in range(n):
            matrix_sigma[i, :] = gaussian_kernel(
                self.previous_hyperparameters[i, :], self.previous_hyperparameters
            )

        cov_k = gaussian_kernel(hyperparameters, hyperparameters[None, ...])

        aux_vector = vector_k.T @ np.linalg.inv(matrix_sigma)
        mean = cast(float, aux_vector @ self.previous_performances)
        std = cast(float, cov_k - aux_vector @ vector_k)

        return mean, std

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
        d["all_performance"] = self.previous_performances.tolist()
        d["all_hyperparameters"] = self.previous_hyperparameters.tolist()
        return d
