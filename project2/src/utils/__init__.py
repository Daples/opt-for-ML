import numpy as np
import pandas as pd
from numpy.random import Generator


def euclidean(point: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """It calculates the squared Euclidean metric from a point to each row of a matrix.

    Parameters
    ----------
    point: numpy.ndarray
        The reference point of dimension `m`.
    matrix: numpy.ndarray
        The matrix of points to measure distance. It is of dimensions `n x m`.

    Returns
    -------
    numpy.ndarray
        The array of dimension `n` of distances to each point.
    """

    return np.sum(np.power(np.subtract(matrix, point), 2), axis=1)


def gaussian_kernel(
    point: np.ndarray, matrix: np.ndarray, gamma: float = 0.5
) -> np.ndarray:
    """It calculates the Gaussian kernelized distanced from a point to each row of a
    matrix.

    Parameters
    ----------
    point: numpy.ndarray
        The reference point of dimension`m`.
    matrix: numpy.ndarray
        The matrix of points to measure distance. It is of dimensions `n x m`.
    gamma: float, optional
        The scaling factor of the Gaussian kernel.
    """

    return np.exp(-gamma * euclidean(point, matrix))


def get_data(path: str) -> tuple[np.ndarray, np.ndarray]:
    """It reads the data and returns the input and labels (assuming the labels are on
    the last column.

    Parameters
    ----------
    path: str
        The path to the input data.

    Returns
    -------
    numpy.ndarray
        The input data.
    numpy.ndarray
        The data labels.
    """

    dataframe = pd.read_csv(path)
    data_matrix = dataframe.to_numpy()
    return data_matrix[:, :-1], data_matrix[:, -1]


def get_random_hyperparameters(
    intervals: np.ndarray, generator: Generator
) -> np.ndarray:
    """It returns a vector of random hyperparameters chosen within the defined
    intervals.

    Returns
    -------
    numpy.ndarray
        The vector of hyperparameters.
    """

    n = intervals.shape[0]
    hyperparams = np.zeros(n)
    for i in range(n):
        lower = intervals[i, 0]
        upper = intervals[i, 1]
        hyperparams[i] = generator.uniform(lower, upper)
    return hyperparams


def is_within_intervals(point: np.ndarray, intervals: np.ndarray) -> bool:
    """It checks if a point is within the specified intervals.

    Parameters
    ----------
    point: numpy.ndarray
        An n-dimensional point.
    intervals: numpy.ndarray
        The array of interval bounds.

    Returns
    -------
    bool
        Whether the input point is inside the specified intervals.
    """

    is_within = True
    dim = point.shape[0]
    for i in range(dim):
        lower = intervals[i, 0]
        upper = intervals[i, 1]
        is_within = is_within and lower <= point[i] <= upper
    return is_within
