import numpy as np


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
