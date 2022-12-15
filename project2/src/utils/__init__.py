import pandas as pd
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
