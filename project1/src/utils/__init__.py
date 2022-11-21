import json
import os

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame


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


def get_data_frame(path: str) -> DataFrame:
    """"""

    return pd.read_csv(path)


def get_data_matrix(path: str, remove_index: bool) -> np.ndarray:
    """It reads the input file and applies corrections if needed.

    Parameters
    ----------
    path: str
        The path to the CSV file.
    remove_index: bool
        If the first column should be removed.

    Returns
    -------
    numpy.ndarray
        The data matrix.

    Raises
    ------
    ValueError
        When the path does not correspond with a CSV file.
    FileNotFoundError
        When the specified file does not exist.
    """

    if path.split(".")[-1] != "csv":
        raise ValueError("The path does not correspond with a CSV file.")

    if not os.path.exists(path):
        raise FileNotFoundError("The file does not exist.")

    data_frame = pd.read_csv(path)
    return __correct_dataframe__(data_frame, remove_index)


def __correct_dataframe__(data_frame: DataFrame, remove_index: bool) -> np.ndarray:
    """Applies the desired corrections and returns a NumPy array.

    Parameters
    ----------
    data_frame: pandas.core.frame.DataFrame
        The input dataframe.
    remove_index: bool
        If the first column should be removed from the data.

    Returns
    -------
    numpy.ndarray
        The corrected matrix.
    """

    data_matrix = data_frame.to_numpy()
    if remove_index:
        data_matrix = data_matrix[:, 1:]
    return data_matrix


def write_json(obj: dict, outfile: str) -> None:
    """Writes the input object to JSON format.

    Parameters
    ----------
    obj: dict
        The dictionary to write.
    outfile: str
        The file path.
    """

    with open(outfile, "w") as out_file:
        json.dump(obj, out_file, indent=4)


def read_membership(path: str) -> dict:
    """It reads a membership map from a file.

    Parameters
    ----------
    path: str
        The path to the file.

    Returns
    -------
    dict
        The membership map.

    Raises
    ------
    FileNotFoundError
        When the path does not exist.
    """

    if not os.path.exists(path):
        raise FileNotFoundError("The file does not exist.")

    with open(path, "r") as source_file:
        return json.load(source_file)
