from abc import ABC, abstractmethod

from utils import gaussian_kernel, euclidean

import numpy as np


class AdjacencyMatrix(ABC):
    """An abstract class to represent a method to estimate the adjacency matrix.

    Attributes
    ----------
    data: numpy.ndarray
        The data matrix.
    _gamma: float, optional
        The scaling factor for the Gaussian kernel. Default: 0.5

    Properties
    ----------
    adjacency_matrix: numpy.ndarray
        The adjacency matrix.
    """

    def __init__(self, data_matrix: np.ndarray, gamma: float) -> None:
        self.data: np.ndarray = data_matrix
        self._gamma: float = gamma
        self._adjacency_matrix: np.ndarray | None = None
        self._degree_matrix: np.ndarray | None = None

    @property
    def adjacency_matrix(self) -> np.ndarray:
        """Property for the adjacency matrix.

        Returns
        -------
        numpy.ndarray
            The initialized adjacency matrix.

        Raises
        ------
        ValueError
            When the adjacency matrix has not been initialized yet.
        """

        if self._adjacency_matrix is None:
            raise ValueError("The adjacency matrix has not been initialized yet.")
        return self._adjacency_matrix

    @property
    def degree_matrix(self) -> np.ndarray:
        """It returns the degree matrix.

        Returns
        -------
        numpy.ndarray
            The initialized degree matrix.

        Raises
        ------
        ValueError
            When the degree matrix has not been initialized yet.
        """

        if self._degree_matrix is None:
            sums = self.adjacency_matrix.sum(axis=1)
            self._degree_matrix = np.diag(sums)
        return self._degree_matrix

    @abstractmethod
    def _get_indices(self, similarity: np.ndarray) -> np.ndarray:
        """It returns the indices of the connected points.

        Parameters
        ----------
        similarity: numpy.ndarray
            The similarity array between the current point and all others.

        Returns
        -------
        numpy.ndarray
            The array of indices.
        """

    def fit(self) -> None:
        """It fills the adjacency matrix."""

        # Ensure that it is not computed twice
        if self._adjacency_matrix is not None:
            return

        n = self.data.shape[0]
        self._adjacency_matrix = np.zeros((n, n))
        for i in range(n):
            # Compute similarity
            similarity = gaussian_kernel(self.data[i, :], self.data, self._gamma)

            # Get indices of connected points
            idx = self._get_indices(similarity)

            for j in idx:
                self._adjacency_matrix[i, j] = 1
                self._adjacency_matrix[j, i] = 1


class NearestNeighborsAdjacency(AdjacencyMatrix):
    """A class to represent an adjacency matrix based on nearest neighbors.

    Attributes
    ----------
    n_neighbors: int
        The number of neighbors.
    """

    def __init__(
        self, data_matrix: np.ndarray, n_neighbors: int, gamma: float = 0.5
    ) -> None:
        super().__init__(data_matrix, gamma)
        self.n_neighbors: int = n_neighbors

    def _get_indices(self, similarity: np.ndarray) -> np.ndarray:
        """It returns the indices of the connected points by nearest neighbors.

        Parameters
        ----------
        similarity: numpy.ndarray
            The similarity array between the current point and all others.

        Returns
        -------
        numpy.ndarray
            The array of indices.
        """

        return np.argpartition(similarity, -self.n_neighbors)[-self.n_neighbors :]


class SimilarityThresholdAdjacency(AdjacencyMatrix):
    """A class to represent an adjacency matrix constructed by using a threshold on the
    similarity.

    Attributes
    ----------
    beta: float
        The minimum similarity to connect two points.
    """

    def __init__(
        self, data_matrix: np.ndarray, beta: float, gamma: float = 0.5
    ) -> None:
        super().__init__(data_matrix, gamma)
        self.beta: float = beta

    def _get_indices(self, similarity: np.ndarray) -> np.ndarray:
        """It returns the indices of the points with similarity above `beta`.

        Parameters
        ----------
        similarity: numpy.ndarray
            The similarity array between the current point and all others.

        Returns
        -------
        numpy.ndarray
            The array of indices.
        """

        return similarity > self.beta
