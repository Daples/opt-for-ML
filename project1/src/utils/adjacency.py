from abc import ABC, abstractmethod
from typing import cast

import numpy as np

from utils import gaussian_kernel
from utils.eigen import eig


class AdjacencyMatrix(ABC):
    """An abstract class to represent a method to estimate the adjacency matrix.

    Attributes
    ----------
    data: numpy.ndarray
        The data matrix.
    eigen_solver: str, optional
        The eigen solver to use. Default: "numpy"
    _eigenvalues: numpy.ndarray | None, optional
        The eigenvalues of the graph Laplacian (if available). Default: None
    _eigenvectors: numpy.ndarray | None, optional
        The eigenvectors of the graph Laplacian (if available). Default: None
    _gamma: float, optional
        The scaling factor for the Gaussian kernel. Default: 0.5

    Properties
    ----------
    adjacency_matrix: numpy.ndarray
        The adjacency matrix.
    degree_matrix: numpy.ndarray
        The degree matrix.
    """

    def __init__(
        self,
        data_matrix: np.ndarray,
        gamma: float,
        eigen_solver: str = "numpy",
        eigenvalues: np.ndarray | None = None,
        eigenvectors: np.ndarray | None = None,
    ) -> None:
        self.data: np.ndarray = data_matrix
        self._eigen_solver: str = eigen_solver
        self._eigenvalues: np.ndarray | None = eigenvalues
        self._eigenvectors: np.ndarray | None = eigenvectors
        self._gamma: float = gamma
        self._adjacency_matrix: np.ndarray | None = None
        self._degree_matrix: np.ndarray | None = None
        self._laplacian_matrix: np.ndarray | None = None

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

    def eigen_decomposition(
        self, laplacian_matrix: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """It returns the eigenvalues and vectors of the input Laplacian matrix.

        Returns
        -------
        numpy.ndarray
            The vector of eigenvalues.
        numpy.ndarray
            The matrix eigenvectors on its columns.

        Raises
        ------
        ValueError
            When the passed eigen solver is not implemented.
        """

        if self._eigenvalues is None and self._eigenvectors is None:
            match self._eigen_solver:
                case "numpy":
                    self._eigenvalues, self._eigenvectors = np.linalg.eig(
                        laplacian_matrix
                    )
                case "custom":
                    self._eigenvalues, self._eigenvectors = eig(laplacian_matrix)
                case _:
                    raise ValueError("The requested eigen solver is not supported.")
        return (
            cast(np.ndarray, self._eigenvalues),
            cast(np.ndarray, self._eigenvectors),
        )


class NearestNeighborsAdjacency(AdjacencyMatrix):
    """A class to represent an adjacency matrix based on nearest neighbors.

    Attributes
    ----------
    n_neighbors: int
        The number of neighbors.
    """

    def __init__(
        self,
        data_matrix: np.ndarray,
        n_neighbors: int,
        gamma: float = 0.5,
        eigen_solver: str = "numpy",
        eigenvalues: np.ndarray | None = None,
        eigenvectors: np.ndarray | None = None,
    ) -> None:
        super().__init__(data_matrix, gamma, eigen_solver, eigenvalues, eigenvectors)
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
        self,
        data_matrix: np.ndarray,
        beta: float,
        gamma: float,
        eigen_solver: str = "numpy",
        eigenvalues: np.ndarray | None = None,
        eigenvectors: np.ndarray | None = None,
    ) -> None:
        super().__init__(data_matrix, gamma, eigen_solver, eigenvalues, eigenvectors)
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
