from abc import ABC, abstractmethod
from typing import Callable

from numpy.random import Generator

import numpy as np

from utils import euclidean


class ClusteringMethod(ABC):
    """An abstract class to represent a clustering algorithm.

    Attributes
    ----------
    metric: np.ndarray, np.ndarray -> np.ndarray, optional
        The operator to evaluate distance. Default: utils.euclidean
    n_clusters: int, optional
        The number of clusters for the clustering method. Default: 1.
    belonging_map: dict[int, numpy.ndarray]
        The map between a cluster index and the indices of data points belonging to that
        cluster.
    _generator: numpy.random.Generator
        The random number generator instance.

    Properties
    ----------
    clusters: numpy.ndarray
        The matrix of cluster centers. Rows are clusters and columns are features.
    membership_matrix: numpy.ndarray
        The matrix U of memberships to a cluster. Each entry `U[i,j]` is 1 if the
        data point `j` belongs to cluster `i`, and 0 in any other case. This matrix is
        of dimensions `n_clusters x n`.
    distances: numpy.ndarray
        The squared distance between each data point and each cluster center. This
        matrix is of dimensions `n_clusters x n`.
    """

    def __init__(
        self,
        metric: Callable[[np.ndarray, np.ndarray], np.ndarray] = euclidean,
        generator: Generator = np.random.default_rng(),
        n_clusters: int = 1,
    ) -> None:
        self.metric: Callable[[np.ndarray, np.ndarray], np.ndarray] = metric
        self.n_clusters: int = n_clusters
        self.belonging_map: dict[int, np.ndarray] = {}
        self._generator: Generator = generator
        self._clusters: np.ndarray | None = None
        self._membership_matrix: np.ndarray | None = None
        self._distances: np.ndarray | None = None

    @property
    def clusters(self) -> np.ndarray:
        """It returns the initialized clusters matrix.

        Returns
        -------
        numpy.ndarray
            The clusters matrix.

        Raises
        ------
        ValueError
            When the clusters have not been initialized.
        """

        if self._clusters is None:
            raise ValueError("The clusters have not been initialized yet.")
        return self._clusters

    @clusters.setter
    def clusters(self, value: np.ndarray) -> None:
        """Setter for the cluster matrix.

        Parameters
        ----------
        value: numpy.ndarray
            The updated cluster matrix.
        """

        self._clusters = value

    @property
    def membership_matrix(self) -> np.ndarray:
        """It returns the initialized membership matrix.

        Returns
        -------
        numpy.ndarray
            The membership matrix.

        Raises
        ------
        ValueError
            When the memberships have not been initialized.
        """

        if self._membership_matrix is None:
            raise ValueError("The memberships have not been initialized yet.")
        return self._membership_matrix

    @membership_matrix.setter
    def membership_matrix(self, value: np.ndarray) -> None:
        """Setter for the membership matrix.

        Parameters
        ----------
        value: numpy.ndarray
            The updated membership matrix.
        """

        self._membership_matrix = value

    @property
    def distances(self) -> np.ndarray:
        """It returns the initialized distance matrix.

        Returns
        -------
        numpy.ndarray
            The square distance matrix.

        Raises
        ------
        ValueError
            When the distances have not been initialized.
        """

        if self._distances is None:
            raise ValueError("The distances have not been initialized yet.")
        return self._distances

    @distances.setter
    def distances(self, value: np.ndarray) -> None:
        """Setter for the distance matrix.

        Parameters
        ----------
        value: numpy.ndarray
            The updated distance matrix.
        """

        self._distances = value

    @abstractmethod
    def fit(self, data_matrix: np.ndarray) -> None:
        """It uses the algorithm to update the memberships.

        Parameters
        ----------
        data_matrix: numpy.ndarray
            The data matrix. Rows are data points and columns are features.
        """
