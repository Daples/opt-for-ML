from typing import Callable

import numpy as np
from numpy.random import Generator

from clustering.core.cluster import ClusteringMethod
from clustering.kmeans import KMeans
from utils import euclidean
from utils.adjacency import AdjacencyMatrix


class Spectral(ClusteringMethod):
    """A class that represent the spectral clustering algorithm.

    Attributes
    ----------
    adjacency: utils.adjacency.AdjacencyMatrix
        The adjacency matrix object.
    """

    def __init__(
        self,
        adjacency: AdjacencyMatrix,
        metric: Callable[[np.ndarray, np.ndarray], np.ndarray] = euclidean,
        generator: Generator = np.random.default_rng(),
        n_clusters: int = 1,
    ) -> None:
        super().__init__(metric, generator, n_clusters)
        self.adjacency: AdjacencyMatrix = adjacency

    def fit(
        self,
        _: np.ndarray,
        epsilon: float | None = None,
        n_iter: int | None = None,
    ) -> None:
        """It uses spectral clustering to obtain the memberships.

        Parameters
        ----------
        data_matrix: numpy.ndarray
            The data matrix. Rows are data points and columns are features.
        epsilon: float | None
            The convergence threshold.
        n_iter: int | None
            The maximum number of iterations.
        """

        # Fit the adjacency matrix (if not done before)
        self.adjacency.fit()

        # Graph Laplacian
        graph_laplacian = self._get_graph_laplacian()

        # Eigen decomposition of the graph Laplacian
        vals, vecs = self.adjacency.eigen_decomposition(graph_laplacian)
        idx_max = np.argpartition(vals, self.n_clusters)[: self.n_clusters]
        H = vecs[:, idx_max]

        # Use k-means
        kmeans = KMeans(
            metric=self.metric, generator=self._generator, n_clusters=self.n_clusters
        )
        kmeans.fit(H, epsilon=epsilon, n_iter=n_iter)

        # Update clustering model attributes
        self.clusters = kmeans.clusters
        self.membership_matrix = kmeans.membership_matrix
        self.distances = kmeans.distances
        self.belonging_map = kmeans.belonging_map

    def _get_graph_laplacian(self) -> np.ndarray:
        """A method that calculates the graph Laplacian.

        Returns
        -------
        np.ndarray
            The graph Laplacian matrix.
        """

        degree_matrix = self.adjacency.degree_matrix
        adjacency_matrix = self.adjacency.adjacency_matrix
        graph_laplacian = degree_matrix - adjacency_matrix  # type: ignore
        return graph_laplacian
