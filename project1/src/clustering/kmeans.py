from clustering.core.cluster import ClusteringMethod
from utils import euclidean

import numpy as np


class KMeans(ClusteringMethod):
    """A class that represent the K-Means clustering algorithm.

    Attributes
    ----------
    metric: numpy.ndarray, numpy.ndarray -> numpy.ndarray
        The metric between two sets of data points.
    """

    def loss_function(self) -> float:
        """The loss for the k-means algorithm.

        Returns
        -------
        float
            The objective function value at the current iteration.
        """

        return np.sum(np.multiply(self.membership_matrix, self.distances))

    def fit(
        self,
        data_matrix: np.ndarray,
        epsilon: float | None = None,
        n_iter: int | None = None,
    ) -> None:
        """It uses k-means to obtain the memberships.

        Parameters
        ----------
        data_matrix: numpy.ndarray
            The data matrix. Rows are data points and columns are features.
        epsilon: float | None
            The convergence threshold.
        n_iter: int | None
            The maximum number of iterations.
        """

        def _is_finished(i: int, improvement: float) -> bool:
            """Auxiliary method that evaluates the stopping criterion for the algorithm.

            Parameters
            ----------
            i: int
                The current iteration number.
            improvement: float
                The objective function improvement.

            Returns
            -------
            bool
                True when one of the stopping criteria is met.
            """

            if epsilon is None and n_iter is None:
                raise ValueError("The algorithm must have a stopping criterion.")
            condition_epsilon = False
            if epsilon is not None:
                condition_epsilon = improvement <= epsilon
            condition_iter = False
            if n_iter is not None:
                condition_iter = n_iter > i
            return condition_epsilon or condition_iter

        # Initialize centers
        n = data_matrix.shape[0]
        indices = np.arange(n)
        indices = self._generator.choice(indices, size=self.n_clusters, replace=False)
        self._clusters = data_matrix[indices, :]

        # Iterate
        improvement = np.inf
        i = 0
        prev_loss = 0
        while not _is_finished(i, improvement):
            # Find memberships
            self._update_membership(data_matrix)

            # Update centers
            sums = np.sum(self.membership_matrix, axis=1)
            self.clusters = self.membership_matrix @ data_matrix / sums[:, None]

            # Calculate loss and improvement
            current_loss = self.loss_function()
            improvement = current_loss - prev_loss
            prev_loss = current_loss
            i += 1

        for cluster_index in range(self.n_clusters):
            self.belonging_map[cluster_index] = self.membership_matrix[
                cluster_index, :
            ].nonzero()[0]

    def _update_membership(self, data_matrix: np.ndarray) -> None:
        """It updates both the membership and distance matrices.

        Parameters
        ----------
        data_matrix: numpy.ndarray
            The data matrix.
        """

        dims = (self.n_clusters, data_matrix.shape[0])
        self.distances = np.zeros(dims)
        self.membership_matrix = np.zeros(dims)

        # Update distances
        # TODO: generalize for other inner products (Kernel)
        for i in range(self.n_clusters):
            self.distances[i, :] = euclidean(self.clusters[i, :], data_matrix)

        # Update membership
        rows = np.argmin(self.distances, axis=0)
        cols = np.arange(data_matrix.shape[0])
        self.membership_matrix[rows, cols] = 1
