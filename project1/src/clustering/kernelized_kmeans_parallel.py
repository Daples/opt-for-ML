import itertools
import multiprocessing
from typing import Callable

import numpy as np
from joblib import Parallel, delayed
from numpy.random import Generator
from tqdm import tqdm

from clustering.kmeans import KMeans
from utils import gaussian_kernel


class KernelizedKMeans(KMeans):
    """A class that represents the k-means algorithm with Kernel application.

    Properties
    ----------
    kernel_similarities

    _cpu_cores
    """

    def __init__(
        self,
        metric: Callable[[np.ndarray, np.ndarray], np.ndarray] = gaussian_kernel,
        generator: Generator = np.random.default_rng(),
        n_clusters: int = 1,
        cpu_cores: int | None = None,
    ) -> None:
        super().__init__(metric, generator, n_clusters)
        self._kernel_similarities: np.ndarray | None = None
        if cpu_cores is None:
            cpu_cores = multiprocessing.cpu_count()
        self._cpu_cores: int = cpu_cores

    @property
    def kernel_similarities(self) -> np.ndarray:
        """It returns the initialized kernelized distances matrix.

        Returns
        -------
        numpy.ndarray
            The kernel distance matrix.

        Raises
        ------
        ValueError
            When the kernel distances have not been initialized.
        """

        if self._kernel_similarities is None:
            raise ValueError("The distances have not been initialized yet.")
        return self._kernel_similarities

    def __init_kernel_distances__(self, data_matrix: np.ndarray) -> None:
        """It initialized the kernel distances matrix.

        Parameters
        ----------
        data_matrix: numpy.ndarray
            The data matrix. Rows are data points and columns are features.
        """

        n = data_matrix.shape[0]
        self._kernel_similarities = np.zeros((n, n))
        for i in range(n):
            self._kernel_similarities[i, :] = self.metric(
                data_matrix[i, :], data_matrix
            )

    def fit(
        self,
        data_matrix: np.ndarray,
        n_iter: int | None = None,
    ) -> None:
        """It uses kernelized k-means to obtain the memberships.

        Parameters
        ----------
        data_matrix: numpy.ndarray
            The data matrix. Rows are data points and columns are features.
        n_iter: int | None
            The maximum number of iterations.
        """

        def _is_finished(i: int) -> bool:
            """Auxiliary method that evaluates the stopping criterion for the algorithm.

            Parameters
            ----------
            i: int
                The current iteration number.

            Returns
            -------
            bool
                True when one of the stopping criteria is met.
            """

            condition_iter = False
            if n_iter is not None:
                condition_iter = n_iter < i
            return condition_iter

        # Initialize the kernel similarities matrix
        self.__init_kernel_distances__(data_matrix)

        # Initialize clusters
        n = data_matrix.shape[0]
        self.membership_matrix = np.zeros((self.n_clusters, n))
        for i in range(n):
            belonging_cluster = self._generator.integers(0, self.n_clusters)
            self.membership_matrix[belonging_cluster, i] = 1

        # Update the initial belonging map
        for cluster_index in range(self.n_clusters):
            self.belonging_map[cluster_index] = list(
                map(
                    lambda x: int(x),
                    self.membership_matrix[cluster_index, :].nonzero()[0],
                )
            )

        # Iterate
        i = 0
        while not _is_finished(i):
            # Find memberships
            self._update_membership(data_matrix)

            # Update centers
            self._update_centers(data_matrix)

            # Update the initial belonging map
            for cluster_index in range(self.n_clusters):
                self.belonging_map[cluster_index] = list(
                    map(
                        lambda x: int(x),
                        self.membership_matrix[cluster_index, :].nonzero()[0],
                    )
                )
            i += 1

    def _update_membership(self, data_matrix: np.ndarray) -> None:
        """It updates the membership matrix.

        Parameters
        ----------
        data_matrix: numpy.ndarray
            The data matrix.
        """

        n = data_matrix.shape[0]
        dims = (self.n_clusters, n)

        def aux(j: int, n_clusters: int, n: int) -> np.ndarray:
            """It estimates the distances. Used for parallel computation."""

            array = np.zeros((n_clusters, n))
            k1 = self.kernel_similarities[j, j]

            for k in range(n_clusters):
                members = self.belonging_map[k]
                indices = np.array(list(itertools.product(members, members)))
                if len(members) == 0:
                    continue
                k2 = self.kernel_similarities[j, members].sum()

                k3 = self.kernel_similarities[indices[:, 0], indices[:, 1]].sum()

                n_members = len(members)
                array[k] = k1 - 2 / n_members * k2 + 1 / n_members**2 * k3

            return array

        self.distances = np.array(
            Parallel(n_jobs=self._cpu_cores)(
                delayed(aux)(j, self.n_clusters, n) for j in tqdm(range(n))
            )
        ).T

        self.membership_matrix = np.zeros(dims)
        min_indices = np.argmin(self.distances, axis=0)
        indices = np.arange(n)
        self.membership_matrix[min_indices, indices] = 1

    def _update_centers(self, _: np.ndarray) -> None:
        """It does nothing."""
