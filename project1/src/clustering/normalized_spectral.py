import numpy as np

from clustering.spectral import Spectral


class NormalizedSpectral(Spectral):
    """A class that represents normalized spectral clustering."""

    def _get_graph_laplacian(self) -> np.ndarray:
        """A method that calculates the normalized graph Laplacian.

        Returns
        -------
        np.ndarray
            The normalized graph Laplacian matrix.
        """

        degree_matrix = self.adjacency.degree_matrix
        adjacency_matrix = self.adjacency.adjacency_matrix

        # Compute normalized D (exploit the fact that it is diagonal)
        diag = np.diag(degree_matrix)
        degree_matrix = np.diag(1 / np.sqrt(diag))
        identity = np.identity(degree_matrix.shape[0])
        graph_laplacian = identity - degree_matrix @ adjacency_matrix @ degree_matrix
        return graph_laplacian
