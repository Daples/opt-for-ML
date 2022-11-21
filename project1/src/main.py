import click
import numpy as np

from clustering.kernelized_kmeans import KernelizedKMeans
from clustering.kmeans import KMeans
from clustering.normalized_spectral import NormalizedSpectral
from clustering.spectral import Spectral
from utils import get_data_matrix, write_json
from utils.adjacency import NearestNeighborsAdjacency

SEED = click.option(
    "-s",
    "--seed",
    default=12345,
    type=int,
    help="The RNG seed.",
    show_default=True,
)
N_CLUSTERS = click.option(
    "-k",
    "--k-clusters",
    default=2,
    type=int,
    help="The number of clusters to use.",
    show_default=True,
)
EPSILON = click.option(
    "-e",
    "--epsilon",
    default=1e-7,
    type=float,
    help="The tolerance for stopping the iterations (desired improvement).",
    show_default=True,
)
M = click.option(
    "-m",
    "--max_iterations",
    default=250,
    type=int,
    help="The maximum number of iterations.",
    show_default=True,
)
REMOVE_INDEX = click.option(
    "-I",
    "--remove-index",
    help="If the first column of the CSV file should be removed.",
    is_flag=True,
)
N_NEIGHBORS = click.option(
    "-n",
    "--neighbors",
    default=15,
    type=int,
    help="The number of neighbors for the adjacency matrix.",
    show_default=True,
)
OUTFILE = click.option(
    "-o",
    "--outfile",
    default="output.csv",
    type=click.Path(),
    help="Filename to write the membership maps.",
)
KERNELIZE = click.option(
    "-K",
    "--kernelize",
    help="If kmeans should use the Kernel trick.",
    is_flag=True,
)
NORMALIZE = click.option(
    "-N",
    "--normalize",
    help="If the spectral clustering should use the normalized Laplacian.",
    is_flag=True,
)

INPUT_DATA = click.argument("input_data", type=click.Path(exists=True))


@click.group()
def main() -> None:
    """The CLI for clustering data."""


@main.command()
@SEED
@N_CLUSTERS
@EPSILON
@M
@REMOVE_INDEX
@KERNELIZE
@OUTFILE
@INPUT_DATA
def kmeans(
    seed: int,
    k_clusters: int,
    epsilon: float,
    max_iterations: int,
    remove_index: bool,
    kernelize: bool,
    outfile: str,
    input_data: str,
) -> None:
    """Use K-Means to cluster the input data."""

    # Read data
    data_matrix = get_data_matrix(input_data, remove_index)

    # Fit K-Means
    generator = np.random.default_rng(seed)
    if kernelize:
        kmeans = KernelizedKMeans(generator=generator)
        args = {}
    else:
        kmeans = KMeans(generator=generator, n_clusters=k_clusters)
        args = {"epsilon": epsilon}
    kmeans.fit(data_matrix, n_iter=max_iterations, **args)

    # Write the membership maps
    write_json(kmeans.belonging_map, outfile)


@main.command()
@SEED
@N_CLUSTERS
@EPSILON
@M
@N_NEIGHBORS
@REMOVE_INDEX
@NORMALIZE
@OUTFILE
@INPUT_DATA
def spectral(
    seed: int,
    k_clusters: int,
    epsilon: float,
    max_iterations: int,
    neighbors: int,
    remove_index: bool,
    normalize: bool,
    outfile: str,
    input_data: str,
) -> None:
    """Use Spectral clustering to split the data."""

    # Read data
    data_matrix = get_data_matrix(input_data, remove_index)

    # Fit the adjacency matrix
    adjacency = NearestNeighborsAdjacency(data_matrix, neighbors)

    # Fit the clustering model
    generator = np.random.default_rng(seed)
    args = {"generator": generator, "n_clusters": k_clusters}
    if normalize:
        spectral = NormalizedSpectral(adjacency, **args)
    else:
        spectral = Spectral(adjacency, **args)
    spectral.fit(data_matrix, epsilon=epsilon, n_iter=max_iterations)

    # Write the membership maps
    write_json(spectral.belonging_map, outfile)


# Execute CLI
if __name__ == "__main__":
    main()
