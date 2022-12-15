import matplotlib.pyplot as plt
import numpy as np

from clustering.normalized_spectral import NormalizedSpectral
from clustering.spectral import Spectral
from utils.adjacency import NearestNeighborsAdjacency
from utils.sample_generation import generate_circles

generator = np.random.default_rng(12345)
X = generate_circles(1000, [1, 5, 7], seed=generator)
epsilon = 1e-7
n_iter = 250
n_clusters = 2

gamma = 0.5
neighbors = 20
adjacency = NearestNeighborsAdjacency(X, neighbors, gamma=gamma, eigen_solver="numpy")
spectral = NormalizedSpectral(adjacency, generator=generator, n_clusters=n_clusters)
spectral.fit(X, epsilon=epsilon, n_iter=n_iter)

# Plot clusters
clustering = spectral
fig = plt.figure()
ax = plt.subplot(111)
kwargs = {"zorder": 3, "s": 8}
for i in range(clustering.n_clusters):
    indices = clustering.belonging_map[i]
    data_cluster = X[indices, :]
    plt.scatter(
        data_cluster[:, 0],
        data_cluster[:, 1],
        label=f"Cluster {i}",
        **kwargs,
    )

# Get legend outside of plot
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc="upper left", bbox_to_anchor=(1, 0.5))

plt.grid()
plt.show()
