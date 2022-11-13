import numpy as np
import matplotlib.pyplot as plt

from utils import euclidean
from utils.adjacency import NearestNeighborsAdjacency
from utils.adjacency import SimilarityThresholdAdjacency
from clustering.kmeans import KMeans
from clustering.spectral import Spectral

generator = np.random.default_rng(123456789)
n = 250

r1 = 5
r2 = 1
xs1 = np.linspace(-r1, r1, n)
xs2 = np.linspace(-r2, r2, n)
data1 = []
data2 = []
f = lambda x, r: np.sqrt(r**2 - x**2) + generator.normal(0, 0.2)

data1 = np.zeros((2 * xs1.size, 2))
data1[:, 0] = np.hstack((xs1, xs1))

data2 = np.zeros((2 * xs2.size, 2))
data2[:, 0] = np.hstack((xs2, xs2))

for i in range(xs1.size):
    data1[i, 1] = f(xs1[i], r1)
    data1[i + xs1.size - 1, 1] = -f(xs1[i], r1)


for i in range(xs1.size):
    data2[i, 1] = f(xs2[i], r2)
    data2[i + xs2.size - 1, 1] = -f(xs2[i], r2)

X = np.vstack((data1, data2))
epsilon = 1e-7
n_iter = 250
n_clusters = 2
# kmeans = KMeans(metric=euclidean, generator=generator, n_clusters=n_clusters)
# kmeans.fit(X, epsilon=epsilon)

gamma = 10
beta = 0.8
neighbors = 10
adjacency = NearestNeighborsAdjacency(X, neighbors, gamma=gamma)
# adjacency = SimilarityThresholdAdjacency(X, beta, gamma)
spectral = Spectral(adjacency, generator=generator, n_clusters=n_clusters)
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
