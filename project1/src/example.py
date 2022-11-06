import numpy as np
import matplotlib.pyplot as plt

from utils import euclidean
from clustering.kmeans import KMeans

point = np.array([0, 0])
matrix = np.array([[1, 1], [3, 4], [3, 4]])
euclidean(point, matrix)

generator = np.random.default_rng(123456789)
n = 3000
pop1 = np.random.multivariate_normal([0, 0], 0.3 * np.identity(2), size=n)
pop2 = np.random.multivariate_normal(
    [2, 3], np.array([[0.5, -0.75], [-0.75, 1.5]]), size=n
)
X = np.vstack((pop1, pop2))

kmeans = KMeans(generator, n_clusters=10)
kmeans.fit(X, epsilon=1e-5)


# Plot clusters
fig = plt.figure()
ax = plt.subplot(111)
kwargs = {"zorder": 3, "s": 8}
for i in range(kmeans.n_clusters):
    indices = kmeans.belonging_map[i]
    data_cluster = X[indices, :]
    plt.scatter(
        data_cluster[:, 0],
        data_cluster[:, 1],
        label=f"Cluster {i}",
        **kwargs,
    )

    center = kmeans.clusters[i, :]
    plt.scatter(center[0], center[1], c="r", **kwargs)

# Get legend outside of plot
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc="upper left", bbox_to_anchor=(1, 0.5))

plt.grid()
plt.show()
