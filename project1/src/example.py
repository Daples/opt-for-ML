import matplotlib.pyplot as plt
import numpy as np

from clustering.kmeans import KMeans

generator = np.random.default_rng(123456789)
n = 3000
pop1 = np.random.multivariate_normal([0, 0], 0.3 * np.identity(2), size=n)
pop2 = np.random.multivariate_normal(
    [2, 3], np.array([[0.5, -0.75], [-0.75, 1.5]]), size=n
)
X = np.vstack((pop1, pop2))


kmeans = KMeans(generator=generator, n_clusters=2)
kmeans.fit(X, n_iter=100)


# Plot clusters
clustering_obj = kmeans
fig = plt.figure()
ax = plt.subplot(111)
kwargs = {"zorder": 3, "s": 8}
for i in range(clustering_obj.n_clusters):
    indices = clustering_obj.belonging_map[i]
    data_cluster = X[indices, :]
    plt.scatter(
        data_cluster[:, 0],
        data_cluster[:, 1],
        label=f"Cluster {i}",
        **kwargs,
    )

    center = clustering_obj.clusters[i, :]
    plt.scatter(center[0], center[1], c="r", **kwargs)

# Get legend outside of plot
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc="upper left", bbox_to_anchor=(1, 0.5))

plt.grid()
plt.show()
