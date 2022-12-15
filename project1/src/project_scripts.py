import os
from time import time

import matplotlib.pyplot as plt
import numpy as np

from clustering.kernelized_kmeans import KernelizedKMeans
from clustering.kmeans import KMeans
from clustering.normalized_spectral import NormalizedSpectral
from clustering.spectral import Spectral
from utils import get_data_frame, get_data_matrix, write_json
from utils.adjacency import NearestNeighborsAdjacency

# General configuration
seed = 12345
generator = np.random.default_rng(seed)
epsilon = 1e-8
max_iter = 50
plt.rc("text", usetex=True)
plt.rcParams.update({"font.size": 18})

# I/O
out_folder = "./outputs"
data_path = "./data.csv"
data_matrix = get_data_matrix(data_path, True)
df = get_data_frame(data_path)

# Number of Clusters
Ks = [i for i in range(2, 12)]
vals = []
for k in Ks:
    kmeans = KMeans(generator=np.random.default_rng(123456), n_clusters=k)
    kmeans.fit(data_matrix, epsilon)
    vals.append(kmeans.loss_function())
plt.plot(vals, "k")
plt.grid()
plt.xlabel("$K$")
plt.ylabel("Loss")
plt.savefig("outputs/loss_ks.pdf", bbox_inches="tight")

n_clusters = 4

# K-Means
out_path = os.path.join(out_folder, f"membership_{n_clusters}_kmeans.json")
kmeans = KMeans(generator=generator, n_clusters=n_clusters)
prev_time = time()
kmeans.fit(data_matrix, epsilon=epsilon, n_iter=max_iter)
print("Fitting time K-Means:", time() - prev_time)
write_json(kmeans.belonging_map, out_path)


# Kernelized K-Means
out_path = os.path.join(out_folder, f"membership_{n_clusters}_kernel_kmeans.json")
kernel_kmeans = KernelizedKMeans(generator=generator, n_clusters=n_clusters)
prev_time = time()
kernel_kmeans.fit(data_matrix, n_iter=max_iter)
print("Fitting time Kernel K-Means:", time() - prev_time)
write_json(kernel_kmeans.belonging_map, out_path)


# Spectral - Custom solver
eigen_path = os.path.join(out_folder, "eig.npz")
obj = np.load(eigen_path)
vals = obj["vals"]
vecs = obj["vecs"]
out_path = os.path.join(out_folder, f"membership_{n_clusters}_spectral_custom.json")
adjacency = NearestNeighborsAdjacency(
    data_matrix,
    n_neighbors=15,
    eigenvalues=vals,
    eigenvectors=vecs,
)
adjacency.fit()
spectral = Spectral(adjacency, generator=generator, n_clusters=n_clusters)
prev_time = time()
spectral.fit(data_matrix, epsilon=epsilon, n_iter=max_iter)
print("Fitting time spectral (custom eig):", time() - prev_time)
write_json(spectral.belonging_map, out_path)


# Spectral - NumPy solver
out_path = os.path.join(out_folder, f"membership_{n_clusters}_spectral_numpy.json")
adjacency = NearestNeighborsAdjacency(data_matrix, n_neighbors=15, eigen_solver="numpy")
adjacency.fit()
spectral = Spectral(adjacency, generator=generator, n_clusters=n_clusters)
prev_time = time()
spectral.fit(data_matrix, epsilon=epsilon, n_iter=max_iter)
print("Fitting time spectral (NumPy eig):", time() - prev_time)
write_json(spectral.belonging_map, out_path)


# Normalized Spectral - Custom
eigen_path = os.path.join(out_folder, "eig_normalized.npz")
obj = np.load(eigen_path)
vals = obj["vals"]
vecs = obj["vecs"]
out_path = os.path.join(
    out_folder, f"membership_{n_clusters}_norm_spectral_custom.json"
)
adjacency = NearestNeighborsAdjacency(
    data_matrix, n_neighbors=25, eigenvalues=vals, eigenvectors=vecs
)
adjacency.fit()
normalized_spectral = NormalizedSpectral(
    adjacency, generator=generator, n_clusters=n_clusters
)
prev_time = time()
normalized_spectral.fit(data_matrix, epsilon=epsilon, n_iter=max_iter)
print("Fitting time normalized spectral (custom eig):", time() - prev_time)
write_json(normalized_spectral.belonging_map, out_path)


# Normalized Spectral - NumPy
out_path = os.path.join(out_folder, f"membership_{n_clusters}_norm_spectral_numpy.json")
adjacency = NearestNeighborsAdjacency(
    data_matrix,
    n_neighbors=25,
    eigen_solver="numpy",
)
adjacency.fit()
normalized_spectral = NormalizedSpectral(
    adjacency, generator=generator, n_clusters=n_clusters
)
prev_time = time()
normalized_spectral.fit(data_matrix, epsilon=epsilon, n_iter=max_iter)
print("Fitting time normalized spectral (NumPy eig):", time() - prev_time)
write_json(normalized_spectral.belonging_map, out_path)
