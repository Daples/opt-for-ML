from time import time

import numpy as np
import pandas as pd

from utils.adjacency import NearestNeighborsAdjacency
from utils.eigen import eig


data_matrix = pd.read_csv("data.csv").to_numpy()
data_matrix = data_matrix[:, 1:]

X = data_matrix
ns = 4
times = np.zeros(2)

prev_time = time()
adjacency = NearestNeighborsAdjacency(X, n_neighbors=15)
adjacency.fit()

degree_matrix = adjacency.degree_matrix
adjacency_matrix = adjacency.adjacency_matrix

# Compute normalized L
diag = np.diag(degree_matrix)
degree_matrix = np.diag(1 / np.sqrt(diag))
identity = np.identity(degree_matrix.shape[0])
graph_laplacian = identity - degree_matrix @ adjacency_matrix @ degree_matrix

# Get eigen decomposition
vals, vecs = eig(graph_laplacian)
times[0] = time() - prev_time

# Sa
np.savez_compressed("./outputs/eig_normalized.npz", vals=vals, vecs=vecs)

prev_time = time()
adjacency = NearestNeighborsAdjacency(X, n_neighbors=15)
adjacency.fit()

# Compute unnormalized L
degree_matrix = adjacency.degree_matrix
adjacency_matrix = adjacency.adjacency_matrix
graph_laplacian = degree_matrix - adjacency_matrix  # type: ignore

# Get eigen decomposition
vals, vecs = eig(graph_laplacian)
times[1] = time() - prev_time
np.savez("./outputs/eig.npz", vals=vals, vecs=vecs)

np.savez_compressed("outputs/times.npz", times=times)
