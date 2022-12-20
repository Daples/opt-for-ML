import json

import numpy as np

from hyper.bayesian import BayesianOptimizer
from hyper.grid_search import GridSearchOptimizer
from utils import get_data
from utils.plotter import Plotter

X, y = get_data("/home/daples/git/opt-for-ML/project2/src/heart.csv")
names = ["C", "gamma"]
intervals = np.array([[0, 9], [-10, 0]])
generator = 98765

# Grid Search
N = 10
gs = GridSearchOptimizer(intervals, N, names, generator)
gs.optimize(X, y)

xx = gs.domain_matrix[0, :]
yy = gs.domain_matrix[1, :]
Plotter.get_heatmatp(xx, yy, gs.performance_matrix, "test.pdf")

with open("output_gs.json", "w") as outfile:
    json.dump(gs.get_json(), outfile, indent=4)

# Bayesian Optimization
bo = BayesianOptimizer(names, intervals, generator)
N = 100
n_init = 6
n_contours = 10
bo.optimize(X, y, n_iter=N, n_init=n_init, n_contours=n_contours)

# Get scatter
cs = bo.previous_hyperparameters[:, 0]
gammas = bo.previous_hyperparameters[:, 1]
Plotter.get_scatter(cs, gammas, "scatter.pdf")

# Get misclassification plot
Plotter.get_misclassification_plot(
    bo.previous_performances, gs.best_performance, "misclassification.pdf"
)

with open(f"output_bo_{N}.json", "w") as outfile:
    json.dump(bo.get_json(), outfile, indent=4)
