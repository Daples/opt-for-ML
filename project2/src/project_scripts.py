import json

import numpy as np

from hyper.bayesian import BayesianOptimizer
from hyper.grid_search import GridSearchOptimizer
from utils import get_data
from utils.plotter import Plotter


X, y = get_data("/home/daples/git/opt-for-ML/project2/src/heart.csv")
names = ["C", "gamma"]
intervals = np.array([[0, 9], [-10, 0]])
generator = 123456

# Grid Search
N = 10
gs = GridSearchOptimizer(intervals, N, names)
gs.optimize(X, y)

xx = gs.domain_matrix[0, :]
yy = gs.domain_matrix[1, :]
Plotter.get_heatmatp(xx, yy, gs.performance_matrix, "hm.pdf")
Plotter.get_heatmatp(xx, yy, gs.times_matrix, "times.pdf")
with open(f"output_gs_{N}.json", "w") as outfile:
    json.dump(gs.get_json(), outfile, indent=4)


# Bayesian Optimization
bo = BayesianOptimizer(names, intervals, generator)
n_iter = 100
n_init = 6
n_contours = 4
bo.optimize(
    X, y, n_iter=n_iter, n_init=n_init, n_contours=n_contours, label=str(generator)
)

# Get scatter
cs = bo.previous_hyperparameters[:, 0]
gammas = bo.previous_hyperparameters[:, 1]
Plotter.get_scatter(cs, gammas, f"scatter_{generator}.pdf")

# Get misclassification plot
Plotter.get_misclassification_plot(
    bo.previous_performances,
    gs.best_performance,
    f"misclassification_{generator}.pdf",
)

# Get times plot
Plotter.get_times_plot(bo.iter_times, f"times_{generator}.pdf")

with open(f"output_bo_{generator}.json", "w") as outfile:
    json.dump(bo.get_json(), outfile, indent=4)
