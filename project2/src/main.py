import json

import numpy as np

from hyper.bayesian import BayesianOptimizer
from utils import get_data

X, y = get_data("/home/daples/git/opt-for-ML/project2/src/heart.csv")
names = ["C", "gamma"]
intervals = np.array([[0, 9], [-10, 0]])
generator = 12345

bo = BayesianOptimizer(names, intervals, generator)

bo.optimize(X, y, n_iter=100)  # type: ignore
with open("output.json", "w") as outfile:
    json.dump(bo.get_json(), outfile, indent=4)
