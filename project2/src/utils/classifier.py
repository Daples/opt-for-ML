import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


class ClassificationModel:
    """A wrapper for sklearn's SVC model (mostly for safe typing reasons).

    Attributes
    ----------
    model: sklearn.svm.SVC
        The Support Vector Classifier object.
    k_cross_val: int
        The number of folds for cross validation.
    """

    def __init__(self, model: SVC, k_cross_val: int = 5) -> None:
        self.model: SVC = model
        self.k_cross_val: int = k_cross_val

    def eval(
        self,
        X: np.ndarray,
        y: np.ndarray,
        hyperparameters: np.ndarray,
        names: list[str],
    ) -> float:
        """It evaluates the performance of the model on the given hyperparameters.

        Parameters
        ----------
        X: numpy.ndarray
            The dataset.
        y: numpy.ndarray
            The data labels.
        hyperparameters: numpy.ndarray
            The hyperparameters to evaluate the model on.
        names: list[str]
            The hyperparameters names.

        Returns
        -------
        float
            The model performance.
        """

        hyperparameters = np.power(10, hyperparameters)
        values = hyperparameters.tolist()
        self.model = self.model.set_params(**dict(list(zip(names, values))))
        scores = cross_val_score(
            self.model, X, y, cv=self.k_cross_val, scoring="accuracy"
        )
        return scores.max()
