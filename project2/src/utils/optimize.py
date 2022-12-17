from typing import Callable

import numpy as np


def incremental_search(
    f: Callable[[float], float],
    max_step: float,
    starting_point: float = 1e-18,
    step: float = 0.05,
) -> tuple[float, float]:
    """It uses accelerated incremental search to find the upper limit where a local
    maximum is contained.

    Parameters
    ----------
    f: (float) -> float
        The objective function.
    max_step: float
        The maximum possible step for the gradient descent.
    starting_point: float, optional
        The starting point of the incremental search.
    step: float, optional
        The step size for incremental search.

    Returns
    -------
    float
        The lower bound of the estimated interval.
    float
        The upper bound of the estimated interval.
    """

    lower = 0
    upper = starting_point
    prev_val = f(starting_point)
    i = 1
    shift = lambda i: (i - 1) * step
    print("Running incremental search...")
    while shift(i) < max_step:
        current_point = starting_point + shift(i)
        current_val = f(current_point)
        if current_val < prev_val:
            break
        prev_val = current_val
        i += 1
    lower = starting_point + shift(i - 1)
    upper = starting_point + shift(i + 1)
    print("Finished incremental search")

    return lower, upper


def backtracking(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    beta: float,
    x: np.ndarray,
) -> float:
    """Optimization of the step size via backtracking.

    Parameters
    ----------
    f: (numpy.ndarray) -> float
        The objective function.
    grad_f: (numpy.ndarray) -> np.ndarray
        The gradient of the objective function.
    beta: float
        The step scaling parameter.
    x: numpy.ndarray
        The point to evaluate the objective function at.

    Returns
    -------
    float
        The estimated step size.
    """

    step = 1
    i = 1
    while True:
        grad = grad_f(x)
        if f(x - step * grad) <= f(x) - step * 1 / 2 * (grad.dot(grad)):
            break
        step = beta * step
        i += 1
    return step


def golden_section_search(
    f: Callable[[float], float], bounds: tuple[float, float], tol: float = 1e-7
) -> tuple[float, float]:
    """It uses Golden section search to shrink the interval where the maximum is
    contained.

    Parameters
    ----------
    f: (float) -> float
        The objective function.
    bounds: tuple[float, float]
        The interval where a local maximum is guaranteed.
    tol: float
        The relative width of the resulting interval.

    Returns
    -------
    float
        The lower bound of the estimated interval.
    float
        The upper bound of the estimated interval.
    """

    lower, upper = bounds
    if not lower < upper:
        raise ValueError("Invalid interval.")

    ratio = 2 / (1 + np.sqrt(5))
    n = np.ceil(np.log(tol) / np.log(ratio))
    x1 = lower + (1 - ratio) * (upper - lower)
    x2 = upper - (1 - ratio) * (upper - lower)
    i = 1
    while i <= n:
        if f(x1) < f(x2):
            lower = x1
            x1 = x2
            x2 = upper - (1 - ratio) * (upper - lower)
        else:
            upper = x2
            x2 = x1
            x1 = lower + (1 - ratio) * (upper - lower)
        i += 1
    return (lower, upper)


def gradient(
    f: Callable[[np.ndarray], float], point: np.ndarray, delta: float = 1e-9
) -> np.ndarray:
    """A finite difference gradient approximation.

    Parameters
    ----------
    f: (numpy.ndarray) -> float
        The function to calculate the gradient of.
    point: numpy.ndarray
        The point to evaluate the gradient at.
    delta: float
        The step to make on each direction.

    Returns
    -------
    numpy.ndarray
        The gradient vector.
    """

    n = point.shape[0]
    grad = np.zeros(n)
    f_val = f(point)

    for i in range(n):
        shift = np.zeros(n)
        shift[i] = delta
        shifted_f = f(point + shift)
        grad[i] = (shifted_f - f_val) / delta
    return grad


def project(point: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    """It projects a point inside the input intervals.

    Parameters
    ----------
    point: numpy.ndarray
        The point in question.
    intervals: numpy.ndarray
        The domain intervals.

    Returns
    -------
    numpy.ndarray
        The projected point.
    """

    dim = point.shape[0]
    for i in range(dim):
        lower = intervals[i, 0]
        upper = intervals[i, 1]
        if point[i] < lower:
            point[i] = lower
        elif point[i] > upper:
            point[i] = upper
    return point
