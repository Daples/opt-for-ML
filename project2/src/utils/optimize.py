from typing import Callable

import numpy as np


def incremental_search(
    f: Callable[[float], float],
    starting_point: float = 0,
    step: float = 1e-5,
) -> tuple[float, float]:
    """It uses accelerated incremental search to find the upper limit where the maximum is
    contained.

    Parameters
    ----------
    f: (float) -> float
        The objective function.
    starting_point: float, optional
        The starting point of the incremental search.
    step: float, optional
        The step size for incremental search.
    """

    lower = 0
    upper = starting_point
    prev_val = f(starting_point)
    i = 1
    shift = lambda i: starting_point + i * step
    while True:
        current_point = shift(i)
        print(current_point)
        current_val = f(current_point)
        if current_val < prev_val:
            lower = shift(i - 1)
            upper = shift(i + 1)
            break
        prev_val = current_val
        i += 1

    return lower, upper


def backtracking(f: Callable, grad_f: Callable, beta, x0) -> float:
    """Minimization"""

    x = x0
    step = 1
    while True:
        grad = grad_f(x)
        if f(x - step * grad) <= f(x) - step / 2 * (grad.dot(grad)):
            break
        step = beta * step
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
    f: Callable[[np.ndarray], float], point: np.ndarray, delta: float = 1
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
        The gradient.
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
