from typing import Callable

import numpy as np


def incremental_search(
    f: Callable[[float], float],
    max_step: float,
    starting_point: float = 1e-18,
    step: float = 0.01,
    max_iter_equal: int = 50,
) -> tuple[float, float]:
    """It uses accelerated incremental search to find the upper limit where a local
    maximum is contained.

    Parameters
    ----------
    f: (float) -> float
        The objective function.
    max_step: float
        The maximum possible step in the gradient direction.
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

    # Initialize
    lower = 0
    upper = starting_point
    prev_val = f(starting_point)
    i = 1
    equal_counter = 0
    shift = lambda i: (i - 1) * step

    # Start search
    while shift(i) < max_step:
        # Update point
        current_point = starting_point + shift(i)
        current_val = f(current_point)

        # Check for local maximum
        if current_val < prev_val:
            break

        # Counter if the function is constant (avoid infinite loops)
        elif current_val == prev_val:
            equal_counter += 1
        if equal_counter > max_iter_equal:
            break

        prev_val = current_val
        i += 1

    lower = starting_point + shift(i - 1)
    upper = starting_point + shift(i + 1)
    return lower, upper


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

    # Estimate number of iterations
    ratio = 2 / (1 + np.sqrt(5))
    n = np.ceil(np.log(tol) / np.log(ratio))

    # Initialize values
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
