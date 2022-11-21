import numpy as np


def generate_circles(
    n: int, rs: list[float], seed: int | np.random.Generator = 12345
) -> np.ndarray:
    generator = np.random.default_rng(seed)

    f = lambda x, r: np.sqrt(r**2 - x**2) + generator.normal(0, 0.2)
    datas = []
    for r in rs:
        xs = np.linspace(-r, r, int(n / 4))

        data1 = np.zeros((2 * xs.size, 2))
        data1[:, 0] = np.hstack((xs, xs))

        for i in range(xs.size):
            data1[i, 1] = f(xs[i], r)
            data1[i + xs.size - 1, 1] = -f(xs[i], r)
        datas.append(data1)

    return np.vstack(datas)
