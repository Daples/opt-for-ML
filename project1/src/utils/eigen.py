""" This code is based on the details presented in:
Eraña Robles, G. (2018). Implementing the QR algorithm for efficiently computing matrix
eigenvalues and eigenvectors.
"""

from typing import cast

from scipy import linalg as lin
import numpy as np


eps = 1e-7


def conjugate_transpose(matrix: np.ndarray) -> np.ndarray:
    """Compute the conjugate transpose of a matrix.

    Parameters
    ----------
    matrix: numpy.ndarray
        The input matrix.

    Returns
    -------
    numpy.ndarray
        The conjugate transpose of the input matrix.
    """

    return np.transpose(np.conj(matrix))


def wilkshift(a: float, b: float, c: float, d: float) -> float:
    """Computes the Wilkinson shift of a 2x2 sub-matrix in the form `[[a, b], [c, d]]`.
    Equivalent to the nearest eigenvalue of this matrix to `d`.

    Parameters
    ----------
    a: float
        First element.
    b: float
        Second element.
    c: float
        Third element.
    d: float
        Last element.

    Returns
    -------
    float
        The Wilkinson kappa shift.
    """

    kappa = d
    s = abs(a) + abs(b) + abs(c) + abs(d)

    if s != 0:
        q = (b / s) * (c / s)

        if q != 0:
            p = 0.5 * ((a / s) - (d / s))
            r1 = p * p + q
            if r1 < 0:
                mult = 1j
            else:
                mult = 1

            r = np.sqrt(abs(r1)) * mult
            if np.real(p) * np.real(r) + np.imag(p) * np.imag(r) < 0:
                r = -r

            kappa -= s * (q / (p + r))

    return kappa


def rotation_gen(a: float, b: float) -> tuple[float, float, float, float]:
    """A function that generates a Givens rotation from elements `a` and `b`.

    Parameters
    ----------
    a: float
        The first input scalar.
    b: float
        The second input scalar.

    Returns
    -------
    float
        Overwritten scalar `a`.
    float
        Overwritten scalar `b`.
    float
        The first scalar that defines the plane rotation.
    float
        The second scalar that defines the plane rotation.
    """

    if b == 0:
        c = 1
        s = 0
    elif a == 0:
        c = 0
        s = 1
        a = b
        b = 0
    else:
        mu = a / abs(a)
        tau = abs(np.real(a)) + abs(np.imag(a)) + abs(np.real(b)) + abs(np.imag(b))
        nu = tau * np.sqrt(abs(a / tau) ** 2 + abs(b / tau) ** 2)
        c = abs(a) / nu
        s = mu * np.conj(b) / nu
        a = nu * mu
        b = 0

    return a, b, c, s


def rot_app(
    c: float, s: float, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Application of a plane rotation `[c, s]` on vectors `[x,y]`.

    Parameters
    ----------
    c: float
        The first scalar that defines the plane rotation.
    s: float
        The second scalar that defines the plane rotation.
    x: numpy.ndarray
        The first vector to rotate.
    y: numpy.ndarray
        The second vector to rotate.
    """

    t = c * x + s * y
    y = c * y - np.conj(s) * x
    x = t

    return x, y


def block_process(
    h: np.ndarray, q: np.ndarray, i2: int
) -> tuple[np.ndarray, np.ndarray]:
    """Process the blocks `h` and `q`.

    Parameters
    ----------
    h: numpy.ndarray
        The first block.
    q: numpy.ndarray
        The second block.

    Returns
    -------
    numpy.ndarray
        The processed block `h`.
    numpy.ndarray
        The processed block `q`.
    """

    n = h.shape[0]
    sigma = wilkshift(h[i2 - 1, i2 - 1], h[i2 - 1, i2], h[i2, i2 - 1], h[i2, i2])

    if np.isreal(sigma) or not np.isreal(h[i2 - 1 : i2 + 1, i2 - 1 : i2 + 1]).all():
        h = h - sigma * np.identity(n)
        _, _, c, s = rotation_gen(
            h[i2 - 1, i2 - 1] / lin.norm(h[i2 - 1, i2 - 1]),
            h[i2, i2 - 1] / lin.norm(h[i2 - 1, i2 - 1]),
        )

        h[i2 - 1, :], h[i2, :] = rot_app(c, -s, h[i2 - 1, :], h[i2, :])
        h[: i2 + 1, i2 - 1], h[: i2 + 1, i2] = rot_app(
            c, -np.conj(s), h[: i2 + 1, i2 - 1], h[: i2 + 1, i2]
        )
        q[:, i2 - 1], q[:, i2] = rot_app(c, -np.conj(s), q[:, i2 - 1], q[:, i2])
        h = h + sigma * np.identity(n)

    return h, q


def back_search(
    h: np.ndarray, q: np.ndarray, z: int
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """This function looks for deflating rows on a real Schur form matrix.

    Parameters
    ----------
    h: numpy.ndarray
        The first starting block.
    q: numpy.ndarray
        The second starting block.
    z: int
        The starting index.

    Returns
    -------
    numpy.ndarray
        The first processed block.
    numpy.ndarray
        The second processed block.
    int
        The first index.
    int
        The second index.
    """

    nh = lin.norm(h)
    i1 = z
    i2 = z
    while i1 > 0:
        if abs(h[i1, i1 - 1]) > eps * nh and i2 != 1:
            i1 -= 1
        else:
            # Deflate
            if i2 != 1:
                h[i1, i1 - 1] = 0

            # Check if the block is 2x2 or 1x1
            if i1 == i2 - 1 or i2 == 1:
                # Process the 2x2 block
                h, q = block_process(h, q, i2)

                if i2 != 2:
                    # If the block is complex, jump to the row above
                    i2 = i1 - 1
                    i1 -= 1
                else:
                    # The first block is reached
                    i1 = 0
                    i2 = 0

            # The block is 1x1. Jump to the row above
            elif i1 == i2:
                i2 = i1 - 1
                i1 -= 1

            else:
                break

    return h, q, i1, i2


def householder_gen(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Generation of a Householder transformation. This step can reduce any matrix into
    an upper Hessenberg form.

    Parameters
    ----------
    x: numpy.ndarray
        The input vector.

    Returns
    -------
    numpy.ndarray
        The computed vector that generates a Householder reflection.
    float
        The associated scaling for generating `x`.
    """

    u = x.ravel()
    nu = lin.norm(x)
    if nu == 0:
        u[0] = np.sqrt(2)
        return u, cast(float, nu)

    if u[0] != 0:
        rho = np.conj(u[0]) / abs(u[0])
    else:
        rho = 1

    u = (rho / nu) * u
    u[0] = 1 + u[0]
    u = u / np.sqrt(u[0])
    nu = -(np.conj(rho)) * nu
    return u, nu


def _compute_init_u(*args: float) -> np.ndarray:
    """Computes the first column of
            ``H^2 - 2Re(kappa) *H + |kappa|I``
       which is a vector that generates a Householder reflection.

    Parameters
    ----------
    *args: float
        The collection of values to compute the initial `u`.

    Returns
    -------
    numpy.ndarray
        The computed vector `u`.
    """

    elem = np.array(args)
    s = 1 / np.max(abs(elem))
    elem = s * elem

    p = elem[8] - elem[0]
    q = elem[5] - elem[0]
    r = elem[3] - elem[0]

    c = np.array(
        [[(p * q - elem[7] * elem[6]) / elem[2] + elem[1]], [r - p - q], [elem[4]]]
    )

    u, _ = householder_gen(c)
    return u


def qr_step(
    h: np.ndarray, q: np.ndarray, u: np.ndarray, i1: int, i2: int
) -> tuple[np.ndarray, np.ndarray]:
    """Make a step of the QR method.

    Parameters
    ----------
    h: numpy.ndarray
        A matrix block.
    q: numpy.ndarray
        Another matrix block.
    i1: float
        The first index.
    i2: float
        The second index.

    Returns
    -------
    numpy.ndarray
        The first transformed matrix block.
    numpy.ndarray
        The second transformed matrix block.
    """

    n = h.shape[0]
    u = np.reshape(u, (-1, 1))
    for i in range(i1, i2 - 1):
        j = max(i - 1, i1)
        v = np.transpose(u).dot(h[i : i + 3, j:])
        h[i : i + 3, j:] = np.subtract(h[i : i + 3, j:], u.dot(v))

        iu = min(i + 3, i2)
        v = h[: iu + 1, i : i + 3].dot(u)
        h[: iu + 1, i : i + 3] = h[: iu + 1, i : i + 3] - v.dot(np.transpose(u))

        v = q[:, i : i + 3].dot(u)
        q[:, i : i + 3] = q[:, i : i + 3] - v.dot(conjugate_transpose(u))

        if i1 != i2 - 2:
            u, _ = householder_gen(h[i + 1 : i + 4, i])
            u = np.reshape(u, (-1, 1))
        if i != i1:
            h[i + 1, j] = 0
            h[i + 2, j] = 0

    if i2 > 1:
        h[i2 - 1, i2 - 2], h[i2, i2 - 2], c, s = rotation_gen(
            h[i2 - 1, i2 - 2], h[i2, i2 - 2]
        )

        h[i2 - 1, i2 - 1 :], h[i2, i2 - 1 :] = rot_app(
            c, s, h[i2 - 1, i2 - 1 :], h[i2, i2 - 1 :]
        )

        h[: i2 + 1, i2 - 1], h[: i2 + 1, i2] = rot_app(
            c, s, h[: i2 + 1, i2 - 1], h[: i2 + 1, i2]
        )

        q[:n, i2 - 1], q[:n, i2] = rot_app(c, s, q[:n, i2 - 1], q[:n, i2])
    return h, q


def shift_method(
    h: np.ndarray, q: np.ndarray, n_max: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Applies the implicitly shifted QR iteration to an upper Hessenberg matrix `H` and
    applied the transformation to `Q`.

    Parameters
    ----------
    h: numpy.ndarray
        An upper Hessenberg matrix.
    Q: numpy.ndarray
        A matrix to store the transformations.
    n_max: int, optional
        A maximum number of iterations. Default: 100.

    Raises
    ------
    Timeouterror
        When the maximum number of iterations is reached.
    """

    n = h.shape[0]
    i2 = n - 1
    it = 0
    while i2 > 0:
        if it > n_max:
            raise TimeoutError("Maximum number of iterations reached.")

        old_i2 = i2
        h, q, i1, i2 = back_search(h, q, i2)

        if i2 == old_i2:
            it += 1
        else:
            it = 0

        if i2 > 0:
            u = _compute_init_u(
                h[i1, i1],
                h[i1, i1 + 1],
                h[i1 + 1, i1],
                h[i1 + 1, i1 + 1],
                h[i1 + 2, i1 + 1],
                h[i2 - 1, i2 - 1],
                h[i2 - 1, i2],
                h[i2, i2 - 1],
                h[i2, i2],
            )

            h, q = qr_step(h, q, u, i1, i2)
    return h, q


def hess(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reduces an arbitrary matrix to an upper Hessenberg form by Householder
    transformations.

    Parameters
    ----------
    a: numpy.ndarray
        The input matrix.

    Returns
    -------
    numpy.ndarray
        The upper Hessenberg matrix.
    numpy.ndarray
        The matrix of transformations.
    """

    n = a.shape[0]
    h = a.astype(float)
    q = np.identity(n)
    for k in range(n - 2):
        # Apply the Householder transformation
        u, h[k + 1, k] = householder_gen(h[k + 1 :, k])
        q[k + 1 :, k] = u

        # Multiply transformations on the left
        u = np.reshape(u, (-1, 1))
        v = conjugate_transpose(u).dot(h[k + 1 :, k + 1 :])
        h[k + 1 :, k + 1 :] = np.subtract(h[k + 1 :, k + 1 :], u.dot(v))
        h[k + 2 :, k] = 0

        # Multiply transformations on the right
        v = h[:, k + 1 :].dot(u)
        h[:, k + 1 :] = h[:, k + 1 :] - v.dot(conjugate_transpose(u))

    # Accumulate transformations on matrix `q`
    ind = np.identity(n)
    for k in range(n - 3, -1, -1):
        u = q[k + 1 :, k]
        u = np.reshape(u, (-1, 1))
        v = conjugate_transpose(u).dot(q[k + 1 :, k + 1 :])
        q[k + 1 :, k + 1 :] = q[k + 1 :, k + 1 :] - u.dot(v)
        q[:, k] = ind[:, k]
    return h, q


def clean(matrix: np.ndarray) -> None:
    """It cleans a matrix of values smaller than a tolerance.

    Parameters
    ----------
    matrix: numpy.ndarray
        The input matrix.
    """

    matrix[matrix < 1e-9] = 0


def real_schur(matrix: np.ndarray, n_max=100) -> tuple[np.ndarray, np.ndarray]:
    """It computes the real Schur form for the input matrix.

    Parameters
    ----------
    matrix: numpy.ndarray
        The input matrix.

    Returns
    -------
    numpy.ndarray
        The shifted upper Hessenberg matrix.
    numpy.ndarray
        The matrix of accumulated transformations.
    """

    h, q = hess(matrix)
    t, q = shift_method(h, q, n_max=n_max)
    clean(t)
    clean(q)
    return t, q


def eigenvals(t: np.ndarray) -> list[complex]:
    """Returns the eigenvalues of block diagonal matrix.

    Parameters
    ----------
    t: numpy.ndarray
        The block diagonal matrix.

    Returns
    -------
    list[complex]
        The list of complex numbers.
    """

    def eigenvals2(a: float, b: float, c: float, d: float) -> tuple[complex, complex]:
        """It returns the eigenvalues of a 2x2 matrix block `[[a, b], [c, d]]`.

        Parameters
        ----------
        a: float
            The first element.
        b: float
            The second element.
        c: float
            The third element.
        d: float
            The fourth element.

        Returns
        -------
        complex:
            The first root.
        complex:
            The second conjugate root.
        """

        trace = a + d
        det = a * d - b * c

        disc = trace**2 / 4 - det
        if disc < 0:
            mult = 1j
        else:
            mult = 1

        l11 = trace / 2 + np.sqrt(abs(disc)) * mult
        l21 = trace / 2 - np.sqrt(abs(disc)) * mult
        return l11, l21

    # Iterate over diagonal blocks
    n = t.shape[0]
    i = n - 1
    val = []
    while i >= 0:
        if i - 1 >= 0 and t[i, i - 1] != 0:
            l1, l2 = eigenvals2(t[i - 1, i - 1], t[i - 1, i], t[i, i - 1], t[i, i])
            val.append(l1)
            val.append(l2)
            i -= 2
        else:
            val.append(t[i, i])
            i -= 1

    # Clean real parts lower than threshold
    j = 0
    for num in val:
        if abs(np.real(num)) < 1e-10:
            val[j] = np.imag(num) * 1j

        if abs(np.imag(num)) < 1e-10:
            val[j] = np.real(num)
        j += 1
    return val


def solve_homogeneous(a: np.ndarray) -> np.ndarray:
    """Solves the auxiliary homogeneous system of linear equations.

    Parameters
    ----------
    a: numpy.ndarray
        The input coefficients matrix.

    Returns
    ------s
    """
    n = a.shape[0]

    _, u = lin.lu(a, permute_l=True)  # type: ignore

    # Make it a solvable problem
    b = np.zeros((n - 1, 1), dtype=complex)
    for j in range(n - 1):
        b[j, 0] = -u[j, n - 1]

    aeq = u[: n - 1, : n - 1].astype(complex)

    sol = lin.solve(aeq, b)
    vec = np.zeros((n, 1), dtype=complex)
    vec[n - 1, 0] = 1
    for j in range(n - 1):
        vec[j, 0] = sol[j, 0]

    vec = vec / lin.norm(vec)
    return vec


def eigenvecs(a: np.ndarray, eigenvals: np.ndarray) -> np.ndarray:
    """Returns the eigenvectors of `a` given the eigenvalues.

    Parameters
    ----------
    a: numpy.ndarray
        The matrix.
    eigenvals: numpy.ndarray
        The array of eigenvalues.

    Returns
    -------
    numpy.ndarray
        A matrix whose columns are the eigenvectors of `a`.
    """

    n = a.shape[0]
    vecs = np.zeros((n, n), dtype=complex)
    for i, val in enumerate(eigenvals):
        a1 = a - val * np.identity(n)
        vec = solve_homogeneous(a1)
        vecs[:, i] = np.reshape(vec, (1, -1))

    return vecs


def eig(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues and eigen vectors of an input matrix.

    Parameters
    ----------
    matrix: numpy.ndarray
        An arbitrary input matrix.

    Returns
    -------
    numpy.ndarray
        The array of eigenvalues.
    numpy.ndarray
        The matrix with the eigenvectors on its columns.
    """

    t, _ = real_schur(matrix)
    eig_vals = np.diag(eigenvals(t))
    vecs = eigenvecs(matrix, eig_vals)

    return eig_vals, vecs
