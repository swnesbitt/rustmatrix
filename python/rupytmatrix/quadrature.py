"""Gaussian quadrature points/weights for arbitrary weighting functions.

Direct port of ``pytmatrix.quadrature.quadrature`` — used by
:func:`rupytmatrix.orientation.orient_averaged_fixed` to build an
integration rule for the orientation PDF in beta.

Pure Python on top of numpy/scipy; no Rust involvement.
"""

from __future__ import annotations

import numpy as np


def discrete_gautschi(z: np.ndarray, w: np.ndarray, n_iter: int):
    """Discrete Gautschi / Stieltjes procedure.

    Builds the three-term recurrence coefficients (a, b) for the
    orthogonal polynomials with respect to the discrete inner product
    defined by abscissas ``z`` and weights ``w``.
    """
    p = np.ones(z.shape)
    p /= np.sqrt(np.dot(p, p))
    p_prev = np.zeros(z.shape)
    wz = z * w
    a = np.empty(n_iter)
    b = np.empty(n_iter)

    for j in range(n_iter):
        p_norm = np.dot(w * p, p)
        a[j] = np.dot(wz * p, p) / p_norm
        b[j] = 0.0 if j == 0 else p_norm / np.dot(w * p_prev, p_prev)
        p_new = (z - a[j]) * p - b[j] * p_prev
        p_prev = p
        p = p_new

    return a, b[1:]


def get_points_and_weights(w_func=None, left=-1.0, right=1.0, num_points=5, n=4096):
    """Quadrature points and weights for a weighting function ``w(x)``.

    Approximates :math:`\\int_{left}^{right} f(x) w(x) dx \\approx \\sum_i
    w_i f(x_i)` by building the orthogonal-polynomial recurrence for
    ``w`` from a dense midpoint sample, diagonalising the Jacobi matrix,
    and reading off points and weights.

    Args:
        w_func: Weighting function; defaults to the constant 1 (Gauss-Legendre).
        left, right: Integration interval.
        num_points: Number of quadrature points to return.
        n: Number of points used to sample ``w`` when building the
            Jacobi matrix.

    Returns:
        ``(points, weights)`` — both 1-D numpy arrays of length
        ``num_points``, sorted by point.
    """
    if w_func is None:
        w_func = lambda x: np.ones(x.shape)  # noqa: E731

    dx = (float(right) - left) / n
    z = np.linspace(left + 0.5 * dx, right - 0.5 * dx, n)
    w = dx * w_func(z)

    a, b = discrete_gautschi(z, w, num_points)
    alpha = a
    beta = np.sqrt(b)

    J = np.diag(alpha)
    J += np.diag(beta, k=-1)
    J += np.diag(beta, k=1)

    points, v = np.linalg.eigh(J)
    ind = points.argsort()
    points = points[ind]
    weights = v[0, :] ** 2 * w.sum()
    weights = weights[ind]

    return points, weights
