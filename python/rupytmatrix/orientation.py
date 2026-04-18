"""Orientation-averaging strategies and PDFs.

Port of ``pytmatrix.orientation``. The three ``orient_*`` functions all
take a :class:`~rupytmatrix.scatterer.Scatterer` instance and return the
``(S, Z)`` pair averaged (or not) over the Euler angles ``(alpha, beta)``
according to the scatterer's ``or_pdf``.

The module is pure Python — it calls :meth:`Scatterer.get_SZ_single`
repeatedly with different orientations, relying on the Rust core only
for the per-orientation evaluation.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from scipy.integrate import dblquad, quad


def gaussian_pdf(std: float = 10.0, mean: float = 0.0) -> Callable[[float], float]:
    """Gaussian orientation PDF, spherically-normalised.

    Returns a callable ``pdf(x)`` that evaluates a Gaussian in ``beta``
    (degrees) multiplied by the spherical Jacobian ``sin(beta)``, and
    normalised to integrate to 1 over ``[0, 180]``.
    """
    norm_const = [1.0]

    def pdf(x):
        return (
            norm_const[0]
            * np.exp(-0.5 * ((x - mean) / std) ** 2)
            * np.sin(np.pi / 180.0 * x)
        )

    norm_dev = quad(pdf, 0.0, 180.0)[0]
    norm_const[0] /= norm_dev
    return pdf


def uniform_pdf() -> Callable[[float], float]:
    """Uniform orientation PDF on the unit sphere.

    Returns ``pdf(beta)`` proportional to ``sin(beta)`` and normalised on
    ``[0, 180]``.
    """
    norm_const = [1.0]

    def pdf(x):
        return norm_const[0] * np.sin(np.pi / 180.0 * x)

    norm_dev = quad(pdf, 0.0, 180.0)[0]
    norm_const[0] /= norm_dev
    return pdf


def orient_single(tm) -> Tuple[np.ndarray, np.ndarray]:
    """No averaging — evaluate at the scatterer's ``(alpha, beta)``."""
    return tm.get_SZ_single()


def orient_averaged_adaptive(tm) -> Tuple[np.ndarray, np.ndarray]:
    """Adaptive (scipy.integrate.dblquad) orientation averaging.

    Slow; use ``orient_averaged_fixed`` for production runs. Integrates
    each of the 4 (real, imag) components of ``S`` and all 16 components
    of ``Z`` separately over ``alpha in [0, 360], beta in [0, 180]``,
    weighted by ``tm.or_pdf(beta)``.
    """
    S = np.zeros((2, 2), dtype=complex)
    Z = np.zeros((4, 4))

    def Sfunc(beta, alpha, i, j, real):
        S_ang, _ = tm.get_SZ_single(alpha=alpha, beta=beta)
        s = S_ang[i, j].real if real else S_ang[i, j].imag
        return s * tm.or_pdf(beta)

    for i in range(2):
        for j in range(2):
            S.real[i, j] = dblquad(
                Sfunc, 0.0, 360.0, lambda x: 0.0, lambda x: 180.0, (i, j, True)
            )[0] / 360.0
            S.imag[i, j] = dblquad(
                Sfunc, 0.0, 360.0, lambda x: 0.0, lambda x: 180.0, (i, j, False)
            )[0] / 360.0

    def Zfunc(beta, alpha, i, j):
        _, Z_ang = tm.get_SZ_single(alpha=alpha, beta=beta)
        return Z_ang[i, j] * tm.or_pdf(beta)

    for i in range(4):
        for j in range(4):
            Z[i, j] = dblquad(
                Zfunc, 0.0, 360.0, lambda x: 0.0, lambda x: 180.0, (i, j)
            )[0] / 360.0

    return S, Z


def orient_averaged_fixed(tm) -> Tuple[np.ndarray, np.ndarray]:
    """Fixed-quadrature orientation averaging.

    Alpha is integrated by uniform sampling (``tm.n_alpha`` points); beta
    is integrated by Gaussian quadrature built against ``tm.or_pdf``
    using ``tm.beta_p`` and ``tm.beta_w`` (populated by
    :meth:`Scatterer._init_orient`). Much faster than the adaptive
    variant and accurate enough for practical use.
    """
    S = np.zeros((2, 2), dtype=complex)
    Z = np.zeros((4, 4))
    ap = np.linspace(0, 360, tm.n_alpha + 1)[:-1]
    aw = 1.0 / tm.n_alpha

    for alpha in ap:
        for beta, w in zip(tm.beta_p, tm.beta_w):
            S_ang, Z_ang = tm.get_SZ_single(alpha=alpha, beta=beta)
            S += w * S_ang
            Z += w * Z_ang

    sw = tm.beta_w.sum()
    S *= aw / sw
    Z *= aw / sw

    return S, Z
