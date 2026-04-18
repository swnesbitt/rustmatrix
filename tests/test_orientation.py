"""Orientation-averaging unit tests.

Exercises the three ``orient_*`` strategies and the quadrature helper
without requiring pytmatrix. Parity against the Fortran pytmatrix lives
in ``test_parity_pytmatrix.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from rupytmatrix import Scatterer, orientation
from rupytmatrix.quadrature import get_points_and_weights


@pytest.fixture
def spheroid():
    """A small oblate spheroid used as a realistic test case."""
    s = Scatterer(
        radius=1.0,
        wavelength=6.283185307,
        axis_ratio=1.5,
        m=complex(1.5, 0.01),
        ddelt=1e-3,
        ndgs=2,
    )
    s.set_geometry((90.0, 90.0, 0.0, 180.0, 0.0, 0.0))
    return s


def test_orient_single_matches_default(spheroid):
    """``get_SZ`` with default orient == orient_single matches get_SZ_single."""
    S_s, Z_s = spheroid.get_SZ_single()
    S, Z = spheroid.get_SZ()
    np.testing.assert_allclose(S, S_s)
    np.testing.assert_allclose(Z, Z_s)


def test_gaussian_pdf_is_normalised():
    pdf = orientation.gaussian_pdf(std=15.0, mean=20.0)
    from scipy.integrate import quad

    total, _ = quad(pdf, 0.0, 180.0)
    assert total == pytest.approx(1.0, rel=1e-6)


def test_uniform_pdf_is_normalised():
    pdf = orientation.uniform_pdf()
    from scipy.integrate import quad

    total, _ = quad(pdf, 0.0, 180.0)
    assert total == pytest.approx(1.0, rel=1e-6)


def test_get_points_and_weights_gauss_legendre():
    """Uniform weight on [-1, 1] reproduces classical Gauss-Legendre."""
    pts, ws = get_points_and_weights(num_points=5)
    # 5-point Gauss-Legendre nodes/weights are well-known constants.
    expected_pts = np.array([-0.9061798, -0.5384693, 0.0, 0.5384693, 0.9061798])
    expected_ws = np.array([0.2369269, 0.4786287, 0.5688889, 0.4786287, 0.2369269])
    np.testing.assert_allclose(pts, expected_pts, atol=1e-4)
    np.testing.assert_allclose(ws, expected_ws, atol=1e-4)


def test_get_points_and_weights_integrates_polynomial():
    """A 5-point rule integrates polynomials up to degree 9 exactly."""
    pts, ws = get_points_and_weights(left=0.0, right=1.0, num_points=5)
    # ∫_0^1 x^8 dx = 1/9
    approx = np.sum(ws * pts ** 8)
    assert approx == pytest.approx(1.0 / 9.0, rel=1e-6)


def test_sphere_orient_averaged_fixed_equals_single(spheroid):
    """For a sphere (axis_ratio=1) orientation averaging is a no-op."""
    sph = Scatterer(
        radius=1.0,
        wavelength=6.283185307,
        axis_ratio=1.0,
        m=complex(1.33, 0.0),
        ddelt=1e-3,
        ndgs=2,
    )
    sph.set_geometry((90.0, 90.0, 0.0, 180.0, 0.0, 0.0))
    S_ref, Z_ref = sph.get_SZ_single()

    sph.orient = orientation.orient_averaged_fixed
    sph.n_alpha = 3
    sph.n_beta = 4
    S, Z = sph.get_SZ()
    np.testing.assert_allclose(S, S_ref, atol=1e-8)
    np.testing.assert_allclose(Z, Z_ref, atol=1e-8)


def test_orient_averaged_fixed_returns_right_shapes(spheroid):
    spheroid.orient = orientation.orient_averaged_fixed
    spheroid.n_alpha = 3
    spheroid.n_beta = 5
    S, Z = spheroid.get_SZ()
    assert S.shape == (2, 2)
    assert S.dtype == np.complex128
    assert Z.shape == (4, 4)
    assert Z.dtype == np.float64


@pytest.mark.slow
def test_orient_averaged_adaptive_matches_fixed(spheroid):
    """Fixed and adaptive should roughly agree. Uses a wide PDF so both
    quadratures resolve the integrand; tolerances loose because the two
    methods approximate the same integral very differently."""
    spheroid.or_pdf = orientation.uniform_pdf()
    spheroid.n_alpha = 5
    spheroid.n_beta = 20

    spheroid.orient = orientation.orient_averaged_fixed
    S_fix, Z_fix = spheroid.get_SZ()

    spheroid.orient = orientation.orient_averaged_adaptive
    S_ad, Z_ad = spheroid.get_SZ()

    # Diagonal dominates; absolute tolerance matches Z-magnitude scale.
    np.testing.assert_allclose(S_ad, S_fix, atol=5e-3, rtol=5e-2)
    np.testing.assert_allclose(Z_ad, Z_fix, atol=5e-3, rtol=5e-2)
