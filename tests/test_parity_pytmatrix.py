"""Parity tests against the original pytmatrix (Fortran backend).

Skipped automatically when pytmatrix is not installed. The tests walk a
matrix of shapes (sphere, spheroid, cylinder) at several size parameters
and refractive indices, and assert that the amplitude matrix `S` and the
phase matrix `Z` match to a tight tolerance.

NOTE: tight parity on spheroids and cylinders is gated on the correctness
of the T-matrix port in `src/tmatrix.rs`. Until verified, the relevant
tests are marked xfail. Remove the xfail marker once you've shaken out
any sign / index-convention bugs. The sphere case (axis_ratio = 1) is
stable because it reduces to Mie theory.
"""

from __future__ import annotations

import numpy as np
import pytest

from rupytmatrix import Scatterer as RsScatterer

pytestmark = pytest.mark.parity


@pytest.fixture(scope="module")
def PyScatterer():
    pytmatrix = pytest.importorskip("pytmatrix.tmatrix")
    return pytmatrix.Scatterer


def _compare(py, rs, s_tol=1e-4, z_tol=1e-4):
    S_ref, Z_ref = py.get_SZ()
    S_got, Z_got = rs.get_SZ()
    np.testing.assert_allclose(S_got, S_ref, rtol=s_tol, atol=s_tol)
    np.testing.assert_allclose(Z_got, Z_ref, rtol=z_tol, atol=z_tol)


@pytest.mark.parametrize(
    "radius, wavelength, mrr, mri",
    [
        (0.5, 6.283185307, 1.33, 0.0),
        (1.0, 6.283185307, 1.33, 0.01),
        (2.0, 6.283185307, 1.5, 0.001),
    ],
)
def test_sphere_parity(PyScatterer, radius, wavelength, mrr, mri):
    geom = (90.0, 90.0, 0.0, 180.0, 0.0, 0.0)
    py = PyScatterer(radius=radius, wavelength=wavelength, axis_ratio=1.0,
                     m=complex(mrr, mri), ddelt=1e-4, ndgs=2)
    py.set_geometry(geom)
    rs = RsScatterer(radius=radius, wavelength=wavelength, axis_ratio=1.0,
                     m=complex(mrr, mri), ddelt=1e-4, ndgs=2)
    rs.set_geometry(geom)
    _compare(py, rs, s_tol=1e-3, z_tol=1e-3)


@pytest.mark.parametrize(
    "radius, wavelength, axis_ratio",
    [
        (1.0, 6.283185307, 0.5),   # prolate
        (1.0, 6.283185307, 2.0),   # oblate
        (1.5, 6.283185307, 1.5),
    ],
)
def test_spheroid_parity(PyScatterer, radius, wavelength, axis_ratio):
    geom = (90.0, 90.0, 0.0, 180.0, 0.0, 0.0)
    m = complex(1.5, 0.01)
    py = PyScatterer(radius=radius, wavelength=wavelength, axis_ratio=axis_ratio,
                     m=m, ddelt=1e-4, ndgs=2)
    py.set_geometry(geom)
    rs = RsScatterer(radius=radius, wavelength=wavelength, axis_ratio=axis_ratio,
                     m=m, ddelt=1e-4, ndgs=2)
    rs.set_geometry(geom)
    _compare(py, rs, s_tol=5e-3, z_tol=5e-3)


def test_cylinder_parity(PyScatterer):
    geom = (90.0, 90.0, 0.0, 180.0, 0.0, 0.0)
    m = complex(1.5, 0.01)
    py = PyScatterer(radius=1.0, wavelength=6.283185307, axis_ratio=0.7,
                     shape=PyScatterer.SHAPE_CYLINDER, m=m, ddelt=1e-4, ndgs=2)
    py.set_geometry(geom)
    rs = RsScatterer(radius=1.0, wavelength=6.283185307, axis_ratio=0.7,
                     shape=RsScatterer.SHAPE_CYLINDER, m=m, ddelt=1e-4, ndgs=2)
    rs.set_geometry(geom)
    _compare(py, rs, s_tol=5e-3, z_tol=5e-3)


def test_orient_averaged_fixed_parity(PyScatterer):
    """Fixed-quadrature orientation averaging should match pytmatrix's."""
    from pytmatrix import orientation as py_orient

    from rupytmatrix import orientation as rs_orient

    geom = (90.0, 90.0, 0.0, 180.0, 0.0, 0.0)
    m = complex(1.5, 0.01)

    py = PyScatterer(radius=1.0, wavelength=6.283185307, axis_ratio=2.0,
                     m=m, ddelt=1e-4, ndgs=2)
    py.set_geometry(geom)
    py.or_pdf = py_orient.gaussian_pdf(std=20.0, mean=90.0)
    py.orient = py_orient.orient_averaged_fixed
    py.n_alpha = 4
    py.n_beta = 8

    rs = RsScatterer(radius=1.0, wavelength=6.283185307, axis_ratio=2.0,
                     m=m, ddelt=1e-4, ndgs=2)
    rs.set_geometry(geom)
    rs.or_pdf = rs_orient.gaussian_pdf(std=20.0, mean=90.0)
    rs.orient = rs_orient.orient_averaged_fixed
    rs.n_alpha = 4
    rs.n_beta = 8

    _compare(py, rs, s_tol=5e-3, z_tol=5e-3)
