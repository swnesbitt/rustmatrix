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


def test_psd_integration_parity(PyScatterer):
    """PSD-integrated S and Z should match pytmatrix for a gamma PSD.

    Mirrors pytmatrix's own ``test_psd`` — a sphere at lambda=6.5 with
    m=1.5+0.5j and a GammaPSD(D0=1.0, Nw=1e3, mu=4) sampled over [0, 10].
    """
    from pytmatrix import psd as py_psd

    from rupytmatrix import psd as rs_psd

    geom = (90.0, 90.0, 0.0, 180.0, 0.0, 0.0)
    m = complex(1.5, 0.5)

    py = PyScatterer(wavelength=6.5, m=m, axis_ratio=1.0)
    py.set_geometry(geom)
    py.psd_integrator = py_psd.PSDIntegrator()
    py.psd_integrator.num_points = 64
    py.psd_integrator.D_max = 10.0
    py.psd = py_psd.GammaPSD(D0=1.0, Nw=1e3, mu=4)
    py.psd_integrator.init_scatter_table(py)

    rs = RsScatterer(wavelength=6.5, m=m, axis_ratio=1.0)
    rs.set_geometry(geom)
    rs.psd_integrator = rs_psd.PSDIntegrator()
    rs.psd_integrator.num_points = 64
    rs.psd_integrator.D_max = 10.0
    rs.psd = rs_psd.GammaPSD(D0=1.0, Nw=1e3, mu=4)
    rs.psd_integrator.init_scatter_table(rs)

    _compare(py, rs, s_tol=1e-3, z_tol=1e-3)


def test_radar_observables_backscatter_parity(PyScatterer):
    """Polarimetric radar observables must match pytmatrix for an oblate drop.

    Uses an X-band water drop (m ≈ 7.94 + 2.33j at 10 °C) shaped like a
    2 mm equi-volume raindrop. Exercises the radar.py layer end-to-end:
    Zdr, rho_hv, delta_hv — all derived from Z in the backscatter geometry.
    """
    from pytmatrix import radar as py_radar

    from rupytmatrix import radar as rs_radar
    from rupytmatrix.tmatrix_aux import dsr_thurai_2007, geom_horiz_back, wl_X

    m = complex(7.94, 2.33)
    D_eq = 2.0  # mm
    axis_ratio = 1.0 / dsr_thurai_2007(D_eq)  # Scatterer takes h/v

    py = PyScatterer(radius=D_eq / 2.0, wavelength=wl_X, m=m,
                     axis_ratio=axis_ratio, ddelt=1e-4, ndgs=2)
    py.set_geometry(geom_horiz_back)

    rs = RsScatterer(radius=D_eq / 2.0, wavelength=wl_X, m=m,
                     axis_ratio=axis_ratio, ddelt=1e-4, ndgs=2)
    rs.set_geometry(geom_horiz_back)

    # Radar cross-sections (H and V), Zdr, rho_hv, delta_hv.
    np.testing.assert_allclose(
        rs_radar.radar_xsect(rs, h_pol=True),
        py_radar.radar_xsect(py, h_pol=True),
        rtol=5e-3,
    )
    np.testing.assert_allclose(
        rs_radar.radar_xsect(rs, h_pol=False),
        py_radar.radar_xsect(py, h_pol=False),
        rtol=5e-3,
    )
    np.testing.assert_allclose(rs_radar.Zdr(rs), py_radar.Zdr(py), rtol=5e-3)
    np.testing.assert_allclose(
        rs_radar.rho_hv(rs), py_radar.rho_hv(py), rtol=5e-3, atol=5e-3,
    )
    # delta_hv is tiny for rain at X-band; compare with an atol instead
    # of rtol since it passes through zero.
    np.testing.assert_allclose(
        rs_radar.delta_hv(rs), py_radar.delta_hv(py), atol=5e-3,
    )


def test_radar_observables_forward_parity(PyScatterer):
    """Kdp and Ai (forward geometry) must match pytmatrix."""
    from pytmatrix import radar as py_radar

    from rupytmatrix import radar as rs_radar
    from rupytmatrix.tmatrix_aux import dsr_thurai_2007, geom_horiz_forw, wl_X

    m = complex(7.94, 2.33)
    D_eq = 2.0
    axis_ratio = 1.0 / dsr_thurai_2007(D_eq)

    py = PyScatterer(radius=D_eq / 2.0, wavelength=wl_X, m=m,
                     axis_ratio=axis_ratio, ddelt=1e-4, ndgs=2)
    py.set_geometry(geom_horiz_forw)

    rs = RsScatterer(radius=D_eq / 2.0, wavelength=wl_X, m=m,
                     axis_ratio=axis_ratio, ddelt=1e-4, ndgs=2)
    rs.set_geometry(geom_horiz_forw)

    np.testing.assert_allclose(rs_radar.Kdp(rs), py_radar.Kdp(py), rtol=5e-3)
    np.testing.assert_allclose(
        rs_radar.Ai(rs, h_pol=True), py_radar.Ai(py, h_pol=True), rtol=5e-3,
    )
    np.testing.assert_allclose(
        rs_radar.Ai(rs, h_pol=False), py_radar.Ai(py, h_pol=False), rtol=5e-3,
    )


def test_refractive_mg_bruggeman_parity():
    """Maxwell-Garnett and Bruggeman EMAs are numeric ports — should match bit-for-bit."""
    py_refr = pytest.importorskip("pytmatrix.refractive")

    from rupytmatrix import refractive as rs_refr

    m_ice = complex(1.78, 2e-3)
    m_air = complex(1.0, 0.0)
    m_water = complex(7.94, 2.33)

    # Two-component mixes (ice in air, water in air).
    for m_inc, frac in [(m_ice, 0.3), (m_water, 0.1), (m_ice, 0.6)]:
        np.testing.assert_allclose(
            rs_refr.mg_refractive((m_air, m_inc), (1 - frac, frac)),
            py_refr.mg_refractive((m_air, m_inc), (1 - frac, frac)),
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            rs_refr.bruggeman_refractive((m_air, m_inc), (1 - frac, frac)),
            py_refr.bruggeman_refractive((m_air, m_inc), (1 - frac, frac)),
            rtol=1e-10,
        )

    # Three-component MG (air + ice + water).
    np.testing.assert_allclose(
        rs_refr.mg_refractive((m_air, m_ice, m_water), (0.5, 0.3, 0.2)),
        py_refr.mg_refractive((m_air, m_ice, m_water), (0.5, 0.3, 0.2)),
        rtol=1e-10,
    )


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


def test_psd_integration_orient_averaged_parity(PyScatterer):
    """PSD + orientation-averaged tabulation should match pytmatrix.

    Exercises the Rust tabulate_scatter_table_orient_avg fast path by
    combining a GammaPSD with orient_averaged_fixed (gaussian PDF). If the
    Rust per-diameter (alpha, beta) loop diverges from the Python
    reference we'd see it here since S and Z are integrated over both
    orientation and size.
    """
    from pytmatrix import orientation as py_orient, psd as py_psd

    from rupytmatrix import orientation as rs_orient, psd as rs_psd

    geom = (90.0, 90.0, 0.0, 180.0, 0.0, 0.0)
    m = complex(1.5, 0.01)

    py = PyScatterer(wavelength=6.283185307, m=m, axis_ratio=2.0,
                     ddelt=1e-4, ndgs=2)
    py.set_geometry(geom)
    py.or_pdf = py_orient.gaussian_pdf(std=20.0, mean=90.0)
    py.orient = py_orient.orient_averaged_fixed
    py.n_alpha = 4
    py.n_beta = 8
    py.psd_integrator = py_psd.PSDIntegrator()
    py.psd_integrator.num_points = 16
    py.psd_integrator.D_max = 4.0
    py.psd = py_psd.GammaPSD(D0=1.0, Nw=1e3, mu=4)
    py.psd_integrator.init_scatter_table(py)

    rs = RsScatterer(wavelength=6.283185307, m=m, axis_ratio=2.0,
                     ddelt=1e-4, ndgs=2)
    rs.set_geometry(geom)
    rs.or_pdf = rs_orient.gaussian_pdf(std=20.0, mean=90.0)
    rs.orient = rs_orient.orient_averaged_fixed
    rs.n_alpha = 4
    rs.n_beta = 8
    rs.psd_integrator = rs_psd.PSDIntegrator()
    rs.psd_integrator.num_points = 16
    rs.psd_integrator.D_max = 4.0
    rs.psd = rs_psd.GammaPSD(D0=1.0, Nw=1e3, mu=4)
    rs.psd_integrator.init_scatter_table(rs)

    _compare(py, rs, s_tol=5e-3, z_tol=5e-3)


def test_psd_integration_angular_parity(PyScatterer):
    """PSD tabulation with angular_integration=True should match pytmatrix.

    Exercises ``tabulate_scatter_table_with_angular``: per-diameter
    sca_xsect / ext_xsect / asym are integrated in Rust on a Gauss-Legendre
    (θ, φ) grid; pytmatrix uses scipy.dblquad per diameter. Compare the
    populated ``_angular_table`` entries element-wise.
    """
    from pytmatrix import psd as py_psd

    from rupytmatrix import psd as rs_psd

    geom = (90.0, 90.0, 0.0, 180.0, 0.0, 0.0)
    m = complex(1.5, 0.01)
    kw = dict(wavelength=6.283185307, m=m, axis_ratio=1.0,
              ddelt=1e-4, ndgs=2)

    py = PyScatterer(**kw)
    py.set_geometry(geom)
    py.psd_integrator = py_psd.PSDIntegrator()
    py.psd_integrator.num_points = 8
    py.psd_integrator.D_max = 4.0
    py.psd = py_psd.GammaPSD(D0=1.0, Nw=1e3, mu=4)
    py.psd_integrator.init_scatter_table(py, angular_integration=True)

    rs = RsScatterer(**kw)
    rs.set_geometry(geom)
    rs.psd_integrator = rs_psd.PSDIntegrator()
    rs.psd_integrator.num_points = 8
    rs.psd_integrator.D_max = 4.0
    rs.psd = rs_psd.GammaPSD(D0=1.0, Nw=1e3, mu=4)
    rs.psd_integrator.init_scatter_table(rs, angular_integration=True)

    for key in ("sca_xsect", "ext_xsect", "asym"):
        for pol in ("h_pol", "v_pol"):
            p = py.psd_integrator._angular_table[key][pol][geom]
            r = rs.psd_integrator._angular_table[key][pol][geom]
            np.testing.assert_allclose(r, p, rtol=1e-3, atol=1e-6)


def test_psd_integration_orient_adaptive_parity(PyScatterer):
    """PSD + orient_averaged_adaptive should match pytmatrix within 5e-3.

    The Rust path uses a dense uniform-α × Gauss-Legendre-β grid with
    ``or_pdf(β)`` folded into the weights, dispatched through the existing
    orientation-averaged tabulator. Pytmatrix calls scipy.dblquad 24 times
    per diameter — this test is slow on the reference side, so keeps
    ``num_points`` small.
    """
    from pytmatrix import orientation as py_orient, psd as py_psd

    from rupytmatrix import orientation as rs_orient, psd as rs_psd

    geom = (90.0, 90.0, 0.0, 180.0, 0.0, 0.0)
    m = complex(1.5, 0.01)
    kw = dict(wavelength=6.283185307, m=m, axis_ratio=1.5,
              ddelt=1e-4, ndgs=2)

    py = PyScatterer(**kw)
    py.set_geometry(geom)
    py.or_pdf = py_orient.gaussian_pdf(std=20.0, mean=90.0)
    py.orient = py_orient.orient_averaged_adaptive
    py.psd_integrator = py_psd.PSDIntegrator()
    py.psd_integrator.num_points = 4
    py.psd_integrator.D_max = 3.0
    py.psd = py_psd.GammaPSD(D0=1.0, Nw=1e3, mu=4)
    py.psd_integrator.init_scatter_table(py)

    rs = RsScatterer(**kw)
    rs.set_geometry(geom)
    rs.or_pdf = rs_orient.gaussian_pdf(std=20.0, mean=90.0)
    rs.orient = rs_orient.orient_averaged_adaptive
    rs.psd_integrator = rs_psd.PSDIntegrator()
    rs.psd_integrator.num_points = 4
    rs.psd_integrator.D_max = 3.0
    rs.psd = rs_psd.GammaPSD(D0=1.0, Nw=1e3, mu=4)
    rs.psd_integrator.init_scatter_table(rs)

    _compare(py, rs, s_tol=5e-3, z_tol=5e-3)
