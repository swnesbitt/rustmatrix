"""Unit tests for PSD classes and PSDIntegrator.

These exercise the pure-Python size-distribution layer without requiring
pytmatrix. End-to-end parity against pytmatrix is in
``test_parity_pytmatrix.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from rupytmatrix import Scatterer
from rupytmatrix.psd import (
    BinnedPSD,
    ExponentialPSD,
    GammaPSD,
    PSDIntegrator,
    UnnormalizedGammaPSD,
)


# ---------- PSD classes ----------

def test_exponential_psd_scalar_and_array():
    psd = ExponentialPSD(N0=1000.0, Lambda=2.0, D_max=5.0)
    assert psd(1.0) == pytest.approx(1000.0 * np.exp(-2.0))
    # Beyond D_max -> zero.
    assert psd(6.0) == 0.0
    arr = psd(np.array([0.5, 2.0, 6.0]))
    np.testing.assert_allclose(
        arr, np.array([1000.0 * np.exp(-1.0), 1000.0 * np.exp(-4.0), 0.0])
    )


def test_exponential_psd_integrates_to_N0_over_Lambda():
    """Integral of N0 exp(-Lambda D) over [0, inf) == N0/Lambda. With a
    generous D_max the truncated integral should be close."""
    psd = ExponentialPSD(N0=1000.0, Lambda=1.0, D_max=20.0)
    total, _ = quad(psd, 0.0, 20.0)
    assert total == pytest.approx(1000.0, rel=1e-6)


def test_unnormalized_gamma_psd_zero_at_D0():
    psd = UnnormalizedGammaPSD(N0=1.0, Lambda=1.0, mu=3.0, D_max=10.0)
    # D=0 should be handled without error and return 0.
    assert psd(0.0) == 0.0
    arr = psd(np.array([0.0, 1.0, 2.0]))
    assert arr[0] == 0.0
    assert arr[1] == pytest.approx(1.0 * np.exp(-1.0))  # 1^3 * exp(-1)


def test_gamma_psd_normalisation_constant():
    """For mu=0, f(mu) = 6/3.67^4 * 3.67^4 / Gamma(4) = 6/6 = 1, so Nw = nf."""
    psd = GammaPSD(D0=1.0, Nw=1000.0, mu=0.0, D_max=10.0)
    assert psd.nf == pytest.approx(1000.0, rel=1e-10)


def test_gamma_psd_equality():
    a = GammaPSD(D0=1.0, Nw=1.0, mu=2.0, D_max=5.0)
    b = GammaPSD(D0=1.0, Nw=1.0, mu=2.0, D_max=5.0)
    c = GammaPSD(D0=1.0, Nw=1.0, mu=3.0, D_max=5.0)
    assert a == b
    assert a != c


def test_binned_psd():
    edges = [0.0, 1.0, 2.0, 3.0]
    values = [10.0, 20.0, 30.0]
    psd = BinnedPSD(edges, values)
    # Scalar
    assert psd(0.5) == 10.0
    assert psd(1.5) == 20.0
    assert psd(2.5) == 30.0
    # Outside
    assert psd(-0.1) == 0.0
    assert psd(4.0) == 0.0
    # Vector
    arr = psd(np.array([0.5, 1.5, 2.5, 10.0]))
    np.testing.assert_array_equal(arr, np.array([10.0, 20.0, 30.0, 0.0]))


def test_binned_psd_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        BinnedPSD([0.0, 1.0], [1.0, 2.0])


# ---------- PSDIntegrator ----------

@pytest.fixture
def sphere():
    """A simple water sphere scatterer at C-band-ish wavelength."""
    return Scatterer(
        radius=1.0,
        wavelength=6.283185307,
        axis_ratio=1.0,
        m=complex(1.5, 0.01),
        ddelt=1e-3,
        ndgs=2,
    )


def test_psd_integrator_requires_D_max(sphere):
    integ = PSDIntegrator()
    with pytest.raises(AttributeError):
        integ.init_scatter_table(sphere)


def test_psd_integrator_requires_init_before_use(sphere):
    integ = PSDIntegrator()
    with pytest.raises(AttributeError):
        integ.get_SZ(GammaPSD(D0=1.0), (90.0, 90.0, 0.0, 180.0, 0.0, 0.0))


def test_psd_integrator_restores_scatterer_state(sphere):
    sphere.set_geometry((90.0, 90.0, 0.0, 180.0, 0.0, 0.0))
    original = (sphere.radius, sphere.m, sphere.axis_ratio, sphere.get_geometry())

    integ = PSDIntegrator()
    integ.D_max = 3.0
    integ.num_points = 8
    integ.init_scatter_table(sphere)

    # After init the scatterer's radius/m/axis_ratio/geom must all be as before.
    assert sphere.radius == original[0]
    assert sphere.m == original[1]
    assert sphere.axis_ratio == original[2]
    assert sphere.get_geometry() == original[3]


def test_psd_integrator_with_zero_psd_gives_zero(sphere):
    """A PSD that's identically zero should integrate to zero S and Z."""
    sphere.set_geometry((90.0, 90.0, 0.0, 180.0, 0.0, 0.0))
    sphere.psd_integrator = PSDIntegrator()
    sphere.psd_integrator.D_max = 3.0
    sphere.psd_integrator.num_points = 16
    sphere.psd_integrator.init_scatter_table(sphere)
    # ExponentialPSD with a D_max so small every sample is clipped.
    sphere.psd = ExponentialPSD(N0=1.0, Lambda=1.0, D_max=0.0)
    S, Z = sphere.get_SZ()
    np.testing.assert_allclose(S, np.zeros((2, 2)))
    np.testing.assert_allclose(Z, np.zeros((4, 4)))


def test_psd_integrator_linearity_in_N0(sphere):
    """Doubling N0 of an exponential PSD must double S and Z (integration is linear)."""
    sphere.set_geometry((90.0, 90.0, 0.0, 180.0, 0.0, 0.0))
    sphere.psd_integrator = PSDIntegrator()
    sphere.psd_integrator.D_max = 3.0
    sphere.psd_integrator.num_points = 32
    sphere.psd_integrator.init_scatter_table(sphere)

    sphere.psd = ExponentialPSD(N0=100.0, Lambda=2.0, D_max=3.0)
    S1, Z1 = sphere.get_SZ()
    S1, Z1 = S1.copy(), Z1.copy()

    sphere.psd = ExponentialPSD(N0=200.0, Lambda=2.0, D_max=3.0)
    S2, Z2 = sphere.get_SZ()

    np.testing.assert_allclose(S2, 2.0 * S1, rtol=1e-10)
    np.testing.assert_allclose(Z2, 2.0 * Z1, rtol=1e-10)


def test_psd_integrator_save_and_load_roundtrip(sphere, tmp_path):
    sphere.set_geometry((90.0, 90.0, 0.0, 180.0, 0.0, 0.0))
    integ = PSDIntegrator()
    integ.D_max = 2.0
    integ.num_points = 16
    integ.init_scatter_table(sphere)

    fn = tmp_path / "table.pkl"
    integ.save_scatter_table(str(fn), description="test")

    integ2 = PSDIntegrator()
    _, desc = integ2.load_scatter_table(str(fn))
    assert desc == "test"
    assert integ2.num_points == integ.num_points
    assert integ2.D_max == integ.D_max
    for geom in integ.geometries:
        np.testing.assert_allclose(integ2._S_table[geom], integ._S_table[geom])
        np.testing.assert_allclose(integ2._Z_table[geom], integ._Z_table[geom])


def test_psd_integrator_honours_axis_ratio_func_and_m_func(sphere):
    """Per-diameter axis_ratio and m callbacks must reach the Rust tabulator.

    Checks that tm.m and tm.axis_ratio are still the original values after
    init_scatter_table (state restored) but the tabulated S/Z depend on
    the per-D callbacks (not the scatterer's default m).
    """
    sphere.set_geometry((90.0, 90.0, 0.0, 180.0, 0.0, 0.0))
    integ = PSDIntegrator()
    integ.D_max = 3.0
    integ.num_points = 16
    # Pick m_func that actually varies with D to be sure it's used.
    integ.m_func = lambda D: complex(1.5 + 0.1 * D, 0.01)
    integ.axis_ratio_func = lambda D: 1.0 + 0.1 * D
    integ.init_scatter_table(sphere)

    # State fully restored.
    assert sphere.m == complex(1.5, 0.01)
    assert sphere.axis_ratio == 1.0

    # m_table was populated from the callback.
    expected = np.array(
        [complex(1.5 + 0.1 * D, 0.01) for D in integ._psd_D]
    )
    np.testing.assert_allclose(integ._m_table, expected)


def test_psd_integrator_multiple_geometries(sphere):
    """Tabulating multiple geometries produces distinct entries."""
    from rupytmatrix.tmatrix_aux import geom_horiz_back, geom_horiz_forw

    sphere.psd_integrator = PSDIntegrator(geometries=(geom_horiz_back, geom_horiz_forw))
    sphere.psd_integrator.D_max = 2.0
    sphere.psd_integrator.num_points = 12
    sphere.psd_integrator.init_scatter_table(sphere)
    sphere.psd = ExponentialPSD(N0=1.0, Lambda=1.0, D_max=2.0)

    sphere.set_geometry(geom_horiz_back)
    S_back, _ = sphere.get_SZ()
    sphere.set_geometry(geom_horiz_forw)
    S_forw, _ = sphere.get_SZ()
    # Different geometries, different amplitude matrices.
    assert not np.allclose(S_back, S_forw)
