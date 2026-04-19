"""Unit tests for :mod:`rustmatrix.hd_mix`.

Covers:

* identity — a single-component mixture matches the underlying scatterer;
* linearity — two copies of the same scatterer each running half the
  PSD sum to the same S/Z as one scatterer running the full PSD;
* validation — wavelength mismatch, unregistered geometry, empty
  mixture;
* geometry propagation through ``set_geometry``;
* physical sanity for a rain + oriented-ice mixture at C-band
  (Z_h additive in linear units; Z_dr, K_dp additive; ρ_hv
  strictly below the min of the two components, since heterogeneity
  *must* decorrelate the H/V channels);
* pytmatrix parity (skipped automatically when pytmatrix is not
  installed — it pins numpy<2).
"""

from __future__ import annotations

import numpy as np
import pytest

from rustmatrix import HydroMix, MixtureComponent, Scatterer, radar, scatter
from rustmatrix.psd import ExponentialPSD, GammaPSD, PSDIntegrator
from rustmatrix.refractive import m_w_10C, mi
from rustmatrix.tmatrix_aux import (
    K_w_sqr,
    dsr_thurai_2007,
    geom_horiz_back,
    geom_horiz_forw,
    wl_C,
)


# ---------- fixtures ----------

def _rain_scatterer():
    s = Scatterer(
        wavelength=wl_C,
        m=m_w_10C[wl_C],
        Kw_sqr=K_w_sqr[wl_C],
        ddelt=1e-4,
        ndgs=2,
    )
    integ = PSDIntegrator()
    integ.D_max = 6.0
    integ.num_points = 32
    integ.axis_ratio_func = lambda D: 1.0 / dsr_thurai_2007(D)
    integ.geometries = (geom_horiz_back, geom_horiz_forw)
    s.psd_integrator = integ
    s.psd_integrator.init_scatter_table(s)
    return s


def _ice_scatterer():
    s = Scatterer(
        wavelength=wl_C,
        m=mi(wl_C, 0.2),       # low-density aggregates (rimed snow-ish)
        Kw_sqr=K_w_sqr[wl_C],
        axis_ratio=0.6,        # prolate
        ddelt=1e-4,
        ndgs=2,
    )
    integ = PSDIntegrator()
    integ.D_max = 8.0
    integ.num_points = 32
    integ.geometries = (geom_horiz_back, geom_horiz_forw)
    s.psd_integrator = integ
    s.psd_integrator.init_scatter_table(s)
    return s


@pytest.fixture(scope="module")
def rain_scatterer():
    return _rain_scatterer()


@pytest.fixture(scope="module")
def ice_scatterer():
    return _ice_scatterer()


@pytest.fixture
def rain_psd():
    return GammaPSD(D0=1.5, Nw=8e3, mu=4, D_max=6.0)


@pytest.fixture
def ice_psd():
    return ExponentialPSD(N0=5e3, Lambda=2.0, D_max=8.0)


# ---------- identity / linearity ----------

def test_single_component_matches_underlying_scatterer(rain_scatterer, rain_psd):
    """A one-component mixture must produce S/Z identical to the raw
    PSD-integrated scatterer output."""
    rain_scatterer.psd = rain_psd
    rain_scatterer.set_geometry(geom_horiz_back)
    S_ref, Z_ref = rain_scatterer.get_SZ()

    mix = HydroMix([MixtureComponent(rain_scatterer, rain_psd, "rain")])
    mix.set_geometry(geom_horiz_back)
    S_mix, Z_mix = mix.get_SZ()

    np.testing.assert_allclose(S_mix, S_ref, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(Z_mix, Z_ref, rtol=1e-12, atol=1e-14)

    # And every radar observable should agree.
    for name, fn in [
        ("refl_h", lambda x: radar.refl(x, True)),
        ("refl_v", lambda x: radar.refl(x, False)),
        ("Zdr", radar.Zdr),
        ("delta_hv", radar.delta_hv),
        ("rho_hv", radar.rho_hv),
    ]:
        a = fn(rain_scatterer)
        b = fn(mix)
        assert np.allclose(a, b, rtol=1e-10, atol=1e-14), name


def test_linearity_two_halves_equal_whole(rain_scatterer):
    """Two components that share the scatterer, each running N/2, should
    produce the same S/Z as one component running the full N."""
    full = GammaPSD(D0=1.5, Nw=8e3, mu=4, D_max=6.0)
    half = GammaPSD(D0=1.5, Nw=4e3, mu=4, D_max=6.0)

    mix_full = HydroMix([MixtureComponent(rain_scatterer, full, "full")])
    mix_halves = HydroMix([
        MixtureComponent(rain_scatterer, half, "half-a"),
        MixtureComponent(rain_scatterer, half, "half-b"),
    ])

    for geom in (geom_horiz_back, geom_horiz_forw):
        mix_full.set_geometry(geom)
        mix_halves.set_geometry(geom)
        S_full, Z_full = mix_full.get_SZ()
        S_halves, Z_halves = mix_halves.get_SZ()
        np.testing.assert_allclose(S_halves, S_full, rtol=1e-10, atol=1e-14)
        np.testing.assert_allclose(Z_halves, Z_full, rtol=1e-10, atol=1e-14)


# ---------- validation ----------

def test_wavelength_mismatch_raises(rain_scatterer):
    """Two components at different wavelengths is physically meaningless —
    reject it up-front."""
    other = Scatterer(
        wavelength=33.3,   # X-band, not C
        m=complex(7.99, 2.21),
        ddelt=1e-3,
        ndgs=2,
    )
    integ = PSDIntegrator()
    integ.D_max = 5.0
    integ.num_points = 16
    integ.geometries = (geom_horiz_back,)
    other.psd_integrator = integ
    other.psd_integrator.init_scatter_table(other)

    mix = HydroMix([MixtureComponent(rain_scatterer, GammaPSD(D0=1, Nw=1e3, mu=0, D_max=6.0))])
    with pytest.raises(ValueError, match="Wavelength mismatch"):
        mix.add(MixtureComponent(other, ExponentialPSD(N0=1e3, Lambda=1.0, D_max=5.0)))


def test_uninitialised_integrator_raises():
    """Forgetting init_scatter_table should produce a clear error, not
    fail somewhere deep in get_SZ."""
    s = Scatterer(wavelength=wl_C, m=m_w_10C[wl_C], ddelt=1e-3)
    integ = PSDIntegrator()
    integ.D_max = 5.0
    integ.num_points = 16
    integ.geometries = (geom_horiz_back,)
    s.psd_integrator = integ
    # Note: init_scatter_table NOT called.
    with pytest.raises(ValueError, match="uninitialised"):
        HydroMix([MixtureComponent(s, GammaPSD(D0=1, Nw=1e3, mu=0, D_max=5.0))])


def test_missing_geometry_raises(rain_scatterer, rain_psd):
    mix = HydroMix([MixtureComponent(rain_scatterer, rain_psd)])
    bogus = (45.0, 45.0, 0.0, 0.0, 0.0, 0.0)
    mix.set_geometry(bogus)
    with pytest.raises(ValueError, match="not registered"):
        mix.get_SZ()


def test_empty_mixture_raises():
    mix = HydroMix()
    with pytest.raises(ValueError, match="no components"):
        mix.get_SZ()


# ---------- geometry propagation ----------

def test_set_geometry_propagates_to_components(rain_scatterer, rain_psd):
    mix = HydroMix([MixtureComponent(rain_scatterer, rain_psd)])
    mix.set_geometry(geom_horiz_forw)
    assert rain_scatterer.get_geometry() == geom_horiz_forw
    assert mix.get_geometry() == geom_horiz_forw
    mix.set_geometry(geom_horiz_back)
    assert rain_scatterer.get_geometry() == geom_horiz_back


# ---------- physical sanity: rain + oriented ice ----------

def _refl_linear(scatterer_like):
    return radar.refl(scatterer_like, h_pol=True)


def test_rain_plus_ice_physical_sanity(rain_scatterer, ice_scatterer,
                                       rain_psd, ice_psd):
    """The linear-in-N(D) observables must add across species, and
    heterogeneity must depress ρ_hv below either single-species value."""
    # Backscatter observables
    rain_scatterer.psd = rain_psd
    rain_scatterer.set_geometry(geom_horiz_back)
    Zh_rain = _refl_linear(rain_scatterer)
    Zdr_rain = radar.Zdr(rain_scatterer)
    rho_rain = radar.rho_hv(rain_scatterer)

    ice_scatterer.psd = ice_psd
    ice_scatterer.set_geometry(geom_horiz_back)
    Zh_ice = _refl_linear(ice_scatterer)
    Zdr_ice = radar.Zdr(ice_scatterer)
    rho_ice = radar.rho_hv(ice_scatterer)

    mix = HydroMix([
        MixtureComponent(rain_scatterer, rain_psd, "rain"),
        MixtureComponent(ice_scatterer, ice_psd, "ice"),
    ])
    mix.set_geometry(geom_horiz_back)
    Zh_mix = _refl_linear(mix)
    Zdr_mix = radar.Zdr(mix)
    rho_mix = radar.rho_hv(mix)

    # Linear reflectivity is strictly additive in N(D).
    np.testing.assert_allclose(Zh_mix, Zh_rain + Zh_ice, rtol=1e-10)

    # Z_dr for an incoherent mixture is (σ_h_rain + σ_h_ice) / (σ_v_rain + σ_v_ice).
    # It must lie between the two per-species Z_dr values.
    lo, hi = sorted((Zdr_rain, Zdr_ice))
    assert lo - 1e-10 <= Zdr_mix <= hi + 1e-10

    # ρ_hv from the summed Z must fall below the minimum of the two
    # single-species ρ_hv — mixing species with different shape/orientation
    # statistics decorrelates the H/V channels.
    assert rho_mix < min(rho_rain, rho_ice)
    assert 0.0 < rho_mix <= 1.0

    # Forward-scatter observables (K_dp) must be additive.
    rain_scatterer.set_geometry(geom_horiz_forw)
    ice_scatterer.set_geometry(geom_horiz_forw)
    Kdp_rain = radar.Kdp(rain_scatterer)
    Kdp_ice = radar.Kdp(ice_scatterer)
    mix.set_geometry(geom_horiz_forw)
    Kdp_mix = radar.Kdp(mix)
    np.testing.assert_allclose(Kdp_mix, Kdp_rain + Kdp_ice, rtol=1e-10)


def test_ext_xsect_works_through_mixture(rain_scatterer, ice_scatterer,
                                         rain_psd, ice_psd):
    """``scatter.ext_xsect`` takes the optical-theorem path on the mixture
    (because ``HydroMix.psd_integrator is None``), so σ_ext must come out
    as the sum of each species' forward-S contribution."""
    mix = HydroMix([
        MixtureComponent(rain_scatterer, rain_psd, "rain"),
        MixtureComponent(ice_scatterer, ice_psd, "ice"),
    ])
    mix.set_geometry(geom_horiz_forw)
    sigma_mix = scatter.ext_xsect(mix, h_pol=True)

    # Compare against summing each species' ext_xsect through the
    # same optical-theorem path. To force that path we temporarily
    # detach psd_integrator on each species (restoring afterward)
    # — but actually we can just set up PSDs and let ext_xsect use
    # the integrator's optical-theorem forward geometry via get_S.
    # Easier: compute forward S for each species and sum Im.
    rain_scatterer.psd = rain_psd
    rain_scatterer.set_geometry(geom_horiz_forw)
    S_rain, _ = rain_scatterer.get_SZ()
    ice_scatterer.psd = ice_psd
    ice_scatterer.set_geometry(geom_horiz_forw)
    S_ice, _ = ice_scatterer.get_SZ()
    sigma_expected = 2 * mix.wavelength * (S_rain[1, 1].imag + S_ice[1, 1].imag)

    np.testing.assert_allclose(sigma_mix, sigma_expected, rtol=1e-10)


# ---------- pytmatrix parity (optional) ----------

def test_parity_hand_sum_matches_mixture(rain_scatterer, ice_scatterer,
                                         rain_psd, ice_psd):
    """Two independent rustmatrix runs summed by hand at the S/Z level
    must match HydroMix to float tolerance. This is the rustmatrix
    self-parity check; the pytmatrix cross-check lives in
    test_parity_pytmatrix.py where available."""
    for geom in (geom_horiz_back, geom_horiz_forw):
        rain_scatterer.psd = rain_psd
        rain_scatterer.set_geometry(geom)
        S1, Z1 = rain_scatterer.get_SZ()

        ice_scatterer.psd = ice_psd
        ice_scatterer.set_geometry(geom)
        S2, Z2 = ice_scatterer.get_SZ()

        mix = HydroMix([
            MixtureComponent(rain_scatterer, rain_psd, "rain"),
            MixtureComponent(ice_scatterer, ice_psd, "ice"),
        ])
        mix.set_geometry(geom)
        S_mix, Z_mix = mix.get_SZ()

        np.testing.assert_allclose(S_mix, S1 + S2, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(Z_mix, Z1 + Z2, rtol=1e-12, atol=1e-14)
