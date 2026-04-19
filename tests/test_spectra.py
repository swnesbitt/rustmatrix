"""Unit tests for :mod:`rustmatrix.spectra`.

Covers:

* Spectral ↔ bulk parity: integrating the spectral matrices over v must
  reproduce the bulk radar observables from :mod:`radar` for every
  observable and across turbulence / beam-broadening configurations.
* Turbulence-convolution identity: constant σ_t spectrum == zero-σ
  spectrum ★ Gaussian(σ_t).
* Beam-broadening identity: u_h·θ_b > 0 broadens the spectrum by the
  expected Doviak–Zrnić width.
* Inertia limits of :class:`InertialZeng2023`.
* :func:`turbulence.from_params` smart-constructor dispatch.
* Fall-speed preset spot-checks.
* HydroMix bimodality (rain + ice).
* Velocity-grid input validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from rustmatrix import (
    HydroMix,
    MixtureComponent,
    Scatterer,
    SpectralIntegrator,
    radar,
    spectra,
)
from rustmatrix.psd import ExponentialPSD, GammaPSD, PSDIntegrator
from rustmatrix.refractive import m_w_10C, mi
from rustmatrix.spectra import (
    GaussianTurbulence,
    InertialZeng2023,
    NoTurbulence,
)
from rustmatrix.tmatrix_aux import (
    K_w_sqr,
    dsr_thurai_2007,
    geom_vert_back,
    geom_vert_forw,
    wl_X,
)


# ---------- fixtures ----------

def _rain_scatterer_vert():
    s = Scatterer(
        wavelength=wl_X,
        m=m_w_10C[wl_X],
        Kw_sqr=K_w_sqr[wl_X],
        ddelt=1e-4,
        ndgs=2,
    )
    integ = PSDIntegrator()
    integ.D_max = 6.0
    integ.num_points = 32
    integ.axis_ratio_func = lambda D: 1.0 / dsr_thurai_2007(D)
    integ.geometries = (geom_vert_back, geom_vert_forw)
    s.psd_integrator = integ
    s.psd_integrator.init_scatter_table(s)
    s.psd = GammaPSD(D0=1.5, Nw=8e3, mu=4, D_max=6.0)
    return s


def _ice_scatterer_vert():
    s = Scatterer(
        wavelength=wl_X,
        m=mi(wl_X, 0.2),
        Kw_sqr=K_w_sqr[wl_X],
        axis_ratio=0.6,
        ddelt=1e-4,
        ndgs=2,
    )
    integ = PSDIntegrator()
    integ.D_max = 8.0
    integ.num_points = 32
    integ.geometries = (geom_vert_back, geom_vert_forw)
    s.psd_integrator = integ
    s.psd_integrator.init_scatter_table(s)
    s.psd = ExponentialPSD(N0=5e3, Lambda=2.0, D_max=8.0)
    return s


@pytest.fixture(scope="module")
def rain_v():
    return _rain_scatterer_vert()


@pytest.fixture(scope="module")
def ice_v():
    return _ice_scatterer_vert()


# ----------------------------------------------------------------------
# Spectral <-> bulk parity (the "most important" consistency test).
# ----------------------------------------------------------------------

def _bulk_backscatter(scatterer):
    scatterer.set_geometry(geom_vert_back)
    Zh = radar.refl(scatterer, h_pol=True)
    Zv = radar.refl(scatterer, h_pol=False)
    Zdr = radar.Zdr(scatterer)
    rho = radar.rho_hv(scatterer)
    delta = radar.delta_hv(scatterer)
    return Zh, Zv, Zdr, rho, delta


def _bulk_forward(scatterer):
    scatterer.set_geometry(geom_vert_forw)
    return radar.Kdp(scatterer)


@pytest.mark.parametrize("turb_kind", ["none", "gaussian", "inertial", "beam"])
def test_spectral_integrates_to_bulk_rain(rain_v, turb_kind):
    """∫S_spec dv == S_bulk and ∫Z_spec dv == Z_bulk for every turbulence
    configuration, AND the non-linear observables re-derived from the
    summed matrices match the bulk radar.* calls."""

    if turb_kind == "none":
        turb = NoTurbulence()
        u_h, theta = 0.0, 0.0
    elif turb_kind == "gaussian":
        turb = GaussianTurbulence(0.4)
        u_h, theta = 0.0, 0.0
    elif turb_kind == "inertial":
        turb = InertialZeng2023(sigma_air=0.5, epsilon=1e-3)
        u_h, theta = 0.0, 0.0
    else:  # beam
        turb = NoTurbulence()
        u_h, theta = 5.0, np.deg2rad(1.0)  # 1° beam, 5 m/s horizontal wind

    integ = SpectralIntegrator(
        rain_v,
        fall_speed=spectra.fall_speed.atlas_srivastava_sekhon_1973,
        turbulence=turb,
        v_min=-2.0,
        v_max=12.0,
        n_bins=2048,
        u_h=u_h,
        beamwidth=theta,
        geometry_backscatter=geom_vert_back,
        geometry_forward=geom_vert_forw,
    )
    res = integ.run()

    # Bulk values for rain at the current geometry.
    Zh_b, Zv_b, Zdr_b, rho_b, delta_b = _bulk_backscatter(rain_v)
    Kdp_b = _bulk_forward(rain_v)

    # Linear-in-spectrum observables: trapezoidal integral matches bulk.
    assert np.trapezoid(res.sZ_h, res.v) == pytest.approx(Zh_b, rel=1e-3)
    assert np.trapezoid(res.sZ_v, res.v) == pytest.approx(Zv_b, rel=1e-3)
    assert np.trapezoid(res.sKdp, res.v) == pytest.approx(Kdp_b, rel=1e-3)

    # Non-linear observables: re-derive from summed S/Z.
    bulk = res.collapse_to_bulk()
    assert radar.refl(bulk, h_pol=True) == pytest.approx(Zh_b, rel=1e-3)
    assert radar.refl(bulk, h_pol=False) == pytest.approx(Zv_b, rel=1e-3)
    assert radar.Zdr(bulk) == pytest.approx(Zdr_b, rel=1e-3)
    assert radar.rho_hv(bulk) == pytest.approx(rho_b, rel=1e-4)
    # delta is ~0 for flat rain at X-band — compare on absolute scale.
    assert radar.delta_hv(bulk) == pytest.approx(delta_b, abs=1e-4)


def test_spectral_integrates_to_bulk_hydromix(rain_v, ice_v):
    """HydroMix: bulk sums still round-trip through the spectrum."""
    mix = HydroMix([
        MixtureComponent(rain_v, rain_v.psd, "rain"),
        MixtureComponent(ice_v, ice_v.psd, "ice"),
    ])
    integ = SpectralIntegrator(
        mix,
        component_kinematics={
            "rain": (
                spectra.fall_speed.atlas_srivastava_sekhon_1973,
                GaussianTurbulence(0.3),
            ),
            "ice": (
                spectra.fall_speed.locatelli_hobbs_1974_aggregates,
                GaussianTurbulence(0.3),
            ),
        },
        v_min=-2.0,
        v_max=12.0,
        n_bins=2048,
        geometry_backscatter=geom_vert_back,
        geometry_forward=geom_vert_forw,
    )
    res = integ.run()

    mix.set_geometry(geom_vert_back)
    Zh_bulk = radar.refl(mix, h_pol=True)
    Zv_bulk = radar.refl(mix, h_pol=False)
    Zdr_bulk = radar.Zdr(mix)
    rho_bulk = radar.rho_hv(mix)
    mix.set_geometry(geom_vert_forw)
    Kdp_bulk = radar.Kdp(mix)

    assert np.trapezoid(res.sZ_h, res.v) == pytest.approx(Zh_bulk, rel=1e-3)
    assert np.trapezoid(res.sZ_v, res.v) == pytest.approx(Zv_bulk, rel=1e-3)
    assert np.trapezoid(res.sKdp, res.v) == pytest.approx(Kdp_bulk, rel=1e-3)

    bulk = res.collapse_to_bulk()
    assert radar.Zdr(bulk) == pytest.approx(Zdr_bulk, rel=1e-3)
    assert radar.rho_hv(bulk) == pytest.approx(rho_bulk, rel=1e-3)


# ----------------------------------------------------------------------
# Turbulence-convolution identity.
# ----------------------------------------------------------------------

def test_turbulence_is_gaussian_convolution(rain_v):
    """sZ_h(σ_t constant) == sZ_h(σ_t=0) ⊛ Gaussian(σ_t)."""
    v = np.linspace(-2.0, 12.0, 4096)
    sigma = 0.5

    no_turb = SpectralIntegrator(
        rain_v,
        fall_speed=spectra.fall_speed.atlas_srivastava_sekhon_1973,
        turbulence=NoTurbulence(),
        v_bins=v,
    ).run()
    with_turb = SpectralIntegrator(
        rain_v,
        fall_speed=spectra.fall_speed.atlas_srivastava_sekhon_1973,
        turbulence=GaussianTurbulence(sigma),
        v_bins=v,
    ).run()

    dv = v[1] - v[0]
    # Gaussian kernel on a symmetric grid.
    k_grid = np.arange(-6 * sigma, 6 * sigma + dv, dv)
    kernel = np.exp(-0.5 * (k_grid / sigma) ** 2) / (np.sqrt(2.0 * np.pi) * sigma)
    kernel /= kernel.sum() * dv  # ensure integral = 1 on this grid

    from scipy.signal import fftconvolve

    conv = fftconvolve(no_turb.sZ_h, kernel, mode="same") * dv

    # Compare where the spectrum has meaningful power.
    mask = with_turb.sZ_h > 1e-3 * with_turb.sZ_h.max()
    assert np.allclose(conv[mask], with_turb.sZ_h[mask], rtol=5e-2, atol=1e-2)


def test_beam_broadening_is_gaussian_convolution(rain_v):
    """u_h·θ_b > 0 broadens by σ_beam = u_h θ_b / (2√(2ln2))."""
    v = np.linspace(-2.0, 12.0, 4096)
    u_h = 8.0
    theta = np.deg2rad(1.5)
    sigma_beam = u_h * theta / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    no_turb = SpectralIntegrator(
        rain_v,
        fall_speed=spectra.fall_speed.atlas_srivastava_sekhon_1973,
        turbulence=NoTurbulence(),
        v_bins=v,
    ).run()
    with_beam = SpectralIntegrator(
        rain_v,
        fall_speed=spectra.fall_speed.atlas_srivastava_sekhon_1973,
        turbulence=NoTurbulence(),
        v_bins=v,
        u_h=u_h,
        beamwidth=theta,
    ).run()

    dv = v[1] - v[0]
    k_grid = np.arange(-6 * sigma_beam, 6 * sigma_beam + dv, dv)
    kernel = np.exp(-0.5 * (k_grid / sigma_beam) ** 2) / (
        np.sqrt(2.0 * np.pi) * sigma_beam
    )
    kernel /= kernel.sum() * dv

    from scipy.signal import fftconvolve

    conv = fftconvolve(no_turb.sZ_h, kernel, mode="same") * dv

    mask = with_beam.sZ_h > 1e-3 * with_beam.sZ_h.max()
    assert np.allclose(conv[mask], with_beam.sZ_h[mask], rtol=5e-2, atol=1e-2)


# ----------------------------------------------------------------------
# Zeng 2023 inertia limits.
# ----------------------------------------------------------------------

def test_inertia_small_stokes_limit():
    """Very small drops → σ_t ≈ σ_air (Gaussian limit)."""
    model = InertialZeng2023(sigma_air=0.6, epsilon=1e-3, L_o=100.0)
    sigma = model(np.array([0.01, 0.05]))
    assert np.allclose(sigma, 0.6, rtol=1e-3)


def test_inertia_large_stokes_limit():
    """Very large terminal velocity → σ_t ≈ 0 (no-turbulence limit)."""
    # Force a very small eddy scale so Stokes is huge for any drop.
    model = InertialZeng2023(sigma_air=0.6, epsilon=1e3, L_o=1e-3)
    sigma = model(np.array([5.0]))
    assert sigma[0] < 1e-3


# ----------------------------------------------------------------------
# turbulence.from_params() dispatch.
# ----------------------------------------------------------------------

def test_from_params_dispatch():
    assert isinstance(spectra.turbulence.from_params(), NoTurbulence)
    assert isinstance(spectra.turbulence.from_params(sigma=0.0), NoTurbulence)
    t1 = spectra.turbulence.from_params(sigma=0.5)
    assert isinstance(t1, GaussianTurbulence) and t1.sigma_t == 0.5
    t2 = spectra.turbulence.from_params(sigma=0.5, epsilon=1e-3)
    assert isinstance(t2, InertialZeng2023) and t2.sigma_air == 0.5
    t3 = spectra.turbulence.from_params(epsilon=1e-3)
    assert isinstance(t3, InertialZeng2023)


# ----------------------------------------------------------------------
# Fall-speed preset spot-checks against published tabulations.
# ----------------------------------------------------------------------

def test_fall_speed_presets_reasonable():
    # Gunn & Kinzer 1949 gives roughly: 1mm ~ 4 m/s, 3mm ~ 8 m/s, 5mm ~ 9 m/s.
    D = np.array([1.0, 3.0, 5.0])
    for fs in [
        spectra.fall_speed.atlas_srivastava_sekhon_1973,
        spectra.fall_speed.brandes_et_al_2002,
        spectra.fall_speed.beard_1976,
    ]:
        v = fs(D)
        assert v[0] == pytest.approx(4.0, abs=0.5), f"{fs.__name__} at 1 mm"
        assert v[1] == pytest.approx(8.0, abs=0.7), f"{fs.__name__} at 3 mm"
        assert v[2] == pytest.approx(9.0, abs=1.0), f"{fs.__name__} at 5 mm"

    agg = spectra.fall_speed.locatelli_hobbs_1974_aggregates(np.array([1.0, 5.0]))
    # Aggregates: 1 mm ~ 0.69 m/s, 5 mm ~ 1.3 m/s.
    assert agg[0] == pytest.approx(0.69, abs=0.05)
    assert agg[1] == pytest.approx(1.3, abs=0.1)

    graup = spectra.fall_speed.locatelli_hobbs_1974_graupel_hex(np.array([1.0, 2.0]))
    assert graup[0] == pytest.approx(1.1, abs=0.05)
    assert graup[1] == pytest.approx(1.63, abs=0.1)


def test_power_law_factory():
    fs = spectra.fall_speed.power_law(a=4.0, b=0.67, D_ref=1.0)
    assert fs(np.array([1.0]))[0] == pytest.approx(4.0)
    assert fs(np.array([2.0]))[0] == pytest.approx(4.0 * 2.0 ** 0.67)


# ----------------------------------------------------------------------
# HydroMix bimodality.
# ----------------------------------------------------------------------

def test_hydromix_bimodal_spectrum(rain_v, ice_v):
    mix = HydroMix([
        MixtureComponent(rain_v, rain_v.psd, "rain"),
        MixtureComponent(ice_v, ice_v.psd, "ice"),
    ])
    integ = SpectralIntegrator(
        mix,
        component_kinematics={
            "rain": (
                spectra.fall_speed.atlas_srivastava_sekhon_1973,
                GaussianTurbulence(0.1),
            ),
            "ice": (
                spectra.fall_speed.locatelli_hobbs_1974_aggregates,
                GaussianTurbulence(0.1),
            ),
        },
        v_min=-1.0, v_max=12.0, n_bins=2048,
    )
    res = integ.run()

    # Split the spectrum into ice-mode (v < 2 m/s) and rain-mode (v > 2 m/s).
    ice_mode = res.v < 2.0
    rain_mode = res.v > 2.0
    ice_peak = res.sZ_h[ice_mode].max()
    rain_peak = res.sZ_h[rain_mode].max()
    # Both modes must have non-trivial power.
    assert ice_peak > 0 and rain_peak > 0
    # There must be a valley between them (min in 1.2-2.5 m/s range).
    valley_mask = (res.v > 1.2) & (res.v < 2.5)
    if valley_mask.any():
        valley_min = res.sZ_h[valley_mask].min()
        assert valley_min < min(ice_peak, rain_peak)


# ----------------------------------------------------------------------
# Velocity-grid input validation.
# ----------------------------------------------------------------------

def test_velocity_grid_requires_one_form(rain_v):
    with pytest.raises(ValueError, match="Pass either"):
        SpectralIntegrator(
            rain_v,
            fall_speed=spectra.fall_speed.atlas_srivastava_sekhon_1973,
        )


def test_velocity_grid_exclusive(rain_v):
    with pytest.raises(ValueError, match="Pass only one"):
        SpectralIntegrator(
            rain_v,
            fall_speed=spectra.fall_speed.atlas_srivastava_sekhon_1973,
            v_bins=np.linspace(0, 10, 64),
            v_min=0, v_max=10, n_bins=64,
        )


def test_velocity_grid_convenience_triple(rain_v):
    integ = SpectralIntegrator(
        rain_v,
        fall_speed=spectra.fall_speed.atlas_srivastava_sekhon_1973,
        turbulence=GaussianTurbulence(0.2),
        v_min=0.0, v_max=10.0, n_bins=128,
    )
    assert integ.v_bins.shape == (128,)
    assert integ.v_bins[0] == 0.0
    assert integ.v_bins[-1] == 10.0


def test_narrow_grid_warns(rain_v):
    with pytest.warns(UserWarning, match="beyond v_bins"):
        SpectralIntegrator(
            rain_v,
            fall_speed=spectra.fall_speed.atlas_srivastava_sekhon_1973,
            turbulence=NoTurbulence(),
            v_min=0.0, v_max=1.0, n_bins=32,  # way too narrow for rain
        ).run()


# ---------- system noise ----------

def _integ_with_noise(rain, noise):
    return SpectralIntegrator(
        rain,
        fall_speed=spectra.fall_speed.atlas_srivastava_sekhon_1973,
        turbulence=GaussianTurbulence(0.3),
        v_min=-1.0, v_max=12.0, n_bins=512,
        geometry_backscatter=geom_vert_back,
        noise=noise,
    )


def test_noise_default_off_preserves_round_trip(rain_v):
    res = _integ_with_noise(rain_v, None).run()
    assert res.noise_h == 0.0 and res.noise_v == 0.0
    rain_v.set_geometry(geom_vert_back)
    Zh_bulk = radar.refl(rain_v)
    Zh_int = np.trapezoid(res.sZ_h, res.v)
    assert abs(Zh_int - Zh_bulk) / Zh_bulk < 1e-3


def test_noise_scalar_adds_to_both_channels(rain_v):
    noise = 1e-2  # -20 dBZ
    res = _integ_with_noise(rain_v, noise).run()
    assert res.noise_h == pytest.approx(noise)
    assert res.noise_v == pytest.approx(noise)
    # Integrated noise floor equals the requested total reflectivity.
    # signal-only integral from the no-noise run:
    sig = _integ_with_noise(rain_v, None).run()
    sZh_int_signal = np.trapezoid(sig.sZ_h, sig.v)
    sZh_int_noisy = np.trapezoid(res.sZ_h, res.v)
    assert sZh_int_noisy - sZh_int_signal == pytest.approx(noise, rel=1e-3)


def test_noise_realistic_uses_default(rain_v):
    res = _integ_with_noise(rain_v, "realistic").run()
    expected = spectra.realistic_noise_floor(rain_v.wavelength)
    assert res.noise_h == pytest.approx(expected)
    assert res.noise_v == pytest.approx(expected)


def test_noise_tuple_asymmetric(rain_v):
    res = _integ_with_noise(rain_v, (1e-2, 5e-3)).run()
    assert res.noise_h == pytest.approx(1e-2)
    assert res.noise_v == pytest.approx(5e-3)


def test_noise_biases_rho_hv_down(rain_v):
    clean = _integ_with_noise(rain_v, None).run()
    noisy = _integ_with_noise(rain_v, 1e-2).run()
    # Pick a bin near the peak of sZ_h where SNR is finite and ρ
    # should still be well-behaved.
    k = int(np.argmax(clean.sZ_h))
    # Signal rho_hv at vertical pointing for axisymmetric drops ≈ 1,
    # so the biased version must be ≤ the signal value.
    assert noisy.srho_hv[k] <= clean.srho_hv[k] + 1e-12


def test_noise_signal_matrices_unaffected(rain_v):
    clean = _integ_with_noise(rain_v, None).run()
    noisy = _integ_with_noise(rain_v, 1e-2).run()
    # Underlying S_spec / Z_spec are signal-only regardless of noise.
    np.testing.assert_allclose(noisy.S_spec, clean.S_spec)
    np.testing.assert_allclose(noisy.Z_spec, clean.Z_spec)
    # collapse_to_bulk operates on signal matrices, so bulk round-trip
    # still holds even when noise is on.
    shim = noisy.collapse_to_bulk()
    rain_v.set_geometry(geom_vert_back)
    assert abs(radar.refl(shim) - radar.refl(rain_v)) / radar.refl(rain_v) < 1e-3


def test_realistic_noise_floor_helper():
    assert spectra.realistic_noise_floor(Z_dBZ=0.0) == pytest.approx(1.0)
    assert spectra.realistic_noise_floor(Z_dBZ=-20.0) == pytest.approx(0.01)


def test_noise_invalid_type_raises(rain_v):
    with pytest.raises(TypeError):
        _integ_with_noise(rain_v, object()).run()


def test_noise_negative_raises(rain_v):
    with pytest.raises(ValueError):
        _integ_with_noise(rain_v, -1e-3).run()
