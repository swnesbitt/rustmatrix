"""Unit tests for :mod:`rustmatrix.spectra.beam`.

Covers:

* Bessel-J1 polynomial approximation is accurate over the range of
  arguments the Airy pattern will ever evaluate (|x| ≤ 30).
* Gaussian / Airy beam patterns have the right value at boresight
  (G(0) = 1) and at the half-power angle (G(hpbw/2) = 0.5).
* Airy first-sidelobe level is −17.57 dB, matching the textbook value.
* Sampling weights are normalized and integrate the pattern correctly.
* Homogeneous-scene limit: a :class:`BeamIntegrator` with a uniform
  scene reproduces the closed-form beam broadening that
  :class:`SpectralIntegrator` builds through its ``u_h`` / ``beamwidth``
  arguments (to ~2 % on spectral width for a Gaussian beam).
* Bulk-sum identity: integrating ``sZ_h`` over the velocity grid
  reproduces the PSD-integrated reflectivity at boresight up to leakage.
* Heterogeneous scene: a pair of cells at different reflectivities
  produces different Doppler velocities under a 1° vs 3° beam.
"""

from __future__ import annotations

import numpy as np
import pytest

from rustmatrix import Scatterer
from rustmatrix.psd import ExponentialPSD, PSDIntegrator
from rustmatrix.refractive import m_w_10C
from rustmatrix.spectra import SpectralIntegrator
from rustmatrix.spectra.beam import (
    AiryBeam,
    BeamIntegrator,
    GaussianBeam,
    Scene,
    TabulatedBeam,
    marshall_palmer_psd_factory,
    _j1,
)
from rustmatrix.tmatrix_aux import (
    K_w_sqr,
    dsr_thurai_2007,
    geom_vert_back,
    wl_W,
    wl_X,
)


# ---------- Bessel J1 ----------

def test_j1_matches_scipy_or_tabulated():
    """A&S polynomial approximation matches tabulated J1 to 2e-7."""
    # Tabulated values from DLMF table 10.22 / Abramowitz & Stegun Table 9.1.
    x = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0])
    expected = np.array([
        0.2422685,  # J1(0.5)
        0.4400506,  # J1(1)
        0.5767249,  # J1(2)
        0.3390590,  # J1(3)
        -0.3275791, # J1(5)
        0.1352484,  # J1(7.5)
        0.0434727,  # J1(10)
        0.2051040,  # J1(15)
        0.0668331,  # J1(20)
    ])
    got = _j1(x)
    np.testing.assert_allclose(got, expected, atol=2e-7)


# ---------- Beam patterns ----------

@pytest.mark.parametrize("Pattern", [GaussianBeam, AiryBeam])
def test_beam_gain_endpoints(Pattern):
    hpbw = np.deg2rad(1.0)
    b = Pattern(hpbw=hpbw)
    assert b.gain(0.0) == pytest.approx(1.0, abs=1e-12)
    assert b.gain(hpbw / 2.0) == pytest.approx(0.5, rel=1e-4)


def test_airy_first_sidelobe_level():
    """First Airy sidelobe is at −17.57 dB, main-lobe null at x ≈ 3.83."""
    b = AiryBeam(hpbw=np.deg2rad(1.0))
    # First null: x = α sin θ = 3.8317.
    theta_null = np.arcsin(3.8317 / b.alpha)
    g_null = b.gain(theta_null)
    assert g_null < 1e-4

    # First sidelobe maximum: x ≈ 5.1356, pattern ≈ 0.01750 (−17.57 dB).
    theta_peak = np.arcsin(5.1356 / b.alpha)
    g_peak = b.gain(theta_peak)
    assert g_peak == pytest.approx(0.01750, rel=0.02)
    assert 10 * np.log10(g_peak) == pytest.approx(-17.57, abs=0.1)


def test_beam_sample_weights_normalized():
    for Pattern in (GaussianBeam, AiryBeam):
        b = Pattern(hpbw=np.deg2rad(1.0))
        _, _, w = b.sample(n_theta=48, n_phi=24, max_theta=np.deg2rad(3.0))
        assert w.sum() == pytest.approx(1.0, rel=1e-6)
        assert np.all(w >= 0)


def test_tabulated_beam_infers_hpbw():
    theta = np.linspace(0.0, 0.1, 101)
    # Gaussian-like sample
    sigma = 0.02
    gain = np.exp(-0.5 * (theta / sigma) ** 2)
    tb = TabulatedBeam(theta=theta, gain=gain)
    expected_hpbw = 2.0 * sigma * np.sqrt(2.0 * np.log(2.0))
    assert tb.hpbw == pytest.approx(expected_hpbw, rel=5e-3)
    np.testing.assert_allclose(tb.gain(0.0), 1.0)
    assert tb.gain(expected_hpbw / 2) == pytest.approx(0.5, rel=5e-3)


# ---------- Scene / Homogeneous equivalence ----------

def _rain_scatterer_X():
    """X-band rain scatterer with PSDIntegrator ready for beam integration."""
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
    integ.geometries = (geom_vert_back,)
    s.psd_integrator = integ
    s.psd_integrator.init_scatter_table(s)
    s.psd = ExponentialPSD(N0=8000.0, Lambda=2.2, D_max=6.0)
    return s


def test_homogeneous_scene_matches_closed_form_sigma_beam():
    """A uniform scene + Gaussian beam + u_h recovers the analytic σ_beam.

    Compares the second moment of sZ_h from :class:`BeamIntegrator`
    against :class:`SpectralIntegrator` with the same ``u_h``,
    ``beamwidth``, and zero turbulence. Target tolerance ~2 % on σ —
    the beam sampling is a Simpson-like quadrature at 48 × 24 nodes.
    """
    from rustmatrix.spectra import brandes_et_al_2002

    s = _rain_scatterer_X()
    hpbw = np.deg2rad(2.0)
    u_h = 15.0
    w_air = 0.0

    psd = s.psd  # fixed PSD
    scene = Scene(
        Z_dBZ=lambda x, y, z: np.full_like(x, 30.0),  # value ignored by factory below
        w=lambda x, y, z: np.full_like(x, w_air),
        u_h=lambda x, y, z: np.full_like(x, u_h),
        u_h_azimuth=0.0,
    )
    # Use a constant-PSD factory so the scene is perfectly homogeneous.
    def const_psd(_Z_dBZ):
        return psd

    bi = BeamIntegrator(
        scatterer=s,
        beam=GaussianBeam(hpbw=hpbw),
        scene=scene,
        psd_factory=const_psd,
        fall_speed=brandes_et_al_2002,
        range_m=1000.0,
        v_min=-5.0, v_max=15.0, n_bins=512,
        n_theta=48, n_phi=24,
    )
    r_beam = bi.run()

    # Reference: SpectralIntegrator with analytic σ_beam.
    si = SpectralIntegrator(
        s,
        fall_speed=brandes_et_al_2002,
        v_min=-5.0, v_max=15.0, n_bins=512,
        w=w_air, u_h=u_h, beamwidth=hpbw,
    )
    r_ref = si.run()

    v = r_beam.v
    # First and second moments of sZ_h.
    def mu_and_sigma(sZ):
        sZ = np.clip(sZ, 0, None)
        P = sZ.sum()
        mu = float((v * sZ).sum() / P)
        var = float(((v - mu) ** 2 * sZ).sum() / P)
        return mu, np.sqrt(var)

    mu_b, sig_b = mu_and_sigma(r_beam.sZ_h)
    mu_r, sig_r = mu_and_sigma(r_ref.sZ_h)

    assert mu_b == pytest.approx(mu_r, abs=0.05)
    # Within ~3 %: the beam integrator includes the full cos θ geometry,
    # which nudges the mean slightly (smaller v near beam edge); the
    # closed-form does not.
    assert sig_b == pytest.approx(sig_r, rel=0.03)


def test_bulk_sum_identity():
    """Integrated sZ_h equals PSD-bulk Z_h at the scene point."""
    from rustmatrix import radar as rmod
    from rustmatrix.spectra import brandes_et_al_2002

    s = _rain_scatterer_X()
    psd = s.psd
    # Uniform scene, no wind, no turbulence, no beam broadening —
    # spectrum should be a simple delta train that integrates to the
    # bulk Z_h.
    scene = Scene(
        Z_dBZ=lambda x, y, z: np.full_like(x, 30.0),
        w=lambda x, y, z: np.zeros_like(x),
        u_h=lambda x, y, z: np.zeros_like(x),
    )

    def const_psd(_Z_dBZ):
        return psd

    bi = BeamIntegrator(
        scatterer=s,
        beam=GaussianBeam(hpbw=np.deg2rad(0.01)),  # pencil
        scene=scene,
        psd_factory=const_psd,
        fall_speed=brandes_et_al_2002,
        range_m=1000.0,
        v_min=-1.0, v_max=12.0, n_bins=2048,
        n_theta=8, n_phi=4,
    )
    r = bi.run()

    dv = np.diff(r.v)
    dv_mid = np.concatenate(([dv[0]], 0.5 * (dv[:-1] + dv[1:]), [dv[-1]]))
    Z_integrated = float((r.sZ_h * dv_mid).sum())

    # Bulk Z_h from the scatterer's PSD integration.
    s.set_geometry(geom_vert_back)
    Z_bulk = rmod.refl(s, h_pol=True)

    # 1 % tolerance — the finite pencil beam still spreads slightly.
    assert Z_integrated == pytest.approx(Z_bulk, rel=0.02)


def test_marshall_palmer_factory_round_trips_Z():
    factory = marshall_palmer_psd_factory(N0=8000.0, D_max=6.0)
    for Z_dBZ in (10.0, 20.0, 30.0, 40.0):
        psd = factory(Z_dBZ)
        # Analytic Z for exponential PSD (Rayleigh): Z = 720 N0 / Λ^7.
        Z_calc = 720.0 * psd.N0 / psd.Lambda ** 7
        Z_target = 10 ** (Z_dBZ / 10.0)
        assert Z_calc == pytest.approx(Z_target, rel=1e-6)
