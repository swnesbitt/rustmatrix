"""Tutorial 07 — Doppler + polarimetric spectra of rain at X-band.

Physics question
----------------
Profilers, cloud radars, and modern disdrometers measure *spectra* — the
Doppler-velocity-resolved version of the polarimetric observables. The
spectrum of a falling rain PSD is a change of variable from drop
diameter to terminal velocity, convolved with turbulence and smeared by
the finite antenna beamwidth. This tutorial walks through the four
knobs that shape the spectrum:

1. No turbulence, no beam broadening → delta-function spectrum binned
   by ``v_t(D)``. Sanity: integrating it recovers the bulk Z_h exactly.
2. Gaussian turbulence with constant σ_t → the classical broadening.
3. Particle-inertia turbulence (Zeng et al. 2023) → heavy drops
   "feel" less turbulence than small ones.
4. Beam broadening via horizontal wind and finite beamwidth.

What this script does
---------------------
* Builds a C-rainfall-like X-band rain scatterer.
* Computes spectra under the four configurations above.
* Verifies that the bin-by-bin spectrum integrated over v reproduces
  the bulk radar observables from :mod:`rustmatrix.radar`.
"""

from __future__ import annotations

import numpy as np

from rustmatrix import Scatterer, SpectralIntegrator, radar, spectra
from rustmatrix.psd import GammaPSD, PSDIntegrator
from rustmatrix.refractive import m_w_10C
from rustmatrix.tmatrix_aux import (
    K_w_sqr,
    dsr_thurai_2007,
    geom_vert_back,
    geom_vert_forw,
    wl_X,
)


def build_rain() -> Scatterer:
    s = Scatterer(
        wavelength=wl_X,
        m=m_w_10C[wl_X],
        Kw_sqr=K_w_sqr[wl_X],
        ddelt=1e-4,
        ndgs=2,
    )
    integ = PSDIntegrator()
    integ.D_max = 6.0
    integ.num_points = 64
    integ.axis_ratio_func = lambda D: 1.0 / dsr_thurai_2007(D)
    integ.geometries = (geom_vert_back, geom_vert_forw)
    s.psd_integrator = integ
    s.psd_integrator.init_scatter_table(s)
    s.psd = GammaPSD(D0=1.5, Nw=8e3, mu=4, D_max=6.0)
    return s


def run_config(rain, label, turbulence, u_h=0.0, beamwidth=0.0):
    integ = SpectralIntegrator(
        rain,
        fall_speed=spectra.fall_speed.atlas_srivastava_sekhon_1973,
        turbulence=turbulence,
        v_min=-1.0, v_max=12.0, n_bins=1024,
        u_h=u_h,
        beamwidth=beamwidth,
        geometry_backscatter=geom_vert_back,
        geometry_forward=geom_vert_forw,
    )
    res = integ.run()
    # Bulk-sum sanity: trapezoidal integral over v should equal the bulk Z_h.
    rain.set_geometry(geom_vert_back)
    Zh_bulk = radar.refl(rain)
    Zh_spec_int = np.trapezoid(res.sZ_h, res.v)
    print(
        f"{label:<34} Z_h_bulk = {Zh_bulk:.3e}  "
        f"∫sZ_h dv = {Zh_spec_int:.3e}  "
        f"rel. diff = {abs(Zh_bulk - Zh_spec_int) / Zh_bulk:.2e}"
    )
    return res


def main() -> None:
    rain = build_rain()

    print(
        f"{'configuration':<34} {'bulk / spectrum-integrated Z_h check'}"
    )
    print("-" * 82)

    run_config(rain, "no turbulence, no beam",
               spectra.NoTurbulence())
    run_config(rain, "Gaussian turbulence σ=0.5 m/s",
               spectra.GaussianTurbulence(0.5))
    run_config(rain, "Zeng 2023 inertia (ε=1e-3)",
               spectra.InertialZeng2023(sigma_air=0.5, epsilon=1e-3))
    run_config(rain, "beam broadening (u_h=8, θ_b=1°)",
               spectra.NoTurbulence(),
               u_h=8.0, beamwidth=np.deg2rad(1.0))

    # Re-derive bulk ρ_hv, Z_dr, K_dp from the spectrum — non-linear but
    # exact when you work from the *summed* S and Z matrices.
    res = run_config(rain, "Gaussian σ=0.3 (full re-derivation)",
                     spectra.GaussianTurbulence(0.3))
    bulk = res.collapse_to_bulk()
    print("\nRe-derived bulk observables from the spectrum:")
    print(
        f"  Z_dr  = {10 * np.log10(radar.Zdr(bulk)):+.3f} dB"
        f" (bulk: {10 * np.log10(radar.Zdr(rain)):+.3f} dB)"
    )
    print(
        f"  ρ_hv  = {radar.rho_hv(bulk):.5f}"
        f" (bulk: {radar.rho_hv(rain):.5f})"
    )


if __name__ == "__main__":
    main()
