"""Tutorial 08 — Spectral polarimetry of a rain + ice mixture.

Physics question
----------------
Rain falls at ~5 m/s, unrimed aggregates at ~1 m/s. In mixed-phase
volumes the two populations show up as distinct *modes* in the Doppler
spectrum. The magic of spectral polarimetry is that each velocity bin
carries its own ``sZ_dr(v)``, ``sρ_hv(v)``, and ``sδ_hv(v)``, so the
"melting" and "ice" components can be diagnosed separately even when
the bulk radar variables average over both.

``HydroMix`` + ``SpectralIntegrator`` handle this natively: each
component gets its own fall-speed and turbulence model, and the
integrator sums per-component spectra on a shared velocity grid
(linearity in N(D) applied per velocity bin).

What this script does
---------------------
1. Builds an X-band rain scatterer and a low-density oriented-ice
   scatterer.
2. Assembles a ``HydroMix`` from both species.
3. Runs :class:`SpectralIntegrator` with per-component fall speeds and
   turbulence.
4. Reports the ice-mode peak velocity, rain-mode peak velocity, and
   the drop in ρ_hv between the modes — the signature of mixed-phase
   structure.
"""

from __future__ import annotations

import numpy as np

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
        ddelt=1e-4, ndgs=2,
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


def build_ice() -> Scatterer:
    s = Scatterer(
        wavelength=wl_X,
        m=mi(wl_X, 0.2),
        Kw_sqr=K_w_sqr[wl_X],
        axis_ratio=0.6,
        ddelt=1e-4, ndgs=2,
    )
    integ = PSDIntegrator()
    integ.D_max = 8.0
    integ.num_points = 64
    integ.geometries = (geom_vert_back, geom_vert_forw)
    s.psd_integrator = integ
    s.psd_integrator.init_scatter_table(s)
    s.psd = ExponentialPSD(N0=5e3, Lambda=2.0, D_max=8.0)
    return s


def main() -> None:
    rain = build_rain()
    ice = build_ice()

    mix = HydroMix([
        MixtureComponent(rain, rain.psd, "rain"),
        MixtureComponent(ice,  ice.psd,  "ice"),
    ])

    integ = SpectralIntegrator(
        mix,
        component_kinematics={
            "rain": (
                spectra.fall_speed.atlas_srivastava_sekhon_1973,
                spectra.GaussianTurbulence(0.2),
            ),
            "ice":  (
                spectra.fall_speed.locatelli_hobbs_1974_aggregates,
                spectra.GaussianTurbulence(0.2),
            ),
        },
        v_min=-1.0, v_max=12.0, n_bins=1024,
        geometry_backscatter=geom_vert_back,
        geometry_forward=geom_vert_forw,
    )
    res = integ.run()

    # Identify the two modes by looking for the ice (v < 2) and rain (v > 2) peaks.
    ice_mask = res.v < 2.0
    rain_mask = res.v > 2.0

    v_ice_peak = res.v[ice_mask][int(np.argmax(res.sZ_h[ice_mask]))]
    v_rain_peak = res.v[rain_mask][int(np.argmax(res.sZ_h[rain_mask]))]

    # Bimodal valley: find the minimum of sZ_h between the two peaks.
    valley_mask = (res.v > v_ice_peak) & (res.v < v_rain_peak)
    v_valley = res.v[valley_mask][int(np.argmin(res.sZ_h[valley_mask]))]
    rho_valley = res.srho_hv[valley_mask][int(np.argmin(res.sZ_h[valley_mask]))]

    print("Rain + ice bimodal Doppler spectrum at X-band, vertical pointing")
    print("-" * 72)
    print(f"  ice-mode peak  : v = {v_ice_peak:.2f} m/s")
    print(f"  rain-mode peak : v = {v_rain_peak:.2f} m/s")
    print(f"  inter-mode valley: v = {v_valley:.2f} m/s, ρ_hv = {rho_valley:.4f}")

    # Bulk-sum identity round-trip.
    bulk = res.collapse_to_bulk()
    mix.set_geometry(geom_vert_back)
    print("\nBulk ↔ spectrum-integrated identity:")
    print(
        f"  Z_h bulk   = {10 * np.log10(radar.refl(mix)):.3f} dBZ   "
        f"from spectrum = {10 * np.log10(radar.refl(bulk)):.3f} dBZ"
    )
    print(
        f"  Z_dr bulk  = {10 * np.log10(radar.Zdr(mix)):+.3f} dB    "
        f"from spectrum = {10 * np.log10(radar.Zdr(bulk)):+.3f} dB"
    )
    print(
        f"  ρ_hv bulk  = {radar.rho_hv(mix):.5f}       "
        f"from spectrum = {radar.rho_hv(bulk):.5f}"
    )


if __name__ == "__main__":
    main()
