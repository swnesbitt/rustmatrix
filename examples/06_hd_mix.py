"""Tutorial 06 — a hydrometeor mixture at C-band (rain + oriented ice).

Physics question
----------------
Real radar volumes rarely contain a single hydrometeor species. Melting
layers, convective updrafts, and mixed-phase clouds all mix water drops
with ice in varying proportions. The combined polarimetric signature is
the incoherent sum of the per-species contributions — but the *non*-linear
observables (Z_dr, ρ_hv, δ_hv) cannot be averaged from per-species
values; they must be computed from the mixture's summed amplitude and
phase matrices.

What this script does
---------------------
1. Builds a C-band rain scatterer (Thurai 2007 drop-shape relation, 10 °C
   water refractive index) with its own PSD integrator.
2. Builds a C-band low-density oriented-ice scatterer (prolate, fixed
   axis ratio) with its own PSD integrator.
3. Assembles a :class:`rustmatrix.HydroMix` from both species and reads
   Z_h, Z_dr, K_dp, A_i, and ρ_hv for rain-only, ice-only, and the
   combined mixture through the usual :mod:`rustmatrix.radar` helpers.

Why this works
--------------
Both ``S`` (forward, K_dp / A_i) and ``Z`` (backscatter intensity, all
polarimetric observables) are linear in ``N(D)``, so summing them across
species is the physically correct incoherent-mixture recipe. Mixing
fractions live inside each species' N(D) directly.
"""

from __future__ import annotations

import numpy as np

from rustmatrix import HydroMix, MixtureComponent, Scatterer, radar
from rustmatrix import psd as rs_psd
from rustmatrix.refractive import m_w_10C, mi
from rustmatrix.tmatrix_aux import (
    K_w_sqr,
    dsr_thurai_2007,
    geom_horiz_back,
    geom_horiz_forw,
    wl_C,
)


def build_rain() -> Scatterer:
    s = Scatterer(
        wavelength=wl_C,
        m=m_w_10C[wl_C],
        Kw_sqr=K_w_sqr[wl_C],
        ddelt=1e-4,
        ndgs=2,
    )
    integ = rs_psd.PSDIntegrator()
    integ.D_max = 6.0
    integ.num_points = 64
    integ.axis_ratio_func = lambda D: 1.0 / dsr_thurai_2007(D)
    integ.geometries = (geom_horiz_back, geom_horiz_forw)
    s.psd_integrator = integ
    s.psd_integrator.init_scatter_table(s)
    return s


def build_ice() -> Scatterer:
    s = Scatterer(
        wavelength=wl_C,
        m=mi(wl_C, 0.2),       # low-density rimed-ice-ish aggregates
        Kw_sqr=K_w_sqr[wl_C],  # reference is still liquid water for Z
        axis_ratio=0.6,        # prolate
        ddelt=1e-4,
        ndgs=2,
    )
    integ = rs_psd.PSDIntegrator()
    integ.D_max = 8.0
    integ.num_points = 64
    integ.geometries = (geom_horiz_back, geom_horiz_forw)
    s.psd_integrator = integ
    s.psd_integrator.init_scatter_table(s)
    return s


def observables(scatterer_like) -> dict:
    """Run the standard C-band polarimetric read-out on any
    Scatterer-shaped object (Scatterer *or* HydroMix)."""
    scatterer_like.set_geometry(geom_horiz_back)
    Zh = radar.refl(scatterer_like, h_pol=True)
    Zdr = radar.Zdr(scatterer_like)
    rho = radar.rho_hv(scatterer_like)
    delta = np.degrees(radar.delta_hv(scatterer_like))
    scatterer_like.set_geometry(geom_horiz_forw)
    Kdp = radar.Kdp(scatterer_like)
    Ai = radar.Ai(scatterer_like, h_pol=True)
    return {
        "Zh_dBZ": 10 * np.log10(Zh),
        "Zdr_dB": 10 * np.log10(Zdr),
        "rho_hv": rho,
        "delta_hv_deg": delta,
        "Kdp_deg_per_km": Kdp,
        "Ai_dB_per_km": Ai,
    }


def main() -> None:
    rain = build_rain()
    ice = build_ice()

    rain_psd = rs_psd.GammaPSD(D0=1.5, Nw=8e3, mu=4, D_max=6.0)
    ice_psd = rs_psd.ExponentialPSD(N0=5e3, Lambda=2.0, D_max=8.0)

    # Rain-only: attach the rain PSD to the rain scatterer.
    rain.psd = rain_psd

    # Ice-only: ditto.
    ice.psd = ice_psd

    # Mixture: hand both (scatterer, psd) pairs to HydroMix.
    mix = HydroMix([
        MixtureComponent(rain, rain_psd, "rain"),
        MixtureComponent(ice, ice_psd, "ice"),
    ])

    cases = [("rain only", rain), ("ice only", ice), ("rain + ice (HydroMix)", mix)]

    print(
        f"{'case':<24} {'Z_h':>8} {'Z_dr':>8} {'ρ_hv':>8} "
        f"{'δ_hv':>8} {'K_dp':>10} {'A_i':>10}"
    )
    print(
        f"{'':<24} {'[dBZ]':>8} {'[dB]':>8} {'':>8} "
        f"{'[°]':>8} {'[°/km]':>10} {'[dB/km]':>10}"
    )
    print("-" * 82)
    for name, obj in cases:
        o = observables(obj)
        print(
            f"{name:<24} {o['Zh_dBZ']:>8.2f} {o['Zdr_dB']:>+8.3f} "
            f"{o['rho_hv']:>8.5f} {o['delta_hv_deg']:>+8.4f} "
            f"{o['Kdp_deg_per_km']:>+10.4f} {o['Ai_dB_per_km']:>10.5f}"
        )

    # Sanity: Z_h in linear units is additive across species.
    rain.set_geometry(geom_horiz_back)
    ice.set_geometry(geom_horiz_back)
    mix.set_geometry(geom_horiz_back)
    zh_sum = radar.refl(rain) + radar.refl(ice)
    zh_mix = radar.refl(mix)
    print(
        f"\nlinear Z_h additivity:  rain + ice = {zh_sum:.3e}   "
        f"mix = {zh_mix:.3e}   rel. diff = "
        f"{abs(zh_sum - zh_mix) / zh_mix:.2e}"
    )


if __name__ == "__main__":
    main()
