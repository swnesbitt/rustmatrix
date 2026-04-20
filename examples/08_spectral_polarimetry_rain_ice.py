"""Tutorial 08 — Dual-frequency non-Rayleigh snowfall spectra (Billault-Roux 2023).

Reference
---------
Billault-Roux, A.-C., Ghiggi, G., Jaffeux, L., Martini, A., Viltard, N.,
and Berne, A., 2023: Dual-frequency spectral radar retrieval of snowfall
microphysics: a physics-driven deep-learning approach, *Atmos. Meas.
Tech.*, 16, 911–931, doi:10.5194/amt-16-911-2023.

Physics question
----------------
At cloud-radar frequencies small, slow-falling snow particles are still
Rayleigh, so their X-band and W-band spectral reflectivities agree. The
large, fast-falling particles, however, are non-Rayleigh at W-band, so
their W-band spectral reflectivity is smaller than X-band. The spectral
dual-wavelength ratio

    sDWR(v) = 10·log₁₀( sZ_X(v) / sZ_W(v) )

stays near 0 at low velocities and rises to several dB at high velocities
— a direct fingerprint of the large-particle tail of the PSD, largely
disentangled from turbulence and wind offsets.

What this script does
---------------------
1. Builds identical snow scatterers at X-band and W-band (ρ = 0.2 g/cm³,
   axis ratio 0.6, exponential PSD with Λ = 0.8 mm⁻¹, D_max = 10 mm).
2. Tabulates single-particle σ_b(D) at both bands to show the
   non-Rayleigh onset around D ≈ 3 mm.
3. Runs both scatterers through `SpectralIntegrator` with the Locatelli–
   Hobbs aggregate fall-speed and σ_t = 0.2 m/s turbulence.
4. Reports bulk DWR plus sDWR(v) at selected velocities.
"""

from __future__ import annotations

import numpy as np

from rustmatrix import Scatterer, SpectralIntegrator, radar, spectra
from rustmatrix.psd import ExponentialPSD, PSDIntegrator
from rustmatrix.refractive import mi
from rustmatrix.tmatrix_aux import K_w_sqr, geom_vert_back, wl_X, wl_W


# PSD and habit grounded in the ICE-GENESIS 23 January 2021 case of
# Billault-Roux et al. 2023 (Fig. 5 snowfall layer):
#   - oblate aggregates, ρ_ice = 0.1 g/cm³ (low-density, mixed-habit),
#     axis ratio 0.6, D_max = 5 mm
#   - exponential PSD, N0 = 2e4 m⁻³ mm⁻¹, Λ = 2.5 mm⁻¹
#   - D0 = 3.67/Λ ≈ 1.5 mm, IWC = π ρ_ice N0 / Λ⁴ ≈ 0.16 g/m³
# These knobs place Z_h(X) near 17 dBZ — a moderate aggregation layer,
# consistent with the paper's Fig. 5 observations — not the 50 dBZ
# outlier produced by Λ = 0.8 mm⁻¹ and D_max = 10 mm.
RHO_ICE = 0.1
AXIS_RATIO = 0.6
D_MAX = 5.0
N0 = 2e4
LAMBDA = 2.5
V_MIN, V_MAX = -2.0, 4.0
N_BINS = 1024


def build_snow(wl: float) -> Scatterer:
    s = Scatterer(wavelength=wl, m=mi(wl, RHO_ICE),
                  Kw_sqr=K_w_sqr[wl], axis_ratio=AXIS_RATIO,
                  ddelt=1e-4, ndgs=2)
    integ = PSDIntegrator()
    integ.D_max = D_MAX
    integ.num_points = 256
    integ.geometries = (geom_vert_back,)
    s.psd_integrator = integ
    s.psd_integrator.init_scatter_table(s)
    s.psd = ExponentialPSD(N0=N0, Lambda=LAMBDA, D_max=D_MAX)
    return s


def sigma_b(wl: float, D_mm: float) -> float:
    s = Scatterer(radius=D_mm / 2, wavelength=wl, m=mi(wl, RHO_ICE),
                  Kw_sqr=K_w_sqr[wl], axis_ratio=AXIS_RATIO,
                  ddelt=1e-4, ndgs=2)
    s.set_geometry(geom_vert_back)
    return radar.radar_xsect(s)


def main() -> None:
    snow_X = build_snow(wl_X)
    snow_W = build_snow(wl_W)

    fall = spectra.fall_speed.locatelli_hobbs_1974_aggregates
    turb = spectra.GaussianTurbulence(0.2)

    def run(sc: Scatterer):
        return SpectralIntegrator(
            sc, fall_speed=fall, turbulence=turb,
            v_min=V_MIN, v_max=V_MAX, n_bins=N_BINS,
            geometry_backscatter=geom_vert_back,
        ).run()

    r_X = run(snow_X)
    r_W = run(snow_W)

    def dBZ(x: np.ndarray) -> np.ndarray:
        return 10 * np.log10(np.maximum(x, 1e-12))

    # σ_b(D) spot-check at a few diameters.
    D_probe = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0])

    Z_X = np.trapezoid(r_X.sZ_h, r_X.v)
    Z_W = np.trapezoid(r_W.sZ_h, r_W.v)

    print("Billault-Roux et al. (2023) — dual-frequency snowfall spectra")
    print("-" * 72)
    print(f"  exponential PSD: N0 = {N0:g} m⁻³ mm⁻¹, Λ = {LAMBDA} mm⁻¹, "
          f"D_max = {D_MAX} mm")
    print(f"  habit: oblate, ρ = {RHO_ICE} g/cm³, axis ratio {AXIS_RATIO}")
    print()
    print("Single-particle σ_b(D) [mm²]:")
    print(f"  {'D [mm]':>8} {'X-band':>12} {'W-band':>12} {'ratio':>10}")
    for D in D_probe:
        sX = sigma_b(wl_X, D); sW = sigma_b(wl_W, D)
        print(f"  {D:>8.2f} {sX:>12.3e} {sW:>12.3e} {sX / sW:>10.2f}")
    print()
    print(f"  bulk Z_h (X-band) = {10 * np.log10(Z_X):.2f} dBZ")
    print(f"  bulk Z_h (W-band) = {10 * np.log10(Z_W):.2f} dBZ")
    print(f"  bulk DWR          = {10 * np.log10(Z_X / Z_W):.2f} dB")
    print()
    print("sDWR(v) at selected velocities:")
    for vs in (0.3, 0.5, 0.8, 1.0, 1.3, 1.6):
        i = int(np.argmin(np.abs(r_X.v - vs)))
        sdwr = dBZ(r_X.sZ_h[i]) - dBZ(r_W.sZ_h[i])
        print(f"  v = {r_X.v[i]:.2f} m/s   sDWR = {float(sdwr):+.2f} dB")


if __name__ == "__main__":
    main()
