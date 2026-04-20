"""Tutorial 11 — Dual-frequency signatures across hydrometeor classes (Honeyager 2013).

Reference
---------
Honeyager, R., 2013: Investigating the use of the T-matrix method as a
proxy for the discrete dipole approximation, M.S. thesis, Florida State
University.

Physics question
----------------
Honeyager's thesis argues that a single T-matrix spheroid — parameterised
by (effective density ρ_eff, axis ratio) — reproduces the single-
scattering properties of geometrically-complex ice habits (bullet
rosettes, plates, dendrites, aggregates) computed with the discrete
dipole approximation, provided χ = 2πr/λ and ρ_eff are right. That
collapses the zoo of habits, to first order, onto a two-parameter family
that `rustmatrix` already supports.

This script steps through four representative classes:

    class                ρ_eff [g/cm³]   axis ratio
    --------------------------------------------------
    rain                 1.00 (water)    Thurai 2007
    low-density agg.     0.10            0.70
    graupel              0.50            0.90
    high-density ice     0.90            1.00

and reports:
1. Single-particle σ_b(D) at X and W bands.
2. Single-particle DWR(D) using Ze normalisation (0 dB in Rayleigh),
   plus the D where each class first walks out of Rayleigh.
3. Bulk DWR vs. median-volume diameter D₀ across an exponential-PSD
   slope sweep.
4. Dual-frequency Doppler spectra and sDWR(v) contrasting a low-density
   aggregate PSD against a graupel PSD tuned for similar X-band Z_h.
"""

from __future__ import annotations

import numpy as np

from rustmatrix import Scatterer, SpectralIntegrator, radar, spectra
from rustmatrix.psd import ExponentialPSD, PSDIntegrator
from rustmatrix.refractive import m_w_10C, mi
from rustmatrix.tmatrix_aux import (K_w_sqr, dsr_thurai_2007,
                                    geom_vert_back, wl_X, wl_W)


CLASSES = {
    "rain":       dict(rho=1.00, axis_ratio=None),   # Thurai below
    "low-ρ agg":  dict(rho=0.10, axis_ratio=0.70),
    "graupel":    dict(rho=0.50, axis_ratio=0.90),
    "high-ρ ice": dict(rho=0.90, axis_ratio=1.00),
}


def refractive_index(wl: float, cls: str) -> complex:
    return m_w_10C[wl] if cls == "rain" else mi(wl, CLASSES[cls]["rho"])


def class_axis_ratio(cls: str, D_mm: float) -> float:
    ar = CLASSES[cls]["axis_ratio"]
    return (1.0 / dsr_thurai_2007(D_mm)) if ar is None else ar


def sigma_b(wl: float, cls: str, D_mm: float) -> float:
    s = Scatterer(radius=D_mm / 2, wavelength=wl,
                  m=refractive_index(wl, cls),
                  Kw_sqr=K_w_sqr[wl],
                  axis_ratio=class_axis_ratio(cls, D_mm),
                  ddelt=1e-4, ndgs=2)
    s.set_geometry(geom_vert_back)
    return radar.radar_xsect(s)


def single_Ze(sig: np.ndarray, wl: float, Kw2: float) -> np.ndarray:
    return wl**4 / (np.pi**5 * Kw2) * sig


def bulk_Zh(wl: float, cls: str, N0: float, lam: float, Dmax: float) -> float:
    s = Scatterer(wavelength=wl, m=refractive_index(wl, cls),
                  Kw_sqr=K_w_sqr[wl], ddelt=1e-4, ndgs=2)
    integ = PSDIntegrator()
    integ.D_max = Dmax
    integ.num_points = 128
    integ.geometries = (geom_vert_back,)
    if CLASSES[cls]["axis_ratio"] is None:
        integ.axis_ratio_func = lambda D: 1.0 / dsr_thurai_2007(D)
    else:
        s.axis_ratio = CLASSES[cls]["axis_ratio"]
    s.psd_integrator = integ
    s.psd_integrator.init_scatter_table(s)
    s.psd = ExponentialPSD(N0=N0, Lambda=lam, D_max=Dmax)
    s.set_geometry(geom_vert_back)
    return radar.refl(s)


def build_spectra_scatterer(cls: str, N0: float, lam: float,
                            Dmax: float, wl: float) -> Scatterer:
    s = Scatterer(wavelength=wl, m=refractive_index(wl, cls),
                  Kw_sqr=K_w_sqr[wl], ddelt=1e-4, ndgs=2)
    integ = PSDIntegrator()
    integ.D_max = Dmax
    integ.num_points = 256
    integ.geometries = (geom_vert_back,)
    if CLASSES[cls]["axis_ratio"] is None:
        integ.axis_ratio_func = lambda D: 1.0 / dsr_thurai_2007(D)
    else:
        s.axis_ratio = CLASSES[cls]["axis_ratio"]
    s.psd_integrator = integ
    s.psd_integrator.init_scatter_table(s)
    s.psd = ExponentialPSD(N0=N0, Lambda=lam, D_max=Dmax)
    return s


def main() -> None:
    # 1 + 2: single-particle σ_b(D) and DWR(D)
    D_grid = np.linspace(0.2, 8.0, 100)
    sigma = {cls: {b: np.array([sigma_b(wl, cls, D) for D in D_grid])
                   for b, wl in [("X", wl_X), ("W", wl_W)]}
             for cls in CLASSES}

    print("Honeyager (2013) — dual-frequency signatures of hydrometeor classes")
    print("-" * 72)
    print("Approximate D at which single-particle DWR(Ze) crosses +3 dB:")
    for cls in CLASSES:
        zeX = single_Ze(sigma[cls]["X"], wl_X, K_w_sqr[wl_X])
        zeW = single_Ze(sigma[cls]["W"], wl_W, K_w_sqr[wl_W])
        dwr = 10 * np.log10(zeX / zeW)
        above = np.where(dwr > 3.0)[0]
        label = f"{D_grid[above[0]]:.2f} mm" if len(above) else "> 8 mm"
        print(f"  {cls:<12} {label}")
    print()

    # 3: bulk DWR vs D0, sweep Λ
    Lambdas = np.linspace(1.2, 9.0, 10)
    D0s = 3.67 / Lambdas
    DMAX = 8.0
    N0 = 8e3

    print("Bulk DWR vs D0 (N0 = 8e3 m⁻³ mm⁻¹, D_max = 8 mm):")
    print(f"  {'D0 [mm]':>8}" + "".join(f"  {cls:>12}" for cls in CLASSES))
    rows = []
    for lam in Lambdas:
        row = []
        for cls in CLASSES:
            zX = bulk_Zh(wl_X, cls, N0, lam, DMAX)
            zW = bulk_Zh(wl_W, cls, N0, lam, DMAX)
            row.append(10 * np.log10(zX / zW))
        rows.append(row)
    for D0, row in zip(D0s, rows):
        print(f"  {D0:>8.2f}" + "".join(f"  {v:>12.2f}" for v in row))
    print()

    # 4: dual-frequency spectra, aggregate vs graupel
    agg_X = build_spectra_scatterer("low-ρ agg", N0=2e4, lam=2.0, Dmax=6.0, wl=wl_X)
    agg_W = build_spectra_scatterer("low-ρ agg", N0=2e4, lam=2.0, Dmax=6.0, wl=wl_W)
    gra_X = build_spectra_scatterer("graupel",   N0=5e3, lam=3.0, Dmax=4.0, wl=wl_X)
    gra_W = build_spectra_scatterer("graupel",   N0=5e3, lam=3.0, Dmax=4.0, wl=wl_W)

    fall_agg = spectra.fall_speed.locatelli_hobbs_1974_aggregates
    fall_gra = spectra.fall_speed.locatelli_hobbs_1974_graupel_hex
    turb = spectra.GaussianTurbulence(0.2)

    def run(sc: Scatterer, fall):
        return SpectralIntegrator(
            sc, fall_speed=fall, turbulence=turb,
            v_min=-0.5, v_max=4.0, n_bins=1024,
            geometry_backscatter=geom_vert_back,
        ).run()

    r_agg_X = run(agg_X, fall_agg); r_agg_W = run(agg_W, fall_agg)
    r_gra_X = run(gra_X, fall_gra); r_gra_W = run(gra_W, fall_gra)

    print("Dual-frequency Doppler spectra (aggregates vs. graupel):")
    print(f"  {'population':<15} {'Z_h X [dBZ]':>12} {'Z_h W [dBZ]':>12} "
          f"{'DWR [dB]':>10}")
    for name, rX, rW in [("low-ρ agg", r_agg_X, r_agg_W),
                          ("graupel",   r_gra_X, r_gra_W)]:
        zX = 10 * np.log10(np.trapezoid(rX.sZ_h, rX.v))
        zW = 10 * np.log10(np.trapezoid(rW.sZ_h, rW.v))
        print(f"  {name:<15} {zX:>12.2f} {zW:>12.2f} {zX - zW:>10.2f}")


if __name__ == "__main__":
    main()
