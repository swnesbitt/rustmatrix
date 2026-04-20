"""Tutorial 10 — Supercooled liquid water vs. snow at cloud-radar frequencies.

Reference
---------
Billault-Roux, A.-C., Georgakaki, P., Grazioli, J., Romanens, G.,
Sotiropoulou, G., Phillips, V., Nenes, A., and Berne, A., 2023:
Distinct secondary ice production processes observed in radar Doppler
spectra: insights from a case study, *Atmos. Chem. Phys.*, 23,
10207–10234, doi:10.5194/acp-23-10207-2023.

Physics question
----------------
Mixed-phase cloud volumes contain supercooled liquid droplets (SLW) and
ice particles side by side. The two populations have completely
different scattering and kinematic signatures:

* **SLW**: small (≲200 µm), spherical, cold refractive index of water,
  falls at a few cm/s. Rayleigh at both X- and W-band ⇒ very low Z_h,
  Z_dr = 0 dB, no polarimetric signature.
* **Snow**: millimetre-scale oblate low-density aggregates, falls at
  ~0.5–1 m/s. Moderately non-Rayleigh at W-band ⇒ higher Z_h, small
  positive Z_dr from the habit anisotropy.

What this script does
---------------------
1. Builds a SLW scatterer (spherical water drops, 0 °C refractive index,
   gamma PSD centred near 30 µm) and a snow scatterer (oblate,
   ρ = 0.2 g/cm³, exponential PSD to 8 mm).
2. Tabulates σ_b(D) at X- and W-band for both populations to contrast
   Rayleigh (SLW) vs. non-Rayleigh (large snow) behaviour.
3. Reports bulk Z_h and DWR.
4. Assembles a `HydroMix` and runs it through `SpectralIntegrator` with
   per-component fall-speed + turbulence at W-band, producing the
   bimodal SLW + snow Doppler spectrum that motivates the Billault-Roux
   et al. (2023) case-study analysis.
"""

from __future__ import annotations

import numpy as np

from rustmatrix import (HydroMix, MixtureComponent, Scatterer,
                        SpectralIntegrator, radar, spectra)
from rustmatrix.psd import ExponentialPSD, GammaPSD, PSDIntegrator
from rustmatrix.refractive import m_w_0C, mi
from rustmatrix.tmatrix_aux import K_w_sqr, geom_vert_back, wl_X, wl_W


SLW_DMAX = 0.2    # mm
SNOW_DMAX = 8.0
RHO_SNOW = 0.2
AXIS_SNOW = 0.6


def build_slw(wl: float) -> Scatterer:
    s = Scatterer(wavelength=wl, m=m_w_0C[wl],
                  Kw_sqr=K_w_sqr[wl], axis_ratio=1.0,
                  ddelt=1e-4, ndgs=2)
    integ = PSDIntegrator()
    integ.D_max = SLW_DMAX
    integ.num_points = 128
    integ.geometries = (geom_vert_back,)
    s.psd_integrator = integ
    s.psd_integrator.init_scatter_table(s)
    s.psd = GammaPSD(D0=0.03, Nw=1e11, mu=4, D_max=SLW_DMAX)
    return s


def build_snow(wl: float) -> Scatterer:
    s = Scatterer(wavelength=wl, m=mi(wl, RHO_SNOW),
                  Kw_sqr=K_w_sqr[wl], axis_ratio=AXIS_SNOW,
                  ddelt=1e-4, ndgs=2)
    integ = PSDIntegrator()
    integ.D_max = SNOW_DMAX
    integ.num_points = 256
    integ.geometries = (geom_vert_back,)
    s.psd_integrator = integ
    s.psd_integrator.init_scatter_table(s)
    s.psd = ExponentialPSD(N0=5e3, Lambda=1.0, D_max=SNOW_DMAX)
    return s


def sigma_b(wl: float, D_mm: float, m: complex, axis_ratio: float) -> float:
    s = Scatterer(radius=D_mm / 2, wavelength=wl, m=m,
                  Kw_sqr=K_w_sqr[wl], axis_ratio=axis_ratio,
                  ddelt=1e-4, ndgs=2)
    s.set_geometry(geom_vert_back)
    return radar.radar_xsect(s)


def bulk_Zh_dBZ(s: Scatterer) -> float:
    s.set_geometry(geom_vert_back)
    return 10 * np.log10(radar.refl(s))


def main() -> None:
    slw_X = build_slw(wl_X)
    slw_W = build_slw(wl_W)
    snow_X = build_snow(wl_X)
    snow_W = build_snow(wl_W)

    print("Billault-Roux et al. (2023) ACP — SLW vs. snow at cloud-radar frequencies")
    print("-" * 72)
    print("Single-particle σ_b(D) [mm²]:")
    print(f"  {'pop':<6} {'D [mm]':>8} {'X-band':>12} {'W-band':>12}")
    for D in (0.03, 0.1):
        sX = sigma_b(wl_X, D, m_w_0C[wl_X], 1.0)
        sW = sigma_b(wl_W, D, m_w_0C[wl_W], 1.0)
        print(f"  {'SLW':<6} {D:>8.3f} {sX:>12.3e} {sW:>12.3e}")
    for D in (0.5, 2.0, 5.0, 8.0):
        sX = sigma_b(wl_X, D, mi(wl_X, RHO_SNOW), AXIS_SNOW)
        sW = sigma_b(wl_W, D, mi(wl_W, RHO_SNOW), AXIS_SNOW)
        print(f"  {'snow':<6} {D:>8.3f} {sX:>12.3e} {sW:>12.3e}")
    print()

    print(f"  {'':<8} {'Z_h X [dBZ]':>12} {'Z_h W [dBZ]':>12} {'DWR [dB]':>10}")
    for name, sX, sW in (("SLW", slw_X, slw_W), ("snow", snow_X, snow_W)):
        zX = bulk_Zh_dBZ(sX); zW = bulk_Zh_dBZ(sW)
        print(f"  {name:<8} {zX:>12.2f} {zW:>12.2f} {zX - zW:>10.2f}")
    print()

    mix = HydroMix([
        MixtureComponent(slw_W, slw_W.psd, "slw"),
        MixtureComponent(snow_W, snow_W.psd, "snow"),
    ])
    kin = {
        "slw":  (lambda D: 0.003 * (D / 0.01) ** 2,
                 spectra.GaussianTurbulence(0.1)),
        "snow": (spectra.fall_speed.locatelli_hobbs_1974_aggregates,
                 spectra.GaussianTurbulence(0.2)),
    }
    integ = SpectralIntegrator(
        mix, component_kinematics=kin,
        v_min=-1.0, v_max=3.0, n_bins=1024,
        geometry_backscatter=geom_vert_back,
        noise="realistic",
    ).run()

    # Find each mode's peak on its own single-species spectrum. Snow is
    # ~50 dB louder than SLW, so its low-v tail would dominate any argmax
    # over the combined (noise-added) spectrum near v ≈ 0.
    slw_only = SpectralIntegrator(
        slw_W, fall_speed=kin["slw"][0], turbulence=kin["slw"][1],
        v_min=-1.0, v_max=3.0, n_bins=1024,
        geometry_backscatter=geom_vert_back,
    ).run()
    snow_only = SpectralIntegrator(
        snow_W, fall_speed=kin["snow"][0], turbulence=kin["snow"][1],
        v_min=-1.0, v_max=3.0, n_bins=1024,
        geometry_backscatter=geom_vert_back,
    ).run()

    v_slw = slw_only.v[int(np.argmax(slw_only.sZ_h))]
    v_snow = snow_only.v[int(np.argmax(snow_only.sZ_h))]

    print("Bimodal W-band Doppler spectrum (SLW + snow HydroMix):")
    print(f"  SLW mode peak  v = {v_slw:+.3f} m/s")
    print(f"  snow mode peak v = {v_snow:+.3f} m/s")


if __name__ == "__main__":
    main()
