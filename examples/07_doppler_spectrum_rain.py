"""Tutorial 07 — W-band Mie Doppler spectrum in convective rain (Kollias 2002).

Reference
---------
Kollias, P., Albrecht, B. A., and Marks Jr., F. D., 2002: Cloud radar
observations of vertical drafts and microphysics in convective rain,
*J. Geophys. Res.*, 107, doi:10.1029/2001JD002033.

Physics question
----------------
At 94 GHz the raindrop backscattering cross-section σ_b(D) is no longer
Rayleigh — it rings through a sequence of Mie maxima and minima. Because
each drop falls at a deterministic terminal velocity v_t(D), those Mie
features map directly onto the observed Doppler spectrum. The first Mie
minimum appears at v ≈ 5.95 m/s (oblate) / 5.88 m/s (sphere) in still
air, so any displacement of the observed feature from that fiducial is a
direct measurement of the mean vertical air motion — independent of the
drop-size distribution.

What this script does
---------------------
1. Computes σ_b(D) at 94 GHz for oblate (Thurai 2007) and spherical drops
   (reproducing the paper's Figure 1).
2. Maps σ_b / (π r²) vs. v_t(D) to locate the first Mie minimum in
   velocity space (Figure 3).
3. Runs an exponential warm-rain DSD through `SpectralIntegrator` at
   W-band with two diameter samplings to highlight the delta-binning
   artifact.
4. Simulates a 1 m/s downward air motion and recovers it by tracking the
   shifted Mie minimum — the Kollias 2002 retrieval technique in a
   nutshell.
"""

from __future__ import annotations

import numpy as np

from rustmatrix import Scatterer, SpectralIntegrator, radar, spectra
from rustmatrix.psd import ExponentialPSD, PSDIntegrator
from rustmatrix.refractive import m_w_20C
from rustmatrix.tmatrix_aux import (K_w_sqr, dsr_thurai_2007,
                                     geom_vert_back, wl_W)


def build_drop(D_mm: float, oblate: bool = True) -> Scatterer:
    axis_ratio = 1.0 / dsr_thurai_2007(D_mm) if oblate else 1.0
    s = Scatterer(radius=D_mm / 2, wavelength=wl_W, m=m_w_20C[wl_W],
                  Kw_sqr=K_w_sqr[wl_W], axis_ratio=axis_ratio,
                  ddelt=1e-4, ndgs=2)
    s.set_geometry(geom_vert_back)
    return s


def build_rain_W(num_points: int = 256) -> Scatterer:
    s = Scatterer(wavelength=wl_W, m=m_w_20C[wl_W],
                  Kw_sqr=K_w_sqr[wl_W], ddelt=1e-4, ndgs=2)
    integ = PSDIntegrator()
    integ.D_max = 5.0
    integ.num_points = num_points
    integ.axis_ratio_func = lambda D: 1.0 / dsr_thurai_2007(D)
    integ.geometries = (geom_vert_back,)
    s.psd_integrator = integ
    s.psd_integrator.init_scatter_table(s)
    s.psd = ExponentialPSD(N0=8e3, Lambda=2.2, D_max=5.0)
    return s


def main() -> None:
    fall = spectra.fall_speed.atlas_srivastava_sekhon_1973

    D_grid = np.linspace(0.05, 5.0, 400)
    sigma_b_obl = np.array([radar.radar_xsect(build_drop(D, True)) for D in D_grid])
    sigma_b_sph = np.array([radar.radar_xsect(build_drop(D, False)) for D in D_grid])
    v_t = fall(D_grid)

    def first_min(sigma_b: np.ndarray) -> tuple[float, float]:
        mask = (D_grid > 1.3) & (D_grid < 2.2)
        i = int(np.argmin(sigma_b[mask]))
        return D_grid[mask][i], v_t[mask][i]

    D_min_obl, v_min_obl = first_min(sigma_b_obl)
    D_min_sph, v_min_sph = first_min(sigma_b_sph)

    rain64 = build_rain_W(num_points=64)
    rain256 = build_rain_W(num_points=256)

    def run(sc: Scatterer, turb, w: float = 0.0):
        return SpectralIntegrator(
            sc, fall_speed=fall, turbulence=turb, w=w,
            v_min=-1.0, v_max=12.0, n_bins=1024,
            geometry_backscatter=geom_vert_back,
        ).run()

    r_sparse = run(rain64, spectra.NoTurbulence())
    r_dense = run(rain256, spectra.NoTurbulence())
    r_turb = run(rain256, spectra.GaussianTurbulence(0.1))

    r_w = run(rain256, spectra.GaussianTurbulence(0.1), w=1.0)
    band = (r_w.v > v_min_obl) & (r_w.v < v_min_obl + 2.5)
    v_obs = r_w.v[band][int(np.argmin(r_w.sZ_h[band]))]

    print("Kollias, Albrecht, Marks (2002) — W-band Mie spectrum in convective rain")
    print("-" * 72)
    print("First Mie minimum:")
    print(f"  oblate (Thurai 2007): D = {D_min_obl:.3f} mm, v = {v_min_obl:.3f} m/s")
    print(f"  sphere              : D = {D_min_sph:.3f} mm, v = {v_min_sph:.3f} m/s")
    print(f"  shift (oblate − sphere) = {100 * (v_min_obl - v_min_sph):+.1f} cm/s")
    print()
    print("Sampling artifact (NoTurbulence, 1024 velocity bins):")
    print(f"  non-zero bins (num_points=64)  = {int((r_sparse.sZ_h > 0).sum())}")
    print(f"  non-zero bins (num_points=256) = {int((r_dense.sZ_h > 0).sum())}")
    print(f"  non-zero bins (σ_t=0.1 m/s)    = {int((r_turb.sZ_h > 0).sum())}")
    print()
    print("Air-motion retrieval (prescribed w = +1.00 m/s):")
    print(f"  observed 1st Mie min    v = {v_obs:.2f} m/s")
    print(f"  still-air 1st Mie min   v = {v_min_obl:.2f} m/s")
    print(f"  retrieved w             = {v_obs - v_min_obl:+.2f} m/s")


if __name__ == "__main__":
    main()
