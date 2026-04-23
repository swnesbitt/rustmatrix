"""Tutorial 14 — Beam-pattern + scene integration for a down-looking radar.

Physics question
----------------
A radar does not measure the scene at its boresight pixel; it measures
a pattern-weighted integral over its solid angle. When the scene is
homogeneous across the beam footprint, the closed-form σ_beam formula
used in Tutorial 13 captures everything. When the scene contains
structure finer than (or comparable to) the beam footprint — convective
cells, updraft/downdraft couplets, horizontal reflectivity gradients —
the beam pattern has to be integrated explicitly.

Two properties of the pattern matter independently:

* **Main-lobe width**. A 1° beam projected from 15 km range has a
  ~260 m footprint; a 3° beam has ~780 m. Features narrower than the
  footprint are smeared and moment retrievals get biased toward the
  beam-averaged reflectivity, velocity, and shear.
* **Sidelobes**. A parabolic dish with uniform aperture illumination
  has an Airy-pattern first sidelobe at **−17.6 dB** relative to the
  main-lobe peak. If a distant bright cell sits inside that sidelobe
  and the main lobe is pointed at a quiet patch, the sidelobe
  contribution can dominate the Doppler moments.

This tutorial builds a synthetic down-looking W-band radar at 20 km
altitude scanning across a rain scene with 20 dBZ background and
500-m-wide 45 dBZ cells. Three co-located vertical-motion patterns
explore how beam-averaging interacts with velocity structure:

* **uniform_updraft** — every cell is a 3 m/s updraft.
* **alternating**    — adjacent cells alternate −3 / +3 m/s.
* **dipole_couplet** — each cell is an updraft/downdraft dipole
  straddling the enhanced-Z peak.

Each scene is sampled with 1° and 3° beams, in both Gaussian and
Airy patterns. Tabulated and plotted: the scanned Doppler velocity
(first moment) and spectral width (second moment), for each
(beam-width, pattern, scene) combination.

Scene assumption
----------------
Horizontal wind is zero everywhere (we isolate the *scene-structure*
contribution to beam broadening — the *uniform-wind* contribution was
the subject of Tutorial 13). Vertical motion ``w`` is co-located with
the cell reflectivity maxima, as the user preference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rustmatrix import Scatterer
from rustmatrix.psd import PSDIntegrator
from rustmatrix.refractive import m_w_10C
from rustmatrix.spectra import brandes_et_al_2002
from rustmatrix.spectra.beam import (AiryBeam, BeamIntegrator, GaussianBeam,
                                      Scene, marshall_palmer_psd_factory)
from rustmatrix.tmatrix_aux import (K_w_sqr, dsr_thurai_2007,
                                     geom_vert_back, wl_W)


# --- geometry constants -----------------------------------------------------

RADAR_ALTITUDE_M = 20000.0
TARGET_ALTITUDE_M = 3000.0          # middle of rain layer (z ≤ 5 km)
RANGE_M = RADAR_ALTITUDE_M - TARGET_ALTITUDE_M   # 17 km slant from zenith
RAIN_TOP_M = 5000.0

BG_DBZ = 20.0
CELL_PEAK_DBZ = 45.0
CELL_WIDTH_M = 500.0                 # 500 m Gaussian FWHM-ish
CELL_SIGMA_M = CELL_WIDTH_M / (2.0 * np.sqrt(2.0 * np.log(2.0)))
CELL_SPACING_M = 1500.0              # spacing between cell centres
CELL_CENTERS_X = np.arange(-3000.0, 3001.0, CELL_SPACING_M)
W_PEAK = 3.0                         # [m/s]  ±sign per scene

SCAN_X = np.linspace(-4000.0, 4000.0, 81)   # radar-position scan
BEAMWIDTHS_DEG = (1.0, 3.0)

V_MIN, V_MAX, N_BINS = -5.0, 15.0, 384


# --- scene builder ----------------------------------------------------------

def _cell_signs(pattern: str, n_cells: int) -> np.ndarray:
    if pattern == "uniform_updraft":
        return -np.ones(n_cells)            # negative = upward
    if pattern == "alternating":
        return np.array([-1.0 if i % 2 == 0 else 1.0 for i in range(n_cells)])
    if pattern == "dipole_couplet":
        return np.zeros(n_cells)            # handled below with dipole offset
    raise ValueError(f"unknown pattern: {pattern}")


def build_scene(pattern: str) -> Scene:
    """Build a Scene instance for one vertical-motion pattern."""
    centers = CELL_CENTERS_X
    sigma = CELL_SIGMA_M
    z_top = RAIN_TOP_M
    Z_bg_lin = 10.0 ** (BG_DBZ / 10.0)
    Z_peak_excess_lin = 10.0 ** (CELL_PEAK_DBZ / 10.0) - Z_bg_lin

    def Z_dBZ(x, y, z):
        mask = (z >= 0) & (z <= z_top)
        Z_lin = np.where(mask, Z_bg_lin, 1e-10)
        for xc in centers:
            bump = Z_peak_excess_lin * np.exp(-0.5 * ((x - xc) / sigma) ** 2)
            Z_lin = Z_lin + bump * mask
        return 10.0 * np.log10(np.maximum(Z_lin, 1e-10))

    signs = _cell_signs(pattern, len(centers))

    if pattern == "dipole_couplet":
        # Down on -x side, up on +x side of each cell centre. Encode
        # as an anti-symmetric profile crossing through zero at xc.
        def w_fn(x, y, z):
            mask = (z >= 0) & (z <= z_top)
            w_total = np.zeros_like(x)
            for xc in centers:
                # odd-symmetric bump: (x - xc) * exp(-(x-xc)²/2σ²) / (σ·√e)
                # peak magnitude ≈ W_PEAK at |x-xc| = σ
                arg = (x - xc) / sigma
                w_total = w_total + W_PEAK * arg * np.exp(0.5 - 0.5 * arg ** 2)
            return w_total * mask
    else:
        def w_fn(x, y, z):
            mask = (z >= 0) & (z <= z_top)
            w_total = np.zeros_like(x)
            for i, xc in enumerate(centers):
                w_total = w_total + signs[i] * W_PEAK * np.exp(
                    -0.5 * ((x - xc) / sigma) ** 2
                )
            return w_total * mask

    def u_h_fn(x, y, z):
        return np.zeros_like(x)

    return Scene(Z_dBZ=Z_dBZ, w=w_fn, u_h=u_h_fn, u_h_azimuth=0.0)


# --- scatterer --------------------------------------------------------------

def build_rain_W() -> Scatterer:
    s = Scatterer(wavelength=wl_W, m=m_w_10C[wl_W],
                  Kw_sqr=K_w_sqr[wl_W], ddelt=1e-4, ndgs=2)
    integ = PSDIntegrator()
    integ.D_max = 5.0
    integ.num_points = 48
    integ.axis_ratio_func = lambda D: 1.0 / dsr_thurai_2007(D)
    integ.geometries = (geom_vert_back,)
    s.psd_integrator = integ
    s.psd_integrator.init_scatter_table(s)
    return s


# --- moments -----------------------------------------------------------------

def moments(v: np.ndarray, sZh: np.ndarray) -> tuple[float, float, float]:
    """Return (reflectivity_dBZ, mean_velocity, spectral_width)."""
    sZh_c = np.clip(sZh, 0.0, None)
    P = sZh_c.sum()
    if P <= 0:
        return float("-inf"), float("nan"), float("nan")
    dv = np.mean(np.diff(v))
    Z_lin = P * dv
    Z_dBZ = 10.0 * np.log10(max(Z_lin, 1e-10))
    mu = float((v * sZh_c).sum() / P)
    var = float(((v - mu) ** 2 * sZh_c).sum() / P)
    return Z_dBZ, mu, float(np.sqrt(max(var, 0.0)))


# --- sweep ------------------------------------------------------------------

@dataclass
class SweepResult:
    pattern_name: str            # 'uniform_updraft', 'alternating', ...
    beam_kind: str               # 'gaussian' | 'airy'
    hpbw_deg: float
    x_scan: np.ndarray
    Z_dBZ: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray


def sweep_radar(
    scatterer: Scatterer,
    scene: Scene,
    psd_factory,
    beam,
    x_scan: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Z_dBZ_arr = np.empty_like(x_scan)
    mu_arr = np.empty_like(x_scan)
    sig_arr = np.empty_like(x_scan)
    for i, xr in enumerate(x_scan):
        bi = BeamIntegrator(
            scatterer=scatterer,
            beam=beam,
            scene=scene,
            psd_factory=psd_factory,
            fall_speed=brandes_et_al_2002,
            radar_position=(xr, 0.0, RADAR_ALTITUDE_M),
            boresight=(0.0, 0.0, -1.0),
            range_m=RANGE_M,
            v_min=V_MIN, v_max=V_MAX, n_bins=N_BINS,
            n_theta=16, n_phi=16,
        )
        r = bi.run()
        Z_dBZ_arr[i], mu_arr[i], sig_arr[i] = moments(r.v, r.sZ_h)
    return Z_dBZ_arr, mu_arr, sig_arr


# --- main -------------------------------------------------------------------

def main() -> list[SweepResult]:
    rain = build_rain_W()
    psd_factory = marshall_palmer_psd_factory(N0=8000.0, D_max=5.0)

    patterns = ("uniform_updraft", "alternating", "dipole_couplet")
    beams = []
    for hpbw_deg in BEAMWIDTHS_DEG:
        hpbw = np.deg2rad(hpbw_deg)
        beams.append(("gaussian", hpbw_deg, GaussianBeam(hpbw=hpbw)))
        beams.append(("airy",     hpbw_deg, AiryBeam(hpbw=hpbw)))

    results: list[SweepResult] = []
    print("Tutorial 14 — beam × scene integration for a W-band down-looking radar")
    print("-" * 72)
    print(f"Radar at {RADAR_ALTITUDE_M/1000:.1f} km, rain top "
          f"{RAIN_TOP_M/1000:.1f} km, target {TARGET_ALTITUDE_M/1000:.1f} km")
    print(f"Range to gate: {RANGE_M/1000:.1f} km. Cell spacing "
          f"{CELL_SPACING_M:.0f} m, width {CELL_WIDTH_M:.0f} m")
    print()

    for pattern in patterns:
        scene = build_scene(pattern)
        for kind, hpbw_deg, beam in beams:
            Zs, mus, sigs = sweep_radar(rain, scene, psd_factory, beam, SCAN_X)
            results.append(SweepResult(
                pattern_name=pattern,
                beam_kind=kind,
                hpbw_deg=hpbw_deg,
                x_scan=SCAN_X.copy(),
                Z_dBZ=Zs, mu=mus, sigma=sigs,
            ))
        print(f"  {pattern:18s}  [done — 4 beam configs, "
              f"{len(SCAN_X)} scan positions each]")

    print()
    # One-line moment summary at x=0 (directly on a cell centre).
    print("Summary — moments at radar x = 0 (cell-centered beam)")
    print("-" * 72)
    header = (f"  {'pattern':18s}  {'beam':9s}  {'θ_b':>5}  "
              f"{'Z':>7}  {'μ':>7}  {'σ':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for rr in results:
        i0 = int(np.argmin(np.abs(rr.x_scan)))
        print(f"  {rr.pattern_name:18s}  {rr.beam_kind:9s}  "
              f"{rr.hpbw_deg:>4.1f}°  {rr.Z_dBZ[i0]:6.2f}  "
              f"{rr.mu[i0]:6.3f}  {rr.sigma[i0]:6.3f}")

    return results


if __name__ == "__main__":
    main()
