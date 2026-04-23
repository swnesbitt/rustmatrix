# Tutorials

The `examples/` directory contains fourteen numbered tutorials. Each
ships both as a runnable `.py` script and as a matching `.ipynb` that
executes on the docs build so you see the figures and printed output
inline.

| # | Notebook | What it covers |
|---|---|---|
| 01 | [Sphere / Mie parity](01_sphere_mie) | Sanity check — T-matrix at spherical shape reduces to Mie. |
| 02 | Raindrop $Z_\mathrm{dr}$ | Single 2 mm oblate raindrop at C-band. |
| 03 | Gamma-PSD rain | Tabulated S, Z → $Z_h$, $Z_\mathrm{dr}$, $K_\mathrm{dp}$, $A_i$ vs rain rate. |
| 04 | Oriented ice | Columnar ice at W-band with a Gaussian canting PDF. |
| 05 | Radar band sweep | Same particle across S/C/X/Ku/Ka/W. |
| 06 | HydroMix | Rain + oriented ice as one combined scatterer. |
| 07 | Doppler spectrum, rain | Reproduces Kollias et al. 2002 W-band Mie minimum. |
| 08 | Spectral polarimetry, rain + ice | Billault-Roux et al. 2023 dual-frequency snow signature. |
| 09 | Zhu 2023 particle inertia | Diameter-dependent turbulence broadening. |
| 10 | SLW vs snow | Bimodal W-band spectrum from a HydroMix. |
| 11 | Honeyager 2013 classes | Rain / aggregate / graupel / dense-ice σ_b and DWR. |
| 12 | Rain + SLW + hail (Lakshmi 2024) | C-band spectral polarimetry at 500 hPa. |
| 13 | Wind × turbulence sensitivity | Closed-form σ_beam validation sweep. |
| 14 | Beam pattern × scene | Gaussian / Airy patterns over convective cells; interactive cell-spacing slider. |

The notebooks build on each other loosely — later tutorials assume you've
seen earlier ones — but any of them run standalone once `rustmatrix` is
installed.

```{toctree}
:hidden:
:maxdepth: 1

01_sphere_mie
```
