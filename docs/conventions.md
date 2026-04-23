# Conventions

A one-stop reference for the sign, unit, geometry, and naming
choices `rustmatrix` makes. These are the places where you are most
likely to get off by a factor / a sign / a degree compared with the
paper you are trying to reproduce.

## Units

| Quantity | Units in rustmatrix |
|---|---|
| Length (drop diameter, radius, wavelength) | **mm** |
| Distance (range, path) | **km** (for $K_{dp}$, $A_i$) |
| Velocity (terminal, Doppler, wind) | **m s⁻¹** |
| Reflectivity, linear | **mm⁶ m⁻³** |
| Reflectivity, log | **dBZ** — $10 \log_{10}$ of the linear value |
| Cross sections | **mm²** |
| $K_{dp}$ | **° km⁻¹** |
| $A_h, A_v$ | **dB km⁻¹** |
| Temperature | **K** (except in fall-speed presets taking $T$) |
| Pressure | **Pa** |
| Angles in API | **radians** for geometry tuples, **degrees** for canting σ |

The dBZ convention is the meteorological one (i.e. the linear value
in the log is always mm⁶ m⁻³, not m⁶ m⁻³).

## Polarization naming

* `h` — horizontal linear polarization.
* `v` — vertical linear polarization.
* Scatterer defaults are reported at the horizontal polarization
  (`h_pol=True`) throughout `rustmatrix.radar`.
* LDR is the co-to-cross ratio, $|S_{hv}|^2 / |S_{hh}|^2$, in dB.

## Axis ratios

Drop-shape relations in the literature and in `rustmatrix.tmatrix_aux`
report **$h/v$** (horizontal over vertical) — ≥ 1 for oblate drops.

The `Scatterer(axis_ratio=...)` argument expects the inverse, **$v/h$**,
because that is what the underlying Mishchenko FORTRAN kernel
consumes. Rain snippets therefore always look like

```python
axis_ratio=1.0 / dsr_thurai_2007(D)
```

This is the single most common foot-gun in porting a `pytmatrix`
script. The convention matches `pytmatrix` for drop-in compatibility.

## Geometries

Pre-built geometry tuples live in `rustmatrix.tmatrix_aux`:

| Name | Use |
|---|---|
| `geom_horiz_back` | horizontally-pointing antenna, 180° back-scatter — drives $Z_h, Z_{dr}, \rho_{hv}, \delta_{hv}$ |
| `geom_horiz_forw` | horizontally-pointing antenna, 0° forward-scatter — drives $K_{dp}, A_i$ |
| `geom_vert_back`  | vertically-pointing antenna, back-scatter — profiler / cloud-radar spectra |

Call `s.set_geometry(...)` to switch. The underlying T-matrix is
cached on the `Scatterer`, so the switch is O(1).

## Doppler sign

Positive Doppler velocity = fall direction = toward a *down*-pointing
radar. This is the convention used in
{cite}`Kollias2002,Zhu2023,BillaultRoux2023`. For an up-looking
profiler, flip the sign of `w` and of `v_bins` passed to
`SpectralIntegrator` — the kernels are even in $v$ so this is a pure
sign flip, not a re-compute.

## Canting angles

* Mean canting angle is measured from the vertical (so a horizontally-
  aligned oblate drop has mean canting 0°).
* `canting_std` is in **degrees**, applied as a zero-mean Gaussian PDF
  around the mean.
* Full random orientation is reached in the limit of large
  `canting_std`; `rustmatrix` also provides closed-form random-
  orientation averaging if that is what you want.

## Refractive index

`rustmatrix.refractive` tables are indexed by wavelength in mm:

```python
from rustmatrix.refractive import m_w_0C, m_w_10C, m_w_20C, m_i
from rustmatrix.tmatrix_aux import wl_C
m_rain = m_w_10C[wl_C]
```

`m_w_*C` are liquid water at 0 / 10 / 20 °C; `m_i` is solid ice.
Refractive index follows the physics convention $m = n + ik$, $k > 0$
for an absorbing medium.

## Wavelength constants

`tmatrix_aux.wl_S, wl_C, wl_X, wl_Ku, wl_Ka, wl_W` — the standard
radar-band wavelengths in mm. Use them as dictionary keys into the
refractive-index and $|K_w|^2$ tables:

```python
from rustmatrix.tmatrix_aux import wl_W, K_w_sqr
kw = K_w_sqr[wl_W]
```

## Scatterer caching

The `Scatterer` caches:

1. the T-matrix itself (invalidated by changes to size, shape,
   refractive index, or wavelength),
2. the amplitude/phase matrices at the current geometry.

This means iterating over PSDs or canting parameters is cheap once
the T-matrix is computed. It also means **mutating `s.m`, `s.radius`,
or `s.wavelength` after the first call silently requires a
re-compute** — `rustmatrix` handles the invalidation for you, but if
you are benchmarking, change one parameter, re-set geometry, call
once to warm the cache.

## Drop-in compatibility with pytmatrix

The `Scatterer`, `radar.*`, and `psd.*` public APIs are intentionally
1:1 with `pytmatrix` where the physics matches. A script written for
`pytmatrix` typically ports to `rustmatrix` by changing only the
import. Places where we deliberately diverge:

* `HydroMix` and `SpectralIntegrator` — new capabilities, no
  `pytmatrix` analogue.
* Full-random-orientation averaging is exact in `rustmatrix`
  (closed form); `pytmatrix` uses numerical quadrature. Both agree to
  floating-point precision on the parity test suite.
* Some `pytmatrix` private attributes are not preserved — only the
  documented public surface is stable across the port.

## References

```{bibliography}
:filter: docname in docnames
```
