# Quickstart

This page walks through a minimal workflow: build a scatterer, set a
geometry, read out reflectivity. It mirrors `examples/02_raindrop_zdr.py`
at the simplest level.

## A single raindrop at X-band

```python
import numpy as np

from rustmatrix import Scatterer, radar
from rustmatrix.refractive import m_w_10C
from rustmatrix.tmatrix_aux import (K_w_sqr, dsr_thurai_2007,
                                    geom_horiz_back, geom_horiz_forw, wl_X)

# 2 mm equivalent-volume raindrop at 10 °C.
D = 2.0
s = Scatterer(
    radius=D / 2.0,
    wavelength=wl_X,
    m=m_w_10C[wl_X],
    axis_ratio=1.0 / dsr_thurai_2007(D),
    Kw_sqr=K_w_sqr[wl_X],
)

# Backscatter geometry → reflectivity-based observables.
s.set_geometry(geom_horiz_back)
Zh = 10 * np.log10(radar.refl(s, h_pol=True))
Zdr = 10 * np.log10(radar.Zdr(s))

# Forward geometry → propagation observables.
s.set_geometry(geom_horiz_forw)
Kdp = radar.Kdp(s)

print(f"Z_h  = {Zh:7.2f} dBZ per drop/m^3")
print(f"Z_dr = {Zdr:7.3f} dB")
print(f"K_dp = {Kdp:7.4f} deg/km per drop/m^3")
```

## What each import does

* **`Scatterer`** — the T-matrix solver. Give it a particle's size, shape,
  material, and wavelength; it caches the full T-matrix and reuses it
  across geometry changes.
* **`radar.*`** — polarimetric observables (`refl`, `Zdr`, `Kdp`,
  `rho_hv`, `delta_hv`, `Ai`) applied to a configured `Scatterer`.
* **`tmatrix_aux`** — radar-band wavelengths, $|K_w|^2$, canonical
  drop-shape relations, and standard scattering geometries.
* **`refractive`** — tabulated refractive indices for water and ice.

## Next steps

Ready for a real workflow? Read the [tutorials](tutorials/index) in
order — each one is a self-contained narrative with runnable code.
