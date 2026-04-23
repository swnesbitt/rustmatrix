# rustmatrix

```{image} https://img.shields.io/pypi/v/rustmatrix.svg
:alt: PyPI
:target: https://pypi.org/project/rustmatrix/
```
```{image} https://img.shields.io/crates/v/rustmatrix.svg
:alt: crates.io
:target: https://crates.io/crates/rustmatrix
```
```{image} https://img.shields.io/docsrs/rustmatrix
:alt: docs.rs
:target: https://docs.rs/rustmatrix
```
```{image} https://img.shields.io/badge/docs-rustmatrix.readthedocs.io-blue.svg
:alt: docs
:target: https://rustmatrix.readthedocs.io
```

**Rust-backed T-matrix scattering for nonspherical particles.** A drop-in
replacement for the numerical core of [pytmatrix](https://github.com/jleinonen/pytmatrix)
with substantially faster orientation averaging, parallel PSD
integration, hydrometeor mixtures, and Doppler / polarimetric spectra.

```{code-block} python
from rustmatrix import Scatterer, radar
from rustmatrix.tmatrix_aux import wl_X, K_w_sqr, geom_horiz_back, dsr_thurai_2007
from rustmatrix.refractive import m_w_10C

s = Scatterer(radius=1.0, wavelength=wl_X, m=m_w_10C[wl_X],
              axis_ratio=1.0 / dsr_thurai_2007(2.0), Kw_sqr=K_w_sqr[wl_X])
s.set_geometry(geom_horiz_back)
print(f"Z_h = {radar.refl(s):.3f} mm⁶/m³")
```

## What's here

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {fas}`download` Install
:link: install
:link-type: doc
Pre-built wheels for macOS / Linux / Windows; source build instructions for everything else.
:::

:::{grid-item-card} {fas}`bolt` Quickstart
:link: quickstart
:link-type: doc
Five-minute walkthrough — your first scatterer, PSD, and reflectivity calculation.
:::

:::{grid-item-card} {fas}`book` Tutorials
:link: tutorials/index
:link-type: doc
Fourteen narrated, executable notebooks reproducing published radar-scattering results.
:::

:::{grid-item-card} {fas}`code` API reference
:link: api/index
:link-type: doc
Every public Python symbol, generated from the source docstrings. Rust crate on [docs.rs](https://docs.rs/rustmatrix).
:::

::::

## Why rustmatrix

* **Faster.** ~6× on orientation averaging, ~10× on PSD tabulation, ~430× on `orient_averaged_adaptive` — the Rust kernels release the GIL and parallelise across cores via `rayon`.
* **More physics.** Hydrometeor mixtures (`HydroMix`), Doppler + polarimetric spectra (`SpectralIntegrator`), and pattern × scene integration (`spectra.beam`) for non-uniform scenes — all on top of the same T-matrix solver.
* **Drop-in.** API mirrors pytmatrix wherever the underlying physics matches, so existing scripts port over without code changes.

## About

rustmatrix is developed by [Steve Nesbitt](https://publish.illinois.edu/swnesbitt),
Professor in the Department of
[Climate, Meteorology & Atmospheric Sciences (CLIMAS)](https://climas.illinois.edu/)
at the University of Illinois Urbana-Champaign. It grows out of
instruction material for *ATMS 410 — Radar Meteorology* and accompanies
the textbook *[Radar Meteorology: A First Course](https://onlinelibrary.wiley.com/doi/book/10.1002/9781118432662)*
(Rauber & Nesbitt, 2018, Wiley).

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card}
:link: https://onlinelibrary.wiley.com/doi/book/10.1002/9781118432662
:text-align: center

```{image} _static/book-cover.png
:alt: Radar Meteorology: A First Course
:width: 120px
```

**Radar Meteorology: A First Course**
Rauber & Nesbitt, Wiley (2018)
:::

:::{grid-item-card}
:link: https://climas.illinois.edu/
:text-align: center

```{image} _static/climas-icon.png
:alt: CLIMAS at Illinois
:width: 120px
```

**CLIMAS** — Climate, Meteorology & Atmospheric Sciences at Illinois
:::

::::

See also the companion project
**[myPSD](https://github.com/swnesbitt/myPSD)** — an interactive
web frontend for radar simulation that drives `rustmatrix` under
the hood, for pedagogy and sensitivity exploration.

```{toctree}
:hidden:
:caption: Get started

install
quickstart
```

```{toctree}
:hidden:
:caption: Tutorials

tutorials/index
```

```{toctree}
:hidden:
:caption: Background

background/tmatrix
background/polarimetry
background/psd
background/spectra
background/beam
conventions
```

```{toctree}
:hidden:
:caption: How-to

howto/index
```

```{toctree}
:hidden:
:caption: Reference

api/index
rust-api/index
```

```{toctree}
:hidden:
:caption: About

contributing
changelog
```
