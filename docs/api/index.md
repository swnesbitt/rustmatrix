# Python API reference

Every public symbol of the `rustmatrix` Python package, rendered from
source docstrings. The module pages below are the authoritative
reference; the [tutorials](../tutorials/index) and
[background pages](../background/tmatrix) show them in use.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Scatterer
:link: scatterer
:link-type: doc

The T-matrix solver. `Scatterer`, `TMatrix` alias, size / shape /
refractive-index constants.
:::

:::{grid-item-card} PSD
:link: psd
:link-type: doc

`PSDIntegrator`, `ExponentialPSD`, `GammaPSD`,
`UnnormalizedGammaPSD`, `BinnedPSD`.
:::

:::{grid-item-card} Radar observables
:link: radar
:link-type: doc

`refl`, `Zdr`, `Kdp`, `rho_hv`, `delta_hv`, `Ai`, `ldr`.
:::

:::{grid-item-card} Scatter helpers
:link: scatter
:link-type: doc

Cross sections, amplitude / phase matrices, asymmetry parameter.
:::

:::{grid-item-card} HydroMix
:link: hd_mix
:link-type: doc

Combine multiple species into one Scatterer-shaped object.
:::

:::{grid-item-card} Spectra
:link: spectra
:link-type: doc

`SpectralIntegrator`, fall-speed presets, turbulence kernels.
:::

:::{grid-item-card} Beam × scene
:link: spectra.beam
:link-type: doc

Explicit beam-pattern × scene integration for non-uniform beams.
:::

:::{grid-item-card} Refractive index
:link: refractive
:link-type: doc

Tabulated water / ice indices; Maxwell-Garnett / Bruggeman mixing.
:::

:::{grid-item-card} tmatrix_aux
:link: tmatrix_aux
:link-type: doc

Radar-band wavelengths, $|K_w|^2$, drop-shape relations, geometries.
:::

:::{grid-item-card} Orientation
:link: orientation
:link-type: doc

Orientation-averaging strategies and canting-angle PDFs.
:::

:::{grid-item-card} Quadrature
:link: quadrature
:link-type: doc

Gautschi quadrature against arbitrary weights.
:::

::::

The Rust crate that backs the Python extension is documented
separately on [docs.rs](https://docs.rs/rustmatrix); see
[](../rust-api/index).

```{toctree}
:hidden:
:maxdepth: 1

scatterer
psd
radar
scatter
hd_mix
spectra
spectra.beam
refractive
tmatrix_aux
orientation
quadrature
```
