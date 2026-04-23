# Polarimetric observables

Dual-polarization weather radars transmit and receive orthogonal
linear polarizations (conventionally horizontal `h` and vertical `v`).
A handful of combinations of the resulting amplitudes and phases
form the standard polarimetric observables implemented in
`rustmatrix.radar`. Definitions, sign conventions, and units below
follow {cite}`BringiChandrasekar2001` and {cite}`DoviakZrnic1993`.

## The scattering matrix

For a single particle at a fixed geometry, the far-field scattered
electric field is related to the incident field by the $2 \times 2$
amplitude scattering matrix $\mathbf{S}$:

$$
\begin{pmatrix} E_h^s \\ E_v^s \end{pmatrix}
= \frac{e^{ikr}}{r}
\begin{pmatrix} S_{hh} & S_{hv} \\ S_{vh} & S_{vv} \end{pmatrix}
\begin{pmatrix} E_h^i \\ E_v^i \end{pmatrix}.
$$

The $S_{ij}$ are complex — they carry both amplitude and phase.
`rustmatrix` returns them from the T-matrix solver at any specified
incident / scattered geometry via
`rustmatrix.scatter.amplitude_matrix`.

Polarimetric observables come in two flavours:

* **Back-scatter quantities** ($Z_h$, $Z_{dr}$, $\rho_{hv}$, $\delta_{hv}$,
  LDR) use $\mathbf{S}$ at the 180° back-scatter geometry and
  integrate $|S|^2$-type quantities over the PSD.
* **Forward-scatter quantities** ($K_{dp}$, $A_h$, $A_v$) use
  $\mathbf{S}$ at the 0° forward-scatter geometry and integrate
  $\Re$ / $\Im$ linear in $S$.

Switch geometries with `s.set_geometry(geom_horiz_back)` /
`s.set_geometry(geom_horiz_forw)`. The T-matrix itself is cached on
the `Scatterer`, so the switch is cheap.

## Back-scatter observables

### Reflectivity factor $Z_h$

$$
Z_h = \frac{\lambda^4}{\pi^5 |K_w|^2}
      \int |S_{hh}^{b}(D)|^2 \, N(D)\, dD
$$

Units: mm⁶ m⁻³ (linear); dBZ after $10 \log_{10}$. The dielectric
factor $|K_w|^2$ for water at the radar band is tabulated in
`rustmatrix.tmatrix_aux.K_w_sqr`. Use `radar.refl(s, h_pol=True)`.

### Differential reflectivity $Z_{dr}$

$$
Z_{dr} = 10 \log_{10} \frac{Z_h}{Z_v}.
$$

Positive $Z_{dr}$ means oblate scatterers aligned with their long
axis horizontal (the equilibrium-drop configuration). Rain at C-band
gives $Z_{dr} \approx 0.3$–3 dB; pristine ice columns give slightly
positive values; randomly-tumbling aggregates give ~0 dB.

### Co-polar correlation $\rho_{hv}$

$$
\rho_{hv} = \frac{
  \left| \int S_{vv}^b S_{hh}^{b*} \, N(D)\, dD \right|
}{
  \sqrt{\int |S_{hh}^b|^2 \, N\, dD \cdot \int |S_{vv}^b|^2 \, N\, dD}
}.
$$

Bounded by $[0, 1]$. Values near 1 mean a uniform population; drops
below 0.97 indicate mixed-phase or irregular scatterers. A key
discriminator between meteorological and non-meteorological echo.

### Differential backscatter phase $\delta_{hv}$

$$
\delta_{hv} = \arg \int S_{hh}^b S_{vv}^{b*} \, N(D)\, dD.
$$

Non-zero only when scatterers are large enough that the Rayleigh
approximation fails — so a resonance fingerprint at C-band for
$D \gtrsim 5$ mm. The [HydroMix tutorial](../tutorials/06_hd_mix)
shows it for mixed rain + ice.

### Linear depolarization ratio LDR

$$
\mathrm{LDR} = 10 \log_{10}
  \frac{\int |S_{hv}^b|^2 \, N\, dD}{\int |S_{hh}^b|^2 \, N\, dD}.
$$

Requires non-zero cross-polar response, i.e. non-trivial canting.
LDR is small (< −25 dB) for rain and rises sharply in the melting
layer and for oriented ice {cite}`Ryzhkov2005,Kumjian2013`.

## Forward-scatter observables

### Specific differential phase $K_{dp}$

$$
K_{dp} = \frac{180}{\pi}\, \lambda \,
         \Re \!\int \! \left[ S_{vv}^f(D) - S_{hh}^f(D) \right]\, N(D)\, dD.
$$

Units: ° km⁻¹. Positive for horizontally-oriented oblate particles
(rain), near-zero for spheres and tumbling ice, slightly negative for
vertically-oriented crystals. $K_{dp}$ is immune to attenuation and
calibration bias — the workhorse for rain-rate retrieval.

### Specific attenuation $A_h$, $A_v$

$$
A_i = 8.686 \cdot 10^{-3} \, \lambda \, \Im \! \int \! S_{ii}^f(D) \,
      N(D)\, dD.
$$

Units: dB km⁻¹. Rises sharply at the higher radar bands (Ka, W) —
see the [radar-band sweep tutorial](../tutorials/05_radar_band_sweep).

## Sign and geometry conventions

`rustmatrix` follows Bringi & Chandrasekar {cite}`BringiChandrasekar2001`:

* **Horizontal polarization** is `h`, vertical is `v`.
* **Equilibrium drop axis ratio** is reported as $h/v$ ≥ 1; the
  `Scatterer(axis_ratio=...)` argument expects $v/h$ (the value the
  Mishchenko code wants), so rain scripts use
  `axis_ratio = 1.0 / dsr_thurai_2007(D)`.
* **Back-scatter geometry** is `geom_horiz_back` from
  `rustmatrix.tmatrix_aux` (the BSA convention — incident antenna
  frame). **Forward-scatter** is `geom_horiz_forw`.
* **Doppler velocity** is positive *downward* (toward the radar for
  a vertically-pointing profiler) throughout `rustmatrix.spectra` —
  matching the convention in {cite}`Kollias2002,Zhu2023`.

## Relationship to the scattering cross-sections

Back-scatter cross-section $\sigma_b$ and total scattering
$\sigma_{sca}$ are also available via
`rustmatrix.scatter.radar_xsect` and `scatter.sca_xsect` — useful for
Mie parity checks and for feeding `spectra.SpectralIntegrator` a
custom $\sigma_b(D)$.

## Further reading

* {cite}`BringiChandrasekar2001`, chapters 3–4 — the definitive
  reference.
* {cite}`Kumjian2013` — a three-part practical tutorial aimed at
  operational meteorologists.
* {cite}`DoviakZrnic1993` — for the radar-equation derivations and
  the sign conventions used in the Doppler / spectral modules.

```{bibliography}
:filter: docname in docnames
```
