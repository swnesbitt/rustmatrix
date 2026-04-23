# Particle size distributions

A radar observable is an integral over the number concentration of
scatterers per unit volume per unit diameter, $N(D)$. `rustmatrix.psd`
provides the standard analytical forms plus a generic binned
distribution, all compatible with `PSDIntegrator`'s cached
tabulation.

## Exponential (Marshall–Palmer)

{cite}`MarshallPalmer1948` observed that stratiform rain drop-size
distributions are well approximated by

$$
N(D) = N_0 \exp(-\Lambda D),
$$

with $N_0 \approx 8000$ mm⁻¹ m⁻³ and $\Lambda$ set by the rain rate.
The two-parameter exponential is the simplest non-trivial PSD and is
wired up as `psd.ExponentialPSD(N0=..., Lambda=...)`.

Use it when you want a tractable stress test — it has no free shape
parameter, so convergence and band dependence are easy to isolate.
The [radar-band sweep tutorial](../tutorials/05_radar_band_sweep)
uses it for exactly that reason.

## Gamma (Ulbrich)

Real rain spectra show a roll-off at small diameters that the
exponential cannot capture. {cite}`Ulbrich1983` proposed

$$
N(D) = N_0 D^\mu \exp(-\Lambda D),
$$

adding a shape parameter $\mu$ — positive for convective rain,
negative for drizzle-dominated distributions. Implemented as
`psd.UnnormalizedGammaPSD(N0, mu, Lambda)`.

The three Ulbrich parameters are strongly correlated across rain
events, so `rustmatrix` also provides the *normalised* form below
which disentangles concentration from shape.

## Normalised gamma (Testud / Bringi–Chandrasekar)

The normalised-gamma PSD {cite}`Testud2001,BringiChandrasekar2001` is
parameterised by quantities that are roughly independent across rain
events:

* $D_0$ — median volume diameter (mm),
* $N_w$ — normalised intercept, constant for a Marshall–Palmer
  distribution,
* $\mu$ — dimensionless shape parameter.

$$
N(D) = N_w \, f(\mu) \, \left(\frac{D}{D_0}\right)^\mu
       \exp\!\left[-(3.67 + \mu)\,\frac{D}{D_0}\right],
$$

where $f(\mu)$ is a normalisation that keeps $N_w$ fixed across
$\mu$. This is the form used in the
[gamma-PSD rain tutorial](../tutorials/03_psd_gamma_rain) and
throughout the operational-radar literature. Constructed as
`psd.GammaPSD(D0=..., Nw=..., mu=...)`.

## Binned / empirical

For disdrometer observations, reanalysis output, or any case where
you have $N$ in arbitrary diameter bins, use
`psd.BinnedPSD(bin_edges, N)` — it interpolates linearly within bins
and evaluates to zero outside. The standard `PSDIntegrator` machinery
works unchanged.

## Numerical integration

`psd.PSDIntegrator`:

1. On `init_scatter_table(s)`, evaluates $\mathbf{S}(D)$ and
   $\mathbf{Z}(D)$ at `num_points` diameters between $D_\min$ and
   $D_\max$. This is the parallel Rust kernel — the only expensive
   step.
2. For each PSD assigned to `s.psd`, integrates the cached tables
   against $N(D)$ by trapezoidal rule.
3. Any number of different PSD shapes can be evaluated from the same
   cached table — swap `s.psd = psd.GammaPSD(...)` and re-read the
   observables at a cost near zero.

Tips:

* **`D_max`**: set a few times the expected largest drop. Too small
  truncates the tail and under-estimates $Z_h$ at C-band and below;
  too large wastes quadrature points on near-zero contributions.
* **`num_points`**: 64 is almost always enough for rain; oriented ice
  at the Mie-resonance bands may want 128.
* **`geometries`**: pass both back-scatter and forward-scatter in the
  same tuple so the table is built once for all observables you need.

## Further reading

* {cite}`BringiChandrasekar2001`, chapter 7 — the canonical treatment
  of PSD retrieval from polarimetric radar.
* {cite}`Testud2001` — the original normalised-gamma paper.
* [The gamma-PSD tutorial](../tutorials/03_psd_gamma_rain) — shows
  $Z_h, Z_{dr}, K_{dp}, A_i$ as functions of rain rate over an
  ensemble of $D_0, N_w, \mu$.

```{bibliography}
:filter: docname in docnames
```
