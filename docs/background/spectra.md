# Doppler and polarimetric spectra

The bulk polarimetric observables in `rustmatrix.radar` are
velocity-integrated. The modules under `rustmatrix.spectra` keep the
velocity dimension — they return the scattering matrix as a function
of Doppler velocity so you can look at where in the spectrum a given
hydrometeor type lives. This is the regime of cloud-radar retrievals
{cite}`Kollias2002,BillaultRoux2023,Zhu2023,Lakshmi2024`.

## Line-of-sight Doppler model

For a vertically-pointing radar, the observed line-of-sight velocity
of a drop of diameter $D$ is

$$
v(D) = v_t(D) + w + \epsilon_t(D) + \epsilon_\mathrm{beam}(\mathbf{u}_h),
$$

where

* $v_t(D)$ is terminal fall speed,
* $w$ is the mean vertical air motion,
* $\epsilon_t$ is a diameter-dependent turbulence broadening, and
* $\epsilon_\mathrm{beam}$ is the deterministic beam-broadening from
  horizontal wind $|\mathbf{u}_h|$ through a beam of full-width
  $\theta_b$.

**Sign convention**: positive velocity = toward a *down*-pointing
radar = fall direction. Flip the sign of `v_bins` and `w` for
up-looking geometries.

## Fall-speed presets

`rustmatrix.spectra` ships several $v_t(D)$ relations so the
tutorials can reproduce published results exactly:

| Function | Regime | Notes |
|---|---|---|
| `atlas_srivastava_sekhon_1973` | rain | simple exponential fit; one-liner |
| `brandes_et_al_2002` | rain | 4th-order polynomial, 0.1–8 mm |
| `beard_1976(T, P)` | rain, arbitrary $(T, P)$ | Brandes + {cite}`Beard1977` density correction |
| `locatelli_hobbs_1974_*` | snow / graupel | power laws by habit |

Drop in your own $v_t(D)$ as any callable — pass a function returning
fall speed in m/s for diameter in mm.

## Turbulence broadening

Subgrid turbulence smears each drop's Doppler return by a Gaussian of
width $\sigma_t(D)$. Two kernels are wired up:

* **Single-Gaussian** — one $\sigma_t$ applied to every diameter;
  cheap and conventional.
* **Particle-inertia (Zhu 2023)** — solves a drag ODE per drop, so
  heavier drops see narrower broadening than lighter drops
  {cite}`Zhu2023`. The [tutorial](../tutorials/09_zhu_2023_particle_inertia)
  reproduces their Fig. 4 showing that conventional broadening
  over-smooths the spectrum.

## Beam broadening

For a vertically-pointing radar in horizontal wind $\mathbf{u}_h$
with Gaussian beam full-width-half-maximum $\theta_b$,

$$
\sigma_\mathrm{beam} = \frac{|\mathbf{u}_h|\,\theta_b}{2\sqrt{2 \ln 2}}
$$

{cite}`DoviakZrnic1993`, §5.3, in the small-$\theta_b$ limit. The
convolution of $\sigma_\mathrm{beam}$ with $\sigma_t(D)$ in quadrature
is exact when the beam is narrow enough that every scatterer inside
it sees the same mean wind. When that assumption breaks (deep
convection, scan geometries, strong shear across the beam), use
`rustmatrix.spectra.beam` for explicit pattern × scene integration —
see [the beam-integration page](beam.md).

## Noise

Real receivers have thermal noise. Pass `noise=` to
`SpectralIntegrator`:

* `None` — signal only (default).
* `"realistic"` — `realistic_noise_floor(wavelength)` picks a band-
  appropriate floor.
* `float` — total noise reflectivity in mm⁶ m⁻³, spread uniformly
  across velocity bins.
* `(noise_h, noise_v)` — separate per-channel noise.

Noise is uncorrelated between H and V, so it adds incoherently to
`sZ_h` and `sZ_v`; signal-only scattering matrices are preserved so
that bulk integration still round-trips to the `radar.*` answers.

## Spectral polarimetric observables

Given `S_spec(v)` and `Z_spec(v)` on a velocity grid, the per-bin
analogues of the bulk observables are

$$
\begin{aligned}
sZ_h(v)      &= (\lambda^4 / \pi^5 |K_w|^2) \, |S_{hh}^b(v)|^2 \, \Delta v^{-1}, \\
sZ_{dr}(v)   &= 10 \log_{10} (sZ_h / sZ_v), \\
sK_{dp}(v)   &\propto \Re [S_{vv}^f(v) - S_{hh}^f(v)], \\
s\rho_{hv}(v) &= \text{normalised cross-product per bin},
\end{aligned}
$$

each evaluated bin-by-bin by applying the bulk `radar.*` formulas to
the per-bin scattering matrix. See the
[spectral polarimetry tutorial](../tutorials/08_spectral_polarimetry_rain_ice)
for the Billault-Roux et al. snowfall case study and
[tutorial 12](../tutorials/12_spectral_polarimetry_rain_slw_hail) for
Lakshmi et al. hail.

## HydroMix spectra

`S_spec` and `Z_spec` are linear in $N(D)$, so a
`rustmatrix.hd_mix.HydroMix` of multiple species is handled by
summing per-component spectra on a shared velocity grid — with
different fall-speed and turbulence kernels per species. The
[SLW-vs-snow tutorial](../tutorials/10_slw_vs_snow) exercises this
end-to-end.

## Further reading

* {cite}`Kollias2002` — the foundational paper on why Mie-minimum
  structure in W-band spectra carries useful retrieval information.
* {cite}`BillaultRoux2023,Zhu2023,Lakshmi2024` — modern cloud-radar
  retrievals reproduced in the tutorials.
* {cite}`DoviakZrnic1993`, chapter 5 — the Doppler-spectrum
  derivations.

```{bibliography}
:filter: docname in docnames
```
