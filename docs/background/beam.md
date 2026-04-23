# Beam pattern × scene integration

The closed-form beam-broadening width
$\sigma_\mathrm{beam} = |\mathbf{u}_h|\,\theta_b / (2\sqrt{2\ln 2})$
{cite}`DoviakZrnic1993` assumes every scatterer inside the beam sees
the same reflectivity and the same wind — a uniform resolution
volume. When the scene is not uniform (deep convection with
sub-beam-scale reflectivity cells, strong shear across the beam, a
scanning antenna over a horizontally-structured storm), the closed
form breaks and you have to integrate the beam pattern against the
scene explicitly. That is what `rustmatrix.spectra.beam` does.

## What goes wrong with closed-form $\sigma_\mathrm{beam}$

The Doviak–Zrnić result assumes

1. a Gaussian beam pattern $G(\theta)$ of angular full-width-half-
   maximum $\theta_b$,
2. a spatially uniform radar reflectivity and PSD inside the beam,
3. a single horizontal wind $\mathbf{u}_h$ across the beam,

and then derives the Doppler-width contribution from the in-beam
distribution of line-of-sight velocities as a closed-form Gaussian.

If any of the three assumptions fails, the measured spectrum is not
$\mathcal{N}(w, \sigma_t^2 + \sigma_\mathrm{beam}^2)$; it's a
$G$-weighted mixture of different reflectivities and velocities. The
[wind × turbulence sensitivity
tutorial](../tutorials/13_wind_turbulence_sensitivity) shows closed-
form $\sigma_\mathrm{beam}$ agreeing with the explicit integration
under uniform conditions; the
[beam × scene tutorial](../tutorials/14_beam_pattern_scene) shows the
case where it fails.

## The integration

For a beam pointing in direction $\hat{\mathbf{n}}$ with angular
pattern $G(\theta)$, and a scene with reflectivity field
$Z(\mathbf{x})$ and wind field $\mathbf{u}(\mathbf{x})$, the received
spectrum at Doppler velocity $v$ is

$$
P(v) = \int G(\theta)^2 \, Z(\mathbf{x}(\theta)) \,
       \mathcal{K}\!\left[v - \hat{\mathbf{n}}\cdot\mathbf{u}(\mathbf{x}(\theta))\right]
       \, d\Omega,
$$

with $\mathcal{K}$ the turbulence + fall-speed kernel at each beam
angle. `rustmatrix.spectra.beam` discretises the beam angles on a
Gauss–Legendre grid, evaluates the scene at each angle, runs the
per-angle spectrum through `SpectralIntegrator`, and sums weighted
by $G^2$.

## Beam patterns

Two patterns are built in:

* **`GaussianBeam(theta_b, ...)`** — the classic Gaussian analytic
  pattern, matched in FWHM to `theta_b`. Sensible default.
* **`AiryBeam(theta_b, ...)`** — the diffraction pattern from a
  circular aperture; has non-zero side-lobes.
* **`TabulatedBeam(theta, gain)`** — wrap a measured antenna pattern
  as a 1-D (angle, gain) table.

## Scene shortcuts

Building a full 3-D reflectivity + wind scene is tedious. For the
common case of *vertically* pointing radars over a 2-D horizontal
reflectivity map, `spectra.beam` ships helpers:

* `marshall_palmer_scene(Z_map, N0=8000)` — turn a dBZ map into a
  per-pixel Marshall–Palmer PSD.
* `convective_cell_scene(...)` — parameterised Gaussian cells for
  sensitivity studies; the NB14 tutorial drives an interactive slider
  over cell spacing.

## When to reach for this

Use `spectra.beam` when **any** of:

* your beam covers horizontally-structured reflectivity (convective
  cells, stratiform-convective transition, the edge of a storm);
* you have strong shear across the beam (upper-level jets, gust
  fronts, scanning geometries with large azimuth steps);
* you need to reproduce a measured antenna side-lobe contamination
  signature.

For uniform stratiform rain with a narrow beam, the closed-form
$\sigma_\mathrm{beam}$ in `SpectralIntegrator` is faster and
sufficient.

## Further reading

* [NB13 — wind × turbulence sensitivity](../tutorials/13_wind_turbulence_sensitivity)
  validates the closed form against explicit integration in its
  regime of validity.
* [NB14 — beam × scene](../tutorials/14_beam_pattern_scene) shows the
  breakdown in convective scenes.
* {cite}`DoviakZrnic1993`, chapter 7 — for the beam-pattern
  mathematics behind the implementation.

```{bibliography}
:filter: docname in docnames
```
