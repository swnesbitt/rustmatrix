# The T-matrix method

The **transition matrix** (or T-matrix) method is the standard tool for
computing electromagnetic scattering by nonspherical particles whose
shape is a rotationally-symmetric surface of revolution — the regime
that covers essentially every meteorological hydrometeor except the
fully tumbling snowflake. It was introduced by Waterman
{cite}`Waterman1971` under the name *extended boundary condition
method*, and reached its modern, numerically-robust formulation in
Mishchenko's FORTRAN implementation {cite}`MishchenkoTravis1998` —
the reference code whose orientation-averaging kernel `rustmatrix`
re-implements in Rust.

## What the T-matrix is

For a single scatterer illuminated by an incident field expanded in
vector spherical harmonics with coefficients $\{a_{mn}, b_{mn}\}$, the
scattered field has expansion coefficients $\{p_{mn}, q_{mn}\}$ given
by the linear map

$$
\begin{pmatrix} p \\ q \end{pmatrix} =
\mathbf{T}\begin{pmatrix} a \\ b \end{pmatrix}.
$$

The matrix $\mathbf{T}$ is a property of the particle alone — its
shape, size, refractive index, and wavelength — and is independent of
the incident direction. Once $\mathbf{T}$ is known, *any* scattering
quantity (amplitude matrix, phase matrix, cross sections, polarimetric
observables) at any incident / scattered geometry follows from a cheap
rotation and contraction. The expensive step is computing
$\mathbf{T}$ itself.

For an axisymmetric particle, $\mathbf{T}$ is block-diagonal in the
azimuthal index $m$, and each block is found by inverting a matrix
built from surface integrals of spherical Bessel / Hankel functions
over the generating curve of the particle (a spheroid in the
rustmatrix case). This is the kernel that `rustmatrix` ports to Rust.

## The spheroidal approximation

Every hydrometeor model in `rustmatrix` — raindrops, oriented ice
columns, aggregates, graupel, hail — is a **spheroid** (oblate or
prolate) parameterised by

* an equivalent-volume radius $r_\mathrm{eq}$,
* an axis ratio $h/v$ (horizontal over vertical), and
* a complex refractive index $m$ at the radar wavelength.

For rain, the axis ratio is a function of equivolume diameter
(see [polarimetry](polarimetry) and [drop-shape relations in the
tutorials](../tutorials/02_raindrop_zdr)); the canonical choice in
`rustmatrix` is {cite}`Thurai2007`, with
{cite}`PruppacherBeard1970` and {cite}`BeardChuang1987` also tabulated
in `rustmatrix.tmatrix_aux`.

The spheroidal approximation is exact in the Mie limit (sphere) and
accurate for pristine columnar / plate ice; it becomes progressively
less faithful for aggregates. {cite}`Honeyager2013` argues that a
single well-parameterised spheroid still captures the bulk dual-
frequency signatures of aggregates, graupel, and dense ice, and the
[hydrometeor-class tutorial](../tutorials/11_honeyager_hydrometeor_classes)
reproduces that result with `rustmatrix`.

## Orientation averaging

Real populations of hydrometeors are not a single fixed orientation.
Raindrops flutter; ice columns cant around a mean orientation with a
Gaussian distribution of canting angles; tumbling aggregates approach
random orientation. `rustmatrix` supports

* **fixed orientation** (direct T-matrix at a specified canting angle),
* **Gaussian canting** around a mean (numerical quadrature over the
  canting PDF), and
* **full random orientation** (closed-form averaging of $\mathbf{T}$).

The Rust kernel parallelises the Gaussian-canting quadrature across
cores via `rayon`, which is where the ~6–430× speedups over
`pytmatrix` come from — orientation averaging is the hot loop in
nearly every PSD tabulation.

## PSD integration

A radar echo is not a single particle. It is an integral over the
particle size distribution (PSD):

$$
Z_{hh} = \frac{\lambda^4}{\pi^5 |K_w|^2}
         \int |S_{hh}(D)|^2 \, N(D)\, dD.
$$

`rustmatrix.psd.PSDIntegrator` precomputes the amplitude matrix
$\mathbf{S}(D)$ and phase matrix $\mathbf{Z}(D)$ on a grid of
diameters (done once per scatterer, in parallel Rust), then evaluates
arbitrary PSD shapes against the cached table — so changing a gamma
shape parameter is free after the first integration. See
{cite}`BringiChandrasekar2001` for the observable definitions and
[the PSD background page](psd.md) for the analytical forms.

## What rustmatrix inherits from pytmatrix

The numerical T-matrix core is a faithful port of the
Mishchenko/Leinonen FORTRAN implementation
{cite}`MishchenkoTravis1998` that `pytmatrix` also wraps — same
quadrature scheme, same orientation-averaging formulation, same drop-
shape tables. What changes is *how* the core is executed: Rust +
`rayon` in place of Fortran + GIL-bound Python glue. The
[parity tutorial](../tutorials/01_sphere_mie) shows the
T-matrix-at-sphere agreement with closed-form Mie to ~1e-4.

## Further reading

* {cite}`MishchenkoBook2002` — the canonical monograph; chapters 5–6
  cover the T-matrix in depth.
* {cite}`BringiChandrasekar2001` — the polarimetric-radar textbook
  that connects T-matrix outputs to every observable in
  `rustmatrix.radar`.
* {cite}`RauberNesbitt2018` — undergraduate-level radar meteorology
  treatment, the textbook `rustmatrix` was built to support.

```{bibliography}
:filter: docname in docnames
```
