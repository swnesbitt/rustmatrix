<p align="center">
  <img src="assets/logo.svg" alt="rustmatrix" width="560">
</p>

# rustmatrix

**Rust-backed T-matrix scattering for nonspherical particles — now with
Doppler and polarimetric spectra.** A drop-in replacement for the
numerical core of
[pytmatrix](https://github.com/jleinonen/pytmatrix), with the Fortran
replaced by pure Rust behind a PyO3 extension module, extended with
hydrometeor mixtures (`HydroMix`) and a full Doppler-spectrum engine
(`rustmatrix.spectra`) that produces sZ_h, sZ_dr, sK_dp, sρ_hv, and
sδ_hv for single-species or mixed-phase PSDs, with literature fall-speed
presets, Gaussian turbulence, Zeng et al. 2023 particle-inertia
turbulence, finite-beamwidth broadening, and optional system noise
baked in. Targets
**Python 3.9–3.13** via ABI3 wheels.

If you have existing code that uses `pytmatrix.tmatrix.Scatterer`,
`pytmatrix.psd`, `pytmatrix.orientation`, `pytmatrix.radar`,
`pytmatrix.refractive`, or `pytmatrix.tmatrix_aux`, you should be able to
change the imports and keep going. The APIs are identical.

---

## Background: the T-matrix method in 60 seconds

Radar meteorology depends on knowing how raindrops, ice crystals, and
hydrometeors scatter microwaves at radar wavelengths. The scattering is
captured by two matrices:

* the **amplitude matrix** *S* (2×2, complex) — relates the incident and
  scattered electric-field components.
* the **phase matrix** *Z* (4×4, real) — relates incident and scattered
  Stokes parameters and feeds straight into polarimetric radar observables
  (*Z*, *Z*<sub>dr</sub>, *K*<sub>dp</sub>, …).

For a sphere, classical Mie theory gives *S* and *Z* in closed form. For a
nonspherical particle — an oblate raindrop, a columnar ice crystal — you
need a method that handles arbitrary shapes and orientations. The
**T-matrix method** (Waterman 1971; Mishchenko 1991) expands the incident
and scattered fields in spherical vector wave functions and solves for the
expansion coefficients that link them. The key property is that the T-matrix
depends only on the particle's *shape, size, and refractive index* — the
incident direction enters only at the very end. Solve the T-matrix once,
reuse it across every incidence/scattering geometry.

This package is a Rust port of the numerical core of
[pytmatrix](https://github.com/jleinonen/pytmatrix), which itself wraps
[Mishchenko's Fortran T-matrix code](https://www.giss.nasa.gov/staff/mmishchenko/t_matrix.html).
The motivation is portable builds, modern tooling, and the freedom to
parallelise the PSD and orientation loops in Rust with the GIL released.

**References.** Mishchenko & Travis (1998), *JQSRT* 60(3): 309–324, for
the algorithm. Leinonen (2014), *Optics Express* 22: 1655, for pytmatrix.

---

## Installation

```bash
# From a checkout:
git clone <your-fork-of-this-repo> rustmatrix
cd rustmatrix

# Dev install — builds the Rust extension and puts it on sys.path.
pip install maturin
maturin develop --release

# Or build a wheel:
maturin build --release
pip install target/wheels/rustmatrix-*.whl
```

Requires a Rust toolchain (`rustup default stable`, 1.75+) and Python 3.9+.
For parity testing against the original, also `pip install pytmatrix` (needs
`gfortran`).

---

## Quick start

```python
from rustmatrix import Scatterer

s = Scatterer(
    radius=1.0,                  # mm — equal-volume-sphere radius
    wavelength=33.3,             # mm — X-band
    m=complex(7.99, 2.21),       # water refractive index at 10 GHz, 10 °C
    axis_ratio=1.0,              # 1.0 = sphere; >1 oblate, <1 prolate
    ddelt=1e-4,                  # T-matrix convergence tolerance
    ndgs=2,                      # quadrature density factor
)
# (thet0, thet, phi0, phi, alpha, beta): incidence/scatter + Euler
s.set_geometry((90.0, 90.0, 0.0, 180.0, 0.0, 0.0))  # horizontal backscatter
S, Z = s.get_SZ()
```

`S` is a 2×2 complex numpy array; `Z` is 4×4 real. All downstream helpers
(`rustmatrix.radar`, `rustmatrix.scatter`) take a `Scatterer` and do the
bookkeeping for you — see the tutorials below.

Shape constants follow pytmatrix's conventions (`SHAPE_SPHEROID = -1`,
`SHAPE_CYLINDER = -2`, `SHAPE_CHEBYSHEV = 1`).

---

## Added functionality

rustmatrix is a drop-in replacement for the numerical core of pytmatrix,
but the port isn't just a transliteration. Three things are genuinely new:

### 1. Rust speedups

The inner loops that dominate most radar-forward-model runs — orientation
averaging and PSD integration — run in Rust with the GIL released and
parallelise across cores via `rayon`. Against the Fortran pytmatrix
backend on the same machine:

* **~6×** on orientation averaging (fixed quadrature, 32 orientations).
* **~4–10×** on PSD tabulation (32–64 diameter points; combined with
  orient averaging, ~10×).
* **~260–300×** on `angular_integration` (σ_sca / σ_ext / g tables).
* **~430×** on `orient_averaged_adaptive` — the worst pytmatrix case,
  where scipy's `dblquad` callbacks cross the Python/Fortran boundary
  hundreds of times per diameter.

Full benchmark table in the [Performance](#performance) section; rerun
with `python benches/bench_vs_pytmatrix.py`.

### 2. Hydrometeor mixtures (`HydroMix`)

Radar volumes sometimes contain more than a single species. The new `rustmatrix.hd_mix`
module combines multiple `(Scatterer, PSD)` pairs — each with its own
refractive index, axis ratio, shape, and orientation PDF — into one
Scatterer-shaped object that feeds straight into the existing
`rustmatrix.radar` helpers:

```python
from rustmatrix import HydroMix, MixtureComponent, Scatterer, radar, psd
from rustmatrix.tmatrix_aux import geom_horiz_back, geom_horiz_forw

# rain_scatterer and ice_scatterer are Scatterers whose PSDIntegrators
# have already been init_scatter_table()'d at both geometries.
rain_psd = psd.GammaPSD(D0=1.5, Nw=8e3, mu=4, D_max=6.0)
ice_psd  = psd.ExponentialPSD(N0=5e3, Lambda=2.0, D_max=8.0)

mix = HydroMix([
    MixtureComponent(rain_scatterer, rain_psd, "rain"),
    MixtureComponent(ice_scatterer,  ice_psd,  "ice"),
])

mix.set_geometry(geom_horiz_back)
Zh, Zdr, rho = radar.refl(mix), radar.Zdr(mix), radar.rho_hv(mix)
mix.set_geometry(geom_horiz_forw)
Kdp, Ai      = radar.Kdp(mix), radar.Ai(mix)
```

The physics: both the amplitude matrix *S* and the phase matrix *Z* are
linear functionals of *N(D)*, so summing *S* and *Z* across species is
the physically correct incoherent-mixture recipe — even for the
non-linear observables (*Z*<sub>dr</sub>, *ρ*<sub>hv</sub>,
*δ*<sub>hv</sub>), which fall out automatically from the combined
matrix. Mixing fractions live directly in each species' N(D); there is
no separate weight scalar. Full end-to-end example in
[`examples/06_hd_mix.py`](examples/06_hd_mix.py).

### 3. Doppler + polarimetric spectra (`rustmatrix.spectra`)

Profilers, cloud radars, and modern disdrometers measure *spectra* — the
Doppler-velocity-resolved version of the bulk polarimetric observables.
`rustmatrix.spectra` reuses the per-diameter *S* and *Z* tables that
`PSDIntegrator` already caches and produces
*sZ*<sub>h</sub>(v), *sZ*<sub>dr</sub>(v), *sK*<sub>dp</sub>(v),
*sρ*<sub>hv</sub>(v), and *sδ*<sub>hv</sub>(v) in one call:

```python
from rustmatrix import SpectralIntegrator, spectra
from rustmatrix.tmatrix_aux import geom_vert_back, geom_vert_forw

integ = SpectralIntegrator(
    rain_scatterer,                            # or a HydroMix
    fall_speed=spectra.fall_speed.atlas_srivastava_sekhon_1973,
    turbulence=spectra.InertialZeng2023(sigma_air=0.5, epsilon=1e-3),
    v_min=-1.0, v_max=12.0, n_bins=1024,
    u_h=8.0, beamwidth=np.deg2rad(1.0),        # optional beam-broadening
    geometry_backscatter=geom_vert_back,
    geometry_forward=geom_vert_forw,
)
res = integ.run()
bulk = res.collapse_to_bulk()                  # round-trip to radar.* helpers
```

What ships out of the box:

* **Fall-speed presets** — Atlas–Srivastava–Sekhon 1973, Brandes 2002,
  Beard 1976, Locatelli–Hobbs 1974 (aggregates and graupel), plus a
  first-class `power_law(a, b, …)` factory and any user-supplied
  `D → v_t` callable.
* **Turbulence models** — `NoTurbulence`, `GaussianTurbulence(σ_t)`,
  and `InertialZeng2023` (diameter-dependent σ_t(D) via Stokes-number
  low-pass response). A `spectra.turbulence.from_params(sigma=…,
  epsilon=…)` helper picks Zeng 2023 automatically whenever ε is
  supplied.
* **Finite-beamwidth broadening** — `|u_h|·θ_b/(2√(2 ln 2))` added in
  quadrature to σ_t, so realistic profiler geometries are one
  keyword away.
* **Optional system noise** — pass `noise="realistic"` (or a scalar /
  `(noise_h, noise_v)` tuple in mm⁶ m⁻³) to add a thermal noise floor
  that biases `sZ_dr`, `sρ_hv`, and `sLDR` the way a real receiver
  would. The underlying `S_spec` / `Z_spec` stay signal-only so
  `collapse_to_bulk()` still round-trips exactly.
* **HydroMix support** — pass a `HydroMix` with per-component
  `(fall_speed, turbulence)` pairs and the integrator sums each
  component's spectrum on a shared velocity grid. Bimodal rain+ice
  spectra with spectral ρ_hv dips between the modes fall out directly.
* **Exact bulk round-trip** — `SpectralResult.collapse_to_bulk()` sums
  *S*(v) and *Z*(v) over velocity and hands the result to the existing
  `rustmatrix.radar` helpers, so the spectral and bulk codepaths agree
  to numerical tolerance for every observable.

Worked examples that exercise this machinery against published results
ship as tutorials 07–11 — see the *Guided tour* below.

---

## Guided tour

The `examples/` directory contains a numbered tutorial set. Each tutorial
ships as both a runnable `.py` script and a matching `.ipynb` notebook
(same code, plus matplotlib plots and narrative markdown):

1. **`01_sphere_mie.py`** — sphere at X-band. Verifies the T-matrix against
   closed-form Mie. Mirrors the sanity checks in
   `tests/test_parity_pytmatrix.py::test_sphere_parity`.
2. **`02_raindrop_zdr.py`** — one 2 mm oblate raindrop at C-band. Computes
   *Z*<sub>h</sub>, *Z*<sub>dr</sub>, *δ*<sub>hv</sub> via
   `rustmatrix.radar`, using `tmatrix_aux.dsr_thurai_2007` for the axis
   ratio and the tabulated 10 °C water index.
3. **`03_psd_gamma_rain.py`** — gamma PSD across 0.1–8 mm. Tabulates *S*, *Z*
   once with `PSDIntegrator`, then evaluates *Z*<sub>h</sub>,
   *Z*<sub>dr</sub>, *K*<sub>dp</sub>, *A*<sub>i</sub> across rain rates.
   This is where the parallel Rust tabulator earns its speedup.
4. **`04_oriented_ice.py`** — pristine columnar ice crystal at W-band with a
   Gaussian canting PDF. Compares `orient_averaged_fixed` (fast, smooth)
   against `orient_averaged_adaptive` (accurate, expensive) and uses
   `refractive.mi` for the ice index.
5. **`05_radar_band_sweep.py`** — same particle across S/C/X/Ku/Ka/W. Shows
   how *Z*<sub>dr</sub> and *K*<sub>dp</sub> vary with wavelength and is the
   natural jumping-off point for multi-frequency retrieval papers.
6. **`06_hd_mix.py`** — a rain + oriented-ice mixture at C-band using the
   new `HydroMix`. Builds one `PSDIntegrator` per species, assembles the
   mixture, and reads *Z*<sub>h</sub>, *Z*<sub>dr</sub>, *K*<sub>dp</sub>,
   *A*<sub>i</sub>, and *ρ*<sub>hv</sub> for rain-only, ice-only, and the
   combined case.
7. **`07_doppler_spectrum_rain.py`** — reproduces
   [Kollias, Albrecht, Marks 2002](https://doi.org/10.1029/2001JD002033):
   the 94-GHz raindrop σ<sub>b</sub>(*D*) Mie oscillations map into the
   Doppler spectrum so that the *first Mie minimum* (≈5.9 m/s in still air)
   serves as a DSD-independent fiducial for retrieving mean vertical air
   motion. Also illustrates the delta-binning artifact that can masquerade
   as physics when the PSD is sampled too sparsely.
8. **`08_spectral_polarimetry_rain_ice.py`** — reproduces the
   dual-frequency non-Rayleigh snowfall signature from
   [Billault-Roux et al. 2023, *AMT*](https://doi.org/10.5194/amt-16-911-2023):
   identical snow scatterer, PSD, fall-speed, and turbulence run through
   `SpectralIntegrator` at X-band and W-band, yielding sDWR(*v*) = 10·log<sub>10</sub>(sZ<sub>X</sub>/sZ<sub>W</sub>)
   that stays near 0 dB at low velocities and rises to several dB at high
   velocities — the large-particle fingerprint the paper exploits.
9. **`09_zhu_2023_particle_inertia.py`** — reproduces the W-band,
   exponential-warm-rain configuration of
   [Zhu, Kollias, Yang 2023](https://zenodo.org/records/7897981) and
   compares conventional Gaussian broadening to the inertia-aware
   `InertialZeng2023` kernel. Large drops under-respond to small-scale
   eddies, so the Doppler spectrum narrows on its fast-falling tail — the
   paper's central finding.
10. **`10_slw_vs_snow.py`** — supercooled liquid water and snow as
    independent `rustmatrix` scatterers, combined in a `HydroMix` to
    produce the bimodal W-band Doppler spectrum that motivates the
    mixed-phase analysis in
    [Billault-Roux et al. 2023, *ACP*](https://doi.org/10.5194/acp-23-10207-2023).
    Highlights the 40+ dB Z<sub>h</sub> gap between SLW droplets and snow
    aggregates and the velocity separation of the two modes.
11. **`11_honeyager_hydrometeor_classes.py`** — following the T-matrix-as-DDA-proxy
    framing of
    [Honeyager 2013](https://fsu.digital.flvc.org/islandora/object/fsu:207427)
    (MS thesis, Florida State University), parameterises four representative
    hydrometeor classes — rain, low-density aggregate, graupel, high-density
    ice — by (ρ<sub>eff</sub>, axis ratio) and walks through single-particle
    σ<sub>b</sub>(*D*), DWR(*D*), bulk DWR vs. *D*<sub>0</sub>, and the
    spectral DWR(*v*) that tells an aggregate PSD apart from a graupel PSD
    even when their bulk Z<sub>h</sub> values overlap.

Each script completes in under about 30 s on a laptop so the reader can
iterate. Every tutorial's module docstring notes which `pytmatrix` section
it mirrors (tutorials 6–11 cover functionality new to rustmatrix).

---

## Capabilities at a glance

| rustmatrix module | pytmatrix counterpart | What it does | Key public symbols |
|---|---|---|---|
| `rustmatrix` | `pytmatrix.tmatrix` | Core solver + `Scatterer` class | `Scatterer`, `calctmat`, `mie_qsca`, `mie_qext`, `SHAPE_*`, `RADIUS_*` |
| `orientation` | `pytmatrix.orientation` | Orientation-averaging rules | `orient_single`, `orient_averaged_fixed`, `orient_averaged_adaptive`, `gaussian_pdf`, `uniform_pdf` |
| `psd` | `pytmatrix.psd` | Particle-size-distribution integration | `PSDIntegrator`, `GammaPSD`, `ExponentialPSD`, `UnnormalizedGammaPSD`, `BinnedPSD` |
| `hd_mix` | *(new — no pytmatrix equivalent)* | Multi-species hydrometeor mixtures. Scatterer-shaped so `radar.*` helpers work unchanged. | `HydroMix`, `MixtureComponent` |
| `spectra` | *(new — no pytmatrix equivalent)* | Doppler + polarimetric spectra (sZ_h, sZ_dr, sK_dp, sρ_hv, sδ_hv) for single species or `HydroMix`, with fall-speed presets, Gaussian / Zeng 2023 turbulence, and beam broadening. | `SpectralIntegrator`, `SpectralResult`, `fall_speed.*`, `GaussianTurbulence`, `InertialZeng2023`, `NoTurbulence` |
| `radar` | `pytmatrix.radar` | Polarimetric radar observables | `radar_xsect`, `refl` (`Zi`), `Zdr`, `delta_hv`, `rho_hv`, `Kdp`, `Ai` |
| `scatter` | `pytmatrix.scatter` | Angular-integrated scattering helpers | `sca_intensity`, `sca_xsect`, `ext_xsect`, `ssa`, `asym`, `ldr` |
| `refractive` | `pytmatrix.refractive` | Refractive-index helpers | `mg_refractive`, `bruggeman_refractive`, `m_w_0C/10C/20C`, `mi`, `ice_refractive` |
| `tmatrix_aux` | `pytmatrix.tmatrix_aux` | Radar-band presets + drop-shape relations | `wl_S…wl_W`, `K_w_sqr`, `geom_horiz_back/forw`, `dsr_thurai_2007`, `dsr_pb`, `dsr_bc` |
| `quadrature` | `pytmatrix.quadrature` | Gautschi quadrature for orientation PDFs | `get_points_and_weights`, `discrete_gautschi` |

---

## Migrating from pytmatrix

The common cases need no code changes beyond the imports:

```python
# before
from pytmatrix.tmatrix import Scatterer
from pytmatrix import orientation, psd, radar, refractive, tmatrix_aux

# after
from rustmatrix import Scatterer
from rustmatrix import orientation, psd, radar, refractive, tmatrix_aux
```

Notes:

* The `Scatterer` constructor, attributes (`radius`, `wavelength`, `m`,
  `axis_ratio`, `shape`, `ddelt`, `ndgs`, `alpha`, `beta`, `thet0/thet`,
  `phi0/phi`, `Kw_sqr`, `orient`, `or_pdf`, `n_alpha`, `n_beta`,
  `psd_integrator`, `psd`), and methods (`get_S`, `get_Z`, `get_SZ`,
  `set_geometry`, `get_geometry`) are identical.
* Shape / radius constants use the same integer values.
* `PSDIntegrator.init_scatter_table` is transparently parallelised across
  diameters via rayon — no caller changes.
* Legacy pytmatrix kwargs (`axi`, `lam`, `eps`, `rat`, `np`, `scatter`) still
  work but raise a `DeprecationWarning`.

---

## Status

Numerically verified against pytmatrix (Fortran backend). All seven parity
tests pass at the tolerances below:

| Shape | Test | Tolerance |
|---|---|---|
| Sphere (`axis_ratio=1`) — 3 cases | ✅ pass | 1×10⁻³ |
| Prolate spheroid (`axis_ratio=0.5`) | ✅ pass | 5×10⁻³ |
| Oblate spheroid (`axis_ratio=2.0`) | ✅ pass | 5×10⁻³ |
| Spheroid (`axis_ratio=1.5`) | ✅ pass | 5×10⁻³ |
| Finite cylinder | ✅ pass | 5×10⁻³ |

Additional PSD / orientation / angular-integration parity tests live in
`tests/test_parity_pytmatrix.py`.

Fully implemented and unit-tested (`cargo test --lib`, `pytest`):

* Gauss-Legendre quadrature, spherical Bessel functions, Wigner *d*,
  shape radii (spheroid / Chebyshev / cylinder / generalised Chebyshev),
  closed-form Mie baseline.
* T-matrix blocks `tmatr0` (m = 0) and `tmatr` (m > 0); amplitude-matrix
  rotation/summation `ampl`.
* Full PyO3 Python API: `calctmat`, `calcampl`, `Scatterer`.
* Orientation averaging: `orient_single`, `orient_averaged_fixed`
  (Gauss quadrature in β, uniform sampling in α),
  `orient_averaged_adaptive` (scipy `dblquad` on the Python side, fixed
  GL grid on the Rust fast path).
* PSD classes (`GammaPSD`, `ExponentialPSD`, `UnnormalizedGammaPSD`,
  `BinnedPSD`) and `PSDIntegrator`. All four tabulation paths —
  single-orient, fixed-orient-avg, adaptive-orient-avg,
  `angular_integration=True` — are parallelised across diameters with
  the GIL released.
* Polarimetric radar observables (`radar_xsect`, `refl`/`Zi`, `Zdr`,
  `delta_hv`, `rho_hv`, `Kdp`, `Ai`) and angular-integrated helpers
  (`sca_xsect`, `ext_xsect`, `ssa`, `asym`, `ldr`, `sca_intensity`).
* Refractive-index helpers: Maxwell-Garnett / Bruggeman EMAs; tabulated
  water indices at 0/10/20 °C across the six radar bands; Warren ice
  interpolator (`refractive.mi`).
* Radar-band presets (`wl_S`…`wl_W`, `K_w_sqr`), canned geometries, and
  Thurai / Pruppacher-Beard / Beard-Chuang drop-shape relations.

---

## Performance

Against the Fortran `pytmatrix` backend on the same machine (Apple M-series,
`benches/bench_vs_pytmatrix.py`; positive ratio = rustmatrix faster):

| Workload | pytmatrix (Fortran) | rustmatrix (Rust) | Speedup |
|---|---:|---:|---:|
| `calctmat` only (spheroid, ax = 1.5) | 0.22 ms | 0.21 ms | 1.1× |
| Single orientation, cold (fresh `Scatterer`) | 0.23 ms | 0.26 ms | 0.9× (slower) |
| Cached re-evaluation (warm T-matrix) | 0.01 ms | 0.00 ms | 1.6× |
| Orientation-averaged fixed (4 × 8 = 32 orientations) | 4.26 ms | 0.75 ms | **5.7×** |
| PSD `init_scatter_table`, 32 points | 12.3 ms | 2.2 ms | **5.7×** |
| PSD `init_scatter_table`, 64 points | 13.5 ms | 3.4 ms | **4.0×** |
| PSD + orient-avg (4 × 8), 32 points | 13.8 ms | 1.7 ms | **8.2×** |
| PSD + orient-avg (4 × 8), 64 points | 23.4 ms | 2.4 ms | **9.7×** |
| PSD + `angular_integration`, 32 points | 13 345 ms | 52 ms | **258×** |
| PSD + `angular_integration`, 64 points | 26 210 ms | 87 ms | **300×** |
| PSD + orient-avg-adaptive, 4 points | 1 758 ms | 4.1 ms | **433×** |

Headline notes:

* The core T-matrix solve is roughly tied — optimised Fortran plus LAPACK
  is hard to beat on pure linear algebra.
* The big wins come from moving the outer loops into Rust: orientation
  averaging (~6× on a single particle) and PSD tabulation (~4× single-orient,
  ~10× combined with orient averaging), because per-diameter T-matrix solves
  are independent and rayon parallelises them across cores with the GIL
  released.
* The 100×–400× speedups on `angular_integration` and
  `orient_averaged_adaptive` come from replacing scipy's per-diameter
  `dblquad` callbacks with a fixed Gauss-Legendre product grid evaluated
  inside Rust. The callbacks cross the Python/Fortran boundary hundreds of
  times per diameter on pytmatrix; the Rust path amortises the T-matrix
  solve across the whole grid and runs diameters in parallel.
* The ~16% slowdown on the "single orient cold" case is Python/PyO3
  boundary overhead that f2py avoids. It disappears as soon as the
  T-matrix is reused (cached re-eval, orient averaging, PSD tabulation).

Reproduce with:

```bash
pip install pytmatrix    # needs gfortran
python benches/bench_vs_pytmatrix.py
```

---

## Architecture

```
rustmatrix/
├─ Cargo.toml                  # Rust crate (PyO3 + maturin)
├─ pyproject.toml              # Python build (maturin backend)
├─ src/
│  ├─ lib.rs                   # Module root + Python extension entrypoint
│  ├─ quadrature.rs            # Gauss-Legendre (port of Mishchenko's GAUSS)
│  ├─ special.rs               # spherical Bessel (RJB, RYB, CJB)
│  ├─ wigner.rs                # Wigner d-functions (VIG, VIGAMPL)
│  ├─ shapes.rs                # Particle shapes (RSP1/2/3/4)
│  ├─ mie.rs                   # Closed-form Mie for the sphere limit
│  ├─ tmatrix.rs               # T-matrix solver (CALCTMAT, CONST, VARY, TMATR0, TMATR)
│  ├─ amplitude.rs             # Amplitude + phase matrix (CALCAMPL, AMPL)
│  └─ pybindings.rs            # PyO3 exposure (incl. parallel PSD tabulators)
├─ python/rustmatrix/         # Pure-Python surface (Scatterer, psd, radar, …)
├─ tests/                      # pytest + parity suite
├─ examples/                   # numbered tutorials (.py + .ipynb)
├─ benches/bench_vs_pytmatrix.py
└─ .github/workflows/ci.yml
```

Heavy linear algebra (complex LU for the Q matrix) is handled by `nalgebra`,
replacing the `lpd.f` LAPACK routines that ship with the original Fortran
backend.

---

## Running the tests

```bash
# Rust-only tests (fast, no Python needed):
cargo test --lib

# Full Python test suite (requires maturin develop first):
pytest -v tests/

# Parity tests against pytmatrix (requires pytmatrix installed):
pip install pytmatrix
pytest -v tests/test_parity_pytmatrix.py
```

CI runs the Rust + Python tests across Python 3.10 / 3.11 / 3.12 / 3.13
on Linux. Parity tests against pytmatrix run locally when the package is
installed; they're not in CI because pytmatrix needs `gfortran`.

---

## License

MIT. See [LICENSE](./LICENSE).

Upstream credits:

* Jussi Leinonen — pytmatrix (the Python wrapper and the modifications
  that made the Mishchenko code MIT-compatible).
* Michael I. Mishchenko (NASA GISS) — the underlying T-matrix Fortran code.
