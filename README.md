<p align="center">
  <img src="assets/logo.svg" alt="rustmatrix" width="560">
</p>

# rustmatrix

**Rust-backed T-matrix scattering for nonspherical particles** — a drop-in
replacement for the numerical core of
[pytmatrix](https://github.com/jleinonen/pytmatrix), with the Fortran replaced
by pure Rust behind a PyO3 extension module. Targets **Python 3.9–3.13** via
ABI3 wheels.

If you have existing code that uses `pytmatrix.tmatrix.Scatterer`,
`pytmatrix.psd`, `pytmatrix.orientation`, `pytmatrix.radar`,
`pytmatrix.refractive`, or `pytmatrix.tmatrix_aux`, you should be able to
change the imports and keep going. The APIs are identical.

> **Status: alpha.** The core solver is numerically verified against the
> reference Fortran pytmatrix across spheres, spheroids, and finite cylinders
> at tolerances ≤ 5×10⁻³ (see [Status](#status)). Parallelised PSD paths
> run 4–400× faster on the same hardware.

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

Each script completes in under about 30 s on a laptop so the reader can
iterate. Every tutorial's module docstring notes which `pytmatrix` section
it mirrors.

---

## Capabilities at a glance

| rustmatrix module | pytmatrix counterpart | What it does | Key public symbols |
|---|---|---|---|
| `rustmatrix` | `pytmatrix.tmatrix` | Core solver + `Scatterer` class | `Scatterer`, `calctmat`, `mie_qsca`, `mie_qext`, `SHAPE_*`, `RADIUS_*` |
| `orientation` | `pytmatrix.orientation` | Orientation-averaging rules | `orient_single`, `orient_averaged_fixed`, `orient_averaged_adaptive`, `gaussian_pdf`, `uniform_pdf` |
| `psd` | `pytmatrix.psd` | Particle-size-distribution integration | `PSDIntegrator`, `GammaPSD`, `ExponentialPSD`, `UnnormalizedGammaPSD`, `BinnedPSD` |
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
