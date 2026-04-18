# rupytmatrix

**Rust-backed T-matrix scattering for nonspherical particles.**

A port of the numerical core of [pytmatrix](https://github.com/jleinonen/pytmatrix) —
itself a Python wrapper around M. I. Mishchenko's Fortran T-matrix code —
with the Fortran replaced by pure Rust behind a PyO3 extension module.
Targets **Python 3.9–3.13** via ABI3.

> **Status: alpha.** The core T-matrix solver is numerically verified against
> the original pytmatrix (Fortran backend) for spheres, prolate/oblate
> spheroids, and finite cylinders. All 7 parity tests pass. See [Status](#status)
> below for specifics.

## Why?

* Replace a Fortran dependency with a pure Rust dependency that cross-compiles
  cleanly to every platform Python 3.13 cares about (including Apple Silicon
  and Windows, where the original pytmatrix has historically been awkward).
* Avoid the `numpy.f2py` / `distutils` build path, which broke in Python 3.12+.
* Modern build tooling (maturin + PyO3 0.22, abi3 wheels).

## Installation

```bash
# From a checkout:
git clone <your-fork-of-this-repo> rupytmatrix
cd rupytmatrix

# Dev install — builds the Rust extension and puts it on sys.path.
pip install maturin
maturin develop --release

# Or build a wheel:
maturin build --release
pip install target/wheels/rupytmatrix-*.whl
```

Requires a Rust toolchain (`rustup default stable`, 1.75+) and Python 3.9+.

## Usage

The `Scatterer` class is API-compatible with `pytmatrix.tmatrix.Scatterer`:

```python
from rupytmatrix import Scatterer

s = Scatterer(
    radius=1.0,                     # mm, equal-volume-sphere radius
    wavelength=33.3,                # mm, X-band
    m=complex(7.99, 2.21),          # water at 10 GHz
    axis_ratio=1.0,                 # 1.0 = sphere
    ddelt=1e-4,                     # convergence tolerance
    ndgs=2,                         # quadrature density factor
)
s.set_geometry((90.0, 90.0, 0.0, 180.0, 0.0, 0.0))
S, Z = s.get_SZ()
```

Shape constants follow pytmatrix's conventions (`SHAPE_SPHEROID = -1`,
`SHAPE_CYLINDER = -2`, `SHAPE_CHEBYSHEV = 1`).

See `examples/basic_usage.py`.

## Architecture

```
rupytmatrix/
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
│  └─ pybindings.rs            # PyO3 exposure
├─ python/rupytmatrix/
│  ├─ __init__.py              # Public Python API
│  ├─ scatterer.py             # Scatterer class (matches pytmatrix signature)
│  └─ _core.pyi                # Type stubs for the Rust extension
├─ tests/
│  ├─ conftest.py              # pytmatrix-availability fixture
│  ├─ test_mie.py              # Mie unit tests
│  ├─ test_scatterer_api.py    # API smoke tests
│  ├─ test_parity_pytmatrix.py # Parity vs. original pytmatrix (skipped if missing)
│  └─ test_quadrature_python.py
├─ examples/basic_usage.py
├─ benches/                    # (empty — reserved for cargo-bench suites)
└─ .github/workflows/ci.yml
```

Heavy linear algebra is handled by `nalgebra` (complex LU inversion for the
Q matrix). This replaces the `lpd.f` LAPACK routines that ship with the
original pytmatrix Fortran backend.

## Status

**Numerically verified** against pytmatrix (Fortran backend) for all supported
shapes. All 7 parity tests pass at tolerances ≤ 5×10⁻³ for S and Z:

| Shape | Test | Tolerance |
|---|---|---|
| Sphere (`axis_ratio=1`) — 3 cases | ✅ pass | 1×10⁻³ |
| Prolate spheroid (`axis_ratio=0.5`) | ✅ pass | 5×10⁻³ |
| Oblate spheroid (`axis_ratio=2.0`) | ✅ pass | 5×10⁻³ |
| Spheroid (`axis_ratio=1.5`) | ✅ pass | 5×10⁻³ |
| Finite cylinder | ✅ pass | 5×10⁻³ |

Fully implemented and unit-tested (`cargo test --lib`, `pytest`):

* Gauss-Legendre quadrature with endpoint / half-range options.
* Spherical Bessel `j_n`, `y_n` (real argument, up-recurrence).
* Spherical Bessel `j_n` of complex argument (down-recurrence, Mishchenko's CJB).
* Riccati-Bessel wrappers with correct derivative conventions.
* Wigner d-function helpers `VIG` / `VIGAMPL`.
* Shape radii for spheroid, Chebyshev, cylinder, gen-Chebyshev.
* Closed-form Mie scattering (sphere baseline).
* `tmatrix::tmatr0` — `m = 0` azimuthal block of T.
* `tmatrix::tmatr` — `m > 0` azimuthal blocks of T.
* `amplitude::ampl` — amplitude matrix rotation / summation.
* Full PyO3 Python API: `calctmat`, `calcampl`, `Scatterer`.
* Orientation averaging: `orient_single` (default), `orient_averaged_fixed`
  (Gauss quadrature in β, uniform sampling in α), and
  `orient_averaged_adaptive` (scipy `dblquad`). Ported pure-Python from
  pytmatrix and parity-verified against it.
* `gaussian_pdf` / `uniform_pdf` orientation PDFs and the Gautschi-based
  `get_points_and_weights` quadrature helper (used internally by
  `orient_averaged_fixed`).

Not yet implemented:

* Size distribution integration (`psd_integrator`).
* Radar / PSD / refractive-index helper modules. These are pure Python in
  pytmatrix and can be copied over verbatim; they don't touch the Fortran core.

## Running the tests

```bash
# Rust-only tests (fast, no Python needed):
cargo test --lib

# Full test suite (requires maturin develop first):
pytest -v tests/

# Parity tests against pytmatrix (requires pytmatrix installed):
pip install pytmatrix
pytest -v tests/test_parity_pytmatrix.py
```

CI runs the Rust + Python tests across Python 3.10 / 3.11 / 3.12 / 3.13
on Linux. Parity tests against pytmatrix run locally when the package is
installed; they're not in CI because pytmatrix needs gfortran.

## License

MIT. See [LICENSE](./LICENSE).

Upstream credits:

* Jussi Leinonen — pytmatrix (the Python wrapper and modifications that
  made the Mishchenko code MIT-compatible).
* Michael I. Mishchenko (NASA GISS) — the underlying T-matrix Fortran code.
