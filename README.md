<p align="center">
  <img src="assets/logo.svg" alt="rustmatrix" width="560">
</p>

<p align="center">
  <a href="https://pypi.org/project/rustmatrix/"><img src="https://img.shields.io/pypi/v/rustmatrix.svg" alt="PyPI"></a>
  <a href="https://crates.io/crates/rustmatrix"><img src="https://img.shields.io/crates/v/rustmatrix.svg" alt="crates.io"></a>
  <a href="https://docs.rs/rustmatrix"><img src="https://img.shields.io/docsrs/rustmatrix" alt="docs.rs"></a>
  <a href="https://rustmatrix.readthedocs.io"><img src="https://img.shields.io/badge/docs-rustmatrix.readthedocs.io-blue.svg" alt="docs"></a>
</p>

# rustmatrix

**Rust-backed T-matrix scattering for nonspherical particles.** A
drop-in replacement for the numerical core of
[pytmatrix](https://github.com/jleinonen/pytmatrix), with
substantially faster orientation averaging, parallel PSD integration,
hydrometeor mixtures (`HydroMix`), and a full Doppler + polarimetric
spectra engine (`rustmatrix.spectra`). Targets **Python 3.9–3.13**
via ABI3 wheels.

Existing code that uses `pytmatrix.tmatrix.Scatterer`, `pytmatrix.psd`,
`pytmatrix.radar`, `pytmatrix.refractive`, or `pytmatrix.tmatrix_aux`
ports over by changing imports — the APIs are identical where the
physics matches.

## Documentation

**📖 Full docs: <https://rustmatrix.readthedocs.io>** — install
guide, 5-minute quickstart, 14 narrated tutorial notebooks,
background theory, every public symbol.

**🦀 Rust crate API: <https://docs.rs/rustmatrix>** — for using the
crate directly from Rust without the Python wrapper.

## Install

```bash
pip install rustmatrix
```

Pre-built ABI3 wheels for CPython 3.8+ ship for macOS (arm64, x86_64),
Linux (manylinux x86_64, aarch64), and Windows x64. For source builds
or dev installs, see
[the install guide](https://rustmatrix.readthedocs.io/en/latest/install.html).

## Quickstart

```python
from rustmatrix import Scatterer, radar
from rustmatrix.tmatrix_aux import wl_X, K_w_sqr, geom_horiz_back, dsr_thurai_2007
from rustmatrix.refractive import m_w_10C

s = Scatterer(radius=1.0, wavelength=wl_X, m=m_w_10C[wl_X],
              axis_ratio=1.0 / dsr_thurai_2007(2.0), Kw_sqr=K_w_sqr[wl_X])
s.set_geometry(geom_horiz_back)
print(f"Z_h = {radar.refl(s):.3f} mm⁶/m³")
```

Full walkthrough in the
[quickstart](https://rustmatrix.readthedocs.io/en/latest/quickstart.html)
— sphere → raindrop → PSD in five minutes.

## What's here

* **Core T-matrix solver** — Mishchenko's Fortran kernel, ported to
  Rust with the GIL released and `rayon`-parallel orientation
  averaging.
* **HydroMix** — combine multiple species (rain + ice + graupel) into
  a single Scatterer-shaped object.
* **Doppler + polarimetric spectra** — `SpectralIntegrator` with
  fall-speed presets, turbulence kernels (including Zhu 2023
  particle-inertia), beam broadening, optional system noise.
* **Pattern × scene integration** — `spectra.beam` for non-uniform
  beams where the closed-form σ_beam breaks down.
* **14 reproducible tutorials** covering published results from
  Kollias 2002 through Lakshmi 2024.

## Performance

~6× on orientation averaging, ~10× on PSD tabulation, ~430× on
`orient_averaged_adaptive` versus the Fortran pytmatrix backend. See
[the profiling how-to](https://rustmatrix.readthedocs.io/en/latest/howto/profiling.html)
for how to measure on your own workload, and
`benches/bench_vs_pytmatrix.py` for the head-to-head script.

## Development

See
[the contributing guide](https://rustmatrix.readthedocs.io/en/latest/contributing.html)
for dev install, tests, style, and release flow.

```bash
pytest tests/                                 # Python (97 tests)
cargo test --lib --release                    # Rust crate
cargo fmt --all && cargo clippy --all-targets -- -D warnings
```

## About

rustmatrix is developed by
[Steve Nesbitt](https://publish.illinois.edu/swnesbitt),
Professor in the Department of
[Climate, Meteorology & Atmospheric Sciences](https://climas.illinois.edu/)
at the University of Illinois Urbana-Champaign. It grows out of
instruction material for *ATMS 410 — Radar Meteorology* and
accompanies the textbook
*[Radar Meteorology: A First Course](https://onlinelibrary.wiley.com/doi/book/10.1002/9781118432662)*
(Rauber & Nesbitt, 2018, Wiley).

See also
[**myPSD**](https://github.com/swnesbitt/myPSD) — an interactive
web frontend for radar simulation that drives `rustmatrix` under
the hood.

## License

MIT — see [LICENSE](LICENSE).
