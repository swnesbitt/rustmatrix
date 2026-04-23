# Rust API

The `rustmatrix` Python package wraps a Rust crate of the same name —
the Rust crate is where the T-matrix solver, orientation-averaging
loops, and PSD integrator actually live. Python callers never need to
touch it directly; it's here because some users want to read the
numerical kernels, embed them in a non-Python program, or extend the
crate with custom functionality.

The Rust API reference is hosted on **[docs.rs/rustmatrix](https://docs.rs/rustmatrix)**
— the standard Rust documentation host, automatically built and
versioned per crate release.

## What's in the crate

| Module | What it does |
|---|---|
| `rustmatrix::tmatrix` | The ported T-matrix solver (Mishchenko's Fortran core → Rust). |
| `rustmatrix::orientation` | Orientation-averaging loops with GIL released + `rayon` parallelism. |
| `rustmatrix::psd` | PSD tabulation + per-diameter amplitude/phase matrix caches. |
| `rustmatrix::amplitude` | Amplitude-matrix rotations (fast geometry changes). |
| `rustmatrix::constants` | Physical constants + `wl_*` radar-band presets. |

## Using the crate without Python

Add it to your `Cargo.toml`:

```toml
[dependencies]
rustmatrix = "2.1"
```

Then see the [docs.rs](https://docs.rs/rustmatrix) reference for the
Rust-native API. The Python wrappers in this package are thin — most of
what you can do from Python, you can do from pure Rust with a few more
lines of setup.
