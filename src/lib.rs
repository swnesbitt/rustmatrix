//! # rustmatrix
//!
//! Rust-backed T-matrix scattering for nonspherical particles.
//!
//! This crate is a port of the numerical core of
//! [pytmatrix](https://github.com/jleinonen/pytmatrix), which itself wraps
//! M. I. Mishchenko's Fortran T-matrix code.
//!
//! ## Layout
//!
//! * [`quadrature`] – Gauss-Legendre nodes and weights (port of Mishchenko's `GAUSS`).
//! * [`special`] – spherical Bessel / Hankel functions, real and complex argument
//!   (port of `RJB`, `RYB`, `CJB`).
//! * [`wigner`] – associated Legendre / Wigner d-function helpers (`VIG`, `VIGAMPL`).
//! * [`shapes`] – particle shape functions (`RSP1..RSP4`), surface-area helpers.
//! * [`mie`] – closed-form Mie scattering for spheres; used as the axis_ratio=1
//!   reference for parity testing.
//! * [`tmatrix`] – the general spheroid/cylinder/Chebyshev T-matrix solver
//!   (direct port of `CALCTMAT`, `CONST`, `VARY`, `TMATR0`, `TMATR`).
//! * [`amplitude`] – amplitude and phase matrix evaluation (`CALCAMPL`, `AMPL`).
//! * [`pybindings`] – PyO3 bindings exposing `calctmat` / `calcampl` to Python.

// The numerical modules are near line-for-line ports of Mishchenko's
// Fortran and intentionally preserve the original index conventions,
// long function signatures, and tight loops. A handful of clippy lints
// fight that style; silence them at the crate level rather than
// reshuffling code and risking a translation bug.
#![allow(
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::needless_range_loop,
    clippy::many_single_char_names,
    clippy::excessive_precision,
    clippy::approx_constant,
    clippy::useless_conversion,
    clippy::manual_range_contains,
    clippy::let_and_return,
    clippy::collapsible_if,
    clippy::collapsible_else_if,
    clippy::redundant_field_names,
    clippy::redundant_closure,
    clippy::neg_multiply,
    clippy::op_ref,
    clippy::assign_op_pattern,
    clippy::identity_op,
    clippy::unnecessary_cast,
    clippy::needless_late_init,
    clippy::comparison_chain,
    clippy::float_cmp,
    clippy::only_used_in_recursion,
    clippy::manual_rem_euclid,
    clippy::derivable_impls,
    clippy::manual_div_ceil
)]

pub mod amplitude;
pub mod mie;
pub mod quadrature;
pub mod shapes;
pub mod special;
pub mod tmatrix;
pub mod wigner;

mod pybindings;

use pyo3::prelude::*;

/// Python module entrypoint. Maturin wires this up as `rustmatrix._core`.
///
/// `gil_used = false` declares the module thread-safe, so free-threaded
/// CPython (3.13t+) keeps the GIL disabled when it is imported. This is
/// sound because all exposed state is immutable after construction
/// (`TMatrixHandle` is a frozen pyclass over plain `Send + Sync` data)
/// and every entrypoint detaches from the runtime for its heavy compute.
#[pymodule(gil_used = false)]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pybindings::register(m)
}
