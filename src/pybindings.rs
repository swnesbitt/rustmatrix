//! PyO3 bindings that expose the Rust T-matrix core to Python.
//!
//! The bindings mirror the Fortran entrypoints `calctmat` / `calcampl` that
//! upstream pytmatrix calls. A `TMatrixHandle` opaque class keeps the
//! precomputed `TMatrixState` alive between the two calls (equivalent to
//! Fortran's COMMON blocks).

use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray2, PyArray4, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::amplitude::calcampl;
use crate::mie;
use crate::tmatrix::{calctmat as rs_calctmat, TMatrixConfig, TMatrixState};

/// Opaque handle wrapping `TMatrixState`. Python calls `calctmat(...)` and
/// gets back this handle plus `nmax`; subsequent `calcampl(handle, ...)`
/// reuses the state.
#[pyclass]
pub struct TMatrixHandle {
    state: TMatrixState,
    lam: f64,
}

#[pymethods]
impl TMatrixHandle {
    #[getter]
    fn nmax(&self) -> usize {
        self.state.nmax
    }

    #[getter]
    fn ngauss(&self) -> usize {
        self.state.ngauss
    }

    fn __repr__(&self) -> String {
        format!(
            "<TMatrixHandle nmax={} ngauss={}>",
            self.state.nmax, self.state.ngauss
        )
    }
}

/// Python-visible `calctmat`. Returns `(handle, nmax)`.
#[pyfunction]
#[pyo3(signature = (axi, rat, lam, mrr, mri, eps, np, ddelt, ndgs))]
pub fn calctmat(
    axi: f64,
    rat: f64,
    lam: f64,
    mrr: f64,
    mri: f64,
    eps: f64,
    np: i32,
    ddelt: f64,
    ndgs: usize,
) -> PyResult<(TMatrixHandle, usize)> {
    if axi <= 0.0 || lam <= 0.0 {
        return Err(PyValueError::new_err("axi and lam must be positive"));
    }
    if eps <= 0.0 {
        return Err(PyValueError::new_err("eps (axis ratio) must be positive"));
    }
    let cfg = TMatrixConfig {
        axi,
        rat,
        lam,
        m: Complex64::new(mrr, mri),
        eps,
        np,
        ddelt,
        ndgs,
    };
    let state = rs_calctmat(cfg);
    let nmax = state.nmax;
    Ok((TMatrixHandle { state, lam }, nmax))
}

/// Python-visible `calcampl`. Returns `(S (2x2 complex128), Z (4x4 float64))`.
#[pyfunction]
#[pyo3(signature = (handle, lam, thet0, thet, phi0, phi, alpha, beta))]
pub fn calcampl_py<'py>(
    py: Python<'py>,
    handle: &TMatrixHandle,
    lam: f64,
    thet0: f64,
    thet: f64,
    phi0: f64,
    phi: f64,
    alpha: f64,
    beta: f64,
) -> PyResult<(Bound<'py, PyArray2<Complex64>>, Bound<'py, PyArray2<f64>>)> {
    // Allow overriding lam (mirrors Fortran signature); otherwise use the one
    // cached on the handle.
    let lam_eff = if lam > 0.0 { lam } else { handle.lam };
    let (s, z) = calcampl(&handle.state, lam_eff, thet0, thet, phi0, phi, alpha, beta);
    let s_arr = ndarray::Array2::from_shape_fn((2, 2), |(i, j)| s[i][j]);
    let z_arr = ndarray::Array2::from_shape_fn((4, 4), |(i, j)| z[i][j]);
    Ok((s_arr.into_pyarray_bound(py), z_arr.into_pyarray_bound(py)))
}

/// Mie scattering efficiency — exposed for testing and convenience.
#[pyfunction]
pub fn mie_qsca(x: f64, mrr: f64, mri: f64) -> f64 {
    mie::qsca(x, Complex64::new(mrr, mri))
}

#[pyfunction]
pub fn mie_qext(x: f64, mrr: f64, mri: f64) -> f64 {
    mie::qext(x, Complex64::new(mrr, mri))
}

/// Batch tabulator for PSD integration.
///
/// For each diameter `D[i]`, builds the T-matrix with `axi = D[i]/2`,
/// `m = ms_real[i] + i*ms_imag[i]`, `eps = axis_ratios[i]`, then evaluates
/// the amplitude and phase matrices at every geometry in `geometries`.
/// The per-diameter solves are run in parallel across CPU cores via
/// rayon, with the GIL released for the duration.
///
/// Returns `(S_table, Z_table)` where
///   `S_table.shape == (num_points, num_geoms, 2, 2)` (complex128)
///   `Z_table.shape == (num_points, num_geoms, 4, 4)` (float64)
#[pyfunction]
#[pyo3(signature = (
    diameters, axis_ratios, ms_real, ms_imag, geometries,
    rat, lam, np, ddelt, ndgs
))]
#[allow(clippy::too_many_arguments)]
pub fn tabulate_scatter_table<'py>(
    py: Python<'py>,
    diameters: PyReadonlyArray1<f64>,
    axis_ratios: PyReadonlyArray1<f64>,
    ms_real: PyReadonlyArray1<f64>,
    ms_imag: PyReadonlyArray1<f64>,
    geometries: Vec<(f64, f64, f64, f64, f64, f64)>,
    rat: f64,
    lam: f64,
    np: i32,
    ddelt: f64,
    ndgs: usize,
) -> PyResult<(Bound<'py, PyArray4<Complex64>>, Bound<'py, PyArray4<f64>>)> {
    let d = diameters.as_slice()?;
    let eps = axis_ratios.as_slice()?;
    let mr = ms_real.as_slice()?;
    let mi = ms_imag.as_slice()?;
    let n = d.len();
    if eps.len() != n || mr.len() != n || mi.len() != n {
        return Err(PyValueError::new_err(
            "diameters, axis_ratios, ms_real, ms_imag must have the same length",
        ));
    }
    if geometries.is_empty() {
        return Err(PyValueError::new_err("at least one geometry required"));
    }
    if lam <= 0.0 {
        return Err(PyValueError::new_err("lam must be positive"));
    }

    // Snapshot into owned buffers so the GIL-released closure can move them.
    let d: Vec<f64> = d.to_vec();
    let eps: Vec<f64> = eps.to_vec();
    let mr: Vec<f64> = mr.to_vec();
    let mi: Vec<f64> = mi.to_vec();
    let ng = geometries.len();

    // Output buffers: row-major (n, ng, 2, 2) and (n, ng, 4, 4).
    let mut s_flat = vec![Complex64::new(0.0, 0.0); n * ng * 4];
    let mut z_flat = vec![0.0_f64; n * ng * 16];
    let s_stride = ng * 4;
    let z_stride = ng * 16;

    // Heavy compute: parallel across diameters, GIL released.
    py.allow_threads(|| {
        // Split the flat output slices into per-diameter chunks so each
        // rayon task writes to a disjoint region — zero contention.
        s_flat
            .par_chunks_mut(s_stride)
            .zip(z_flat.par_chunks_mut(z_stride))
            .enumerate()
            .for_each(|(i, (s_row, z_row))| {
                let cfg = TMatrixConfig {
                    axi: d[i] / 2.0,
                    rat,
                    lam,
                    m: Complex64::new(mr[i], mi[i]),
                    eps: eps[i],
                    np,
                    ddelt,
                    ndgs,
                };
                let state = rs_calctmat(cfg);
                for (g_idx, g) in geometries.iter().enumerate() {
                    let (s, z) =
                        calcampl(&state, lam, g.0, g.1, g.2, g.3, g.4, g.5);
                    let s_off = g_idx * 4;
                    s_row[s_off] = s[0][0];
                    s_row[s_off + 1] = s[0][1];
                    s_row[s_off + 2] = s[1][0];
                    s_row[s_off + 3] = s[1][1];
                    let z_off = g_idx * 16;
                    for a in 0..4 {
                        for b in 0..4 {
                            z_row[z_off + a * 4 + b] = z[a][b];
                        }
                    }
                }
            });
    });

    let s_arr = ndarray::Array4::from_shape_vec((n, ng, 2, 2), s_flat)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let z_arr = ndarray::Array4::from_shape_vec((n, ng, 4, 4), z_flat)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((s_arr.into_pyarray_bound(py), z_arr.into_pyarray_bound(py)))
}

/// Batch tabulator with fixed-quadrature orientation averaging.
///
/// Matches ``orientation.orient_averaged_fixed``: for each diameter a single
/// T-matrix is built, and the amplitude matrix is evaluated over a product
/// grid of ``alphas`` (uniformly sampled in [0, 360)) and ``(betas,
/// beta_weights)`` (Gauss-Legendre against the orientation PDF). Per-diameter
/// S,Z are the weighted average ``sum(w_beta * S_i) * (aw / sum(w_beta))``
/// with ``aw = 1/len(alphas)`` — matching the Python reference exactly.
///
/// Returns `(S_table, Z_table)` with the same layout as
/// `tabulate_scatter_table`.
#[pyfunction]
#[pyo3(signature = (
    diameters, axis_ratios, ms_real, ms_imag, geometries,
    alphas, betas, beta_weights,
    rat, lam, np, ddelt, ndgs
))]
#[allow(clippy::too_many_arguments)]
pub fn tabulate_scatter_table_orient_avg<'py>(
    py: Python<'py>,
    diameters: PyReadonlyArray1<f64>,
    axis_ratios: PyReadonlyArray1<f64>,
    ms_real: PyReadonlyArray1<f64>,
    ms_imag: PyReadonlyArray1<f64>,
    geometries: Vec<(f64, f64, f64, f64, f64, f64)>,
    alphas: PyReadonlyArray1<f64>,
    betas: PyReadonlyArray1<f64>,
    beta_weights: PyReadonlyArray1<f64>,
    rat: f64,
    lam: f64,
    np: i32,
    ddelt: f64,
    ndgs: usize,
) -> PyResult<(Bound<'py, PyArray4<Complex64>>, Bound<'py, PyArray4<f64>>)> {
    let d = diameters.as_slice()?;
    let eps = axis_ratios.as_slice()?;
    let mr = ms_real.as_slice()?;
    let mi = ms_imag.as_slice()?;
    let n = d.len();
    if eps.len() != n || mr.len() != n || mi.len() != n {
        return Err(PyValueError::new_err(
            "diameters, axis_ratios, ms_real, ms_imag must have the same length",
        ));
    }
    let alphas = alphas.as_slice()?.to_vec();
    let betas = betas.as_slice()?.to_vec();
    let beta_w = beta_weights.as_slice()?.to_vec();
    if betas.len() != beta_w.len() {
        return Err(PyValueError::new_err(
            "betas and beta_weights must have the same length",
        ));
    }
    if geometries.is_empty() || alphas.is_empty() || betas.is_empty() {
        return Err(PyValueError::new_err(
            "geometries, alphas, betas must all be non-empty",
        ));
    }
    if lam <= 0.0 {
        return Err(PyValueError::new_err("lam must be positive"));
    }

    let d: Vec<f64> = d.to_vec();
    let eps: Vec<f64> = eps.to_vec();
    let mr: Vec<f64> = mr.to_vec();
    let mi: Vec<f64> = mi.to_vec();
    let ng = geometries.len();
    let aw_over_sw = (1.0 / alphas.len() as f64) / beta_w.iter().sum::<f64>();

    let mut s_flat = vec![Complex64::new(0.0, 0.0); n * ng * 4];
    let mut z_flat = vec![0.0_f64; n * ng * 16];
    let s_stride = ng * 4;
    let z_stride = ng * 16;

    py.allow_threads(|| {
        s_flat
            .par_chunks_mut(s_stride)
            .zip(z_flat.par_chunks_mut(z_stride))
            .enumerate()
            .for_each(|(i, (s_row, z_row))| {
                let cfg = TMatrixConfig {
                    axi: d[i] / 2.0,
                    rat,
                    lam,
                    m: Complex64::new(mr[i], mi[i]),
                    eps: eps[i],
                    np,
                    ddelt,
                    ndgs,
                };
                let state = rs_calctmat(cfg);
                for (g_idx, g) in geometries.iter().enumerate() {
                    // Accumulate weighted sum of S and Z over the (alpha, beta) grid.
                    let mut s_acc = [[Complex64::new(0.0, 0.0); 2]; 2];
                    let mut z_acc = [[0.0_f64; 4]; 4];
                    for &alpha in &alphas {
                        for (&beta, &w) in betas.iter().zip(beta_w.iter()) {
                            let (s, z) = calcampl(
                                &state, lam, g.0, g.1, g.2, g.3, alpha, beta,
                            );
                            for a in 0..2 {
                                for b in 0..2 {
                                    s_acc[a][b] += Complex64::new(w, 0.0) * s[a][b];
                                }
                            }
                            for a in 0..4 {
                                for b in 0..4 {
                                    z_acc[a][b] += w * z[a][b];
                                }
                            }
                        }
                    }
                    let s_off = g_idx * 4;
                    s_row[s_off] = s_acc[0][0] * aw_over_sw;
                    s_row[s_off + 1] = s_acc[0][1] * aw_over_sw;
                    s_row[s_off + 2] = s_acc[1][0] * aw_over_sw;
                    s_row[s_off + 3] = s_acc[1][1] * aw_over_sw;
                    let z_off = g_idx * 16;
                    for a in 0..4 {
                        for b in 0..4 {
                            z_row[z_off + a * 4 + b] = z_acc[a][b] * aw_over_sw;
                        }
                    }
                }
            });
    });

    let s_arr = ndarray::Array4::from_shape_vec((n, ng, 2, 2), s_flat)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let z_arr = ndarray::Array4::from_shape_vec((n, ng, 4, 4), z_flat)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((s_arr.into_pyarray_bound(py), z_arr.into_pyarray_bound(py)))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TMatrixHandle>()?;
    m.add_function(wrap_pyfunction!(calctmat, m)?)?;
    m.add_function(wrap_pyfunction!(calcampl_py, m)?)?;
    m.add_function(wrap_pyfunction!(tabulate_scatter_table, m)?)?;
    m.add_function(wrap_pyfunction!(tabulate_scatter_table_orient_avg, m)?)?;
    m.add_function(wrap_pyfunction!(mie_qsca, m)?)?;
    m.add_function(wrap_pyfunction!(mie_qext, m)?)?;
    // Shape constants.
    m.add("SHAPE_SPHEROID", -1i32)?;
    m.add("SHAPE_CYLINDER", -2i32)?;
    m.add("SHAPE_CHEBYSHEV", 1i32)?;
    m.add("RADIUS_EQUAL_VOLUME", 1.0_f64)?;
    m.add("RADIUS_EQUAL_AREA", 0.0_f64)?;
    m.add("RADIUS_MAXIMUM", 2.0_f64)?;
    Ok(())
}
