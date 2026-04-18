"""Speed comparison: rupytmatrix (Rust) vs pytmatrix (Fortran).

Run in an env where both are importable:
    conda run -n rupy-parity python benches/bench_vs_pytmatrix.py
"""

from __future__ import annotations

import time

import numpy as np
import pytmatrix.tmatrix as py_tm
from pytmatrix import orientation as py_orient
from pytmatrix import psd as py_psd

import rupytmatrix
from rupytmatrix import Scatterer, orientation as rs_orient, psd as rs_psd
from rupytmatrix.tmatrix_aux import geom_horiz_back


def _time(fn, repeats):
    # Discard one warmup call so cold caches don't contaminate the timing.
    fn()
    t = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t) / repeats


def bench_single_orientation():
    """Cold T-matrix build + amplitude matrix, fresh Scatterer each call."""
    geom = geom_horiz_back
    kwargs = dict(radius=1.0, wavelength=6.283185307, axis_ratio=1.5,
                  m=complex(1.5, 0.01), ddelt=1e-4, ndgs=2)

    def py_fn():
        s = py_tm.Scatterer(**kwargs)
        s.set_geometry(geom)
        s.get_SZ()

    def rs_fn():
        s = Scatterer(**kwargs)
        s.set_geometry(geom)
        s.get_SZ()

    return _time(py_fn, 20), _time(rs_fn, 20)


def bench_cached_evaluate():
    """Warm cache — just re-evaluate S/Z at a new orientation."""
    geom = geom_horiz_back
    kwargs = dict(radius=1.0, wavelength=6.283185307, axis_ratio=1.5,
                  m=complex(1.5, 0.01), ddelt=1e-4, ndgs=2)

    py = py_tm.Scatterer(**kwargs)
    py.set_geometry(geom)
    py.get_SZ()  # prime

    rs = Scatterer(**kwargs)
    rs.set_geometry(geom)
    rs.get_SZ()

    def py_fn():
        # Vary alpha so scatter cache misses but T-matrix cache stays warm.
        py.alpha = (py.alpha + 7.0) % 360.0
        py.get_SZ()

    def rs_fn():
        rs.alpha = (rs.alpha + 7.0) % 360.0
        rs.get_SZ()

    return _time(py_fn, 200), _time(rs_fn, 200)


def bench_orient_averaged_fixed():
    """Gaussian-PDF orientation averaging, 4 alpha × 8 beta = 32 orientations."""
    kwargs = dict(radius=1.0, wavelength=6.283185307, axis_ratio=2.0,
                  m=complex(1.5, 0.01), ddelt=1e-4, ndgs=2)

    def py_fn():
        s = py_tm.Scatterer(**kwargs)
        s.set_geometry(geom_horiz_back)
        s.or_pdf = py_orient.gaussian_pdf(std=20.0, mean=90.0)
        s.orient = py_orient.orient_averaged_fixed
        s.n_alpha = 4
        s.n_beta = 8
        s.get_SZ()

    def rs_fn():
        s = Scatterer(**kwargs)
        s.set_geometry(geom_horiz_back)
        s.or_pdf = rs_orient.gaussian_pdf(std=20.0, mean=90.0)
        s.orient = rs_orient.orient_averaged_fixed
        s.n_alpha = 4
        s.n_beta = 8
        s.get_SZ()

    return _time(py_fn, 5), _time(rs_fn, 5)


def bench_psd_tabulate(num_points):
    """Tabulate S,Z at `num_points` diameters (dominant cost of a PSD run)."""
    kwargs = dict(wavelength=6.5, m=complex(1.5, 0.5), axis_ratio=1.0,
                  ddelt=1e-4, ndgs=2)

    def py_fn():
        s = py_tm.Scatterer(**kwargs)
        s.set_geometry(geom_horiz_back)
        s.psd_integrator = py_psd.PSDIntegrator()
        s.psd_integrator.num_points = num_points
        s.psd_integrator.D_max = 10.0
        s.psd = py_psd.GammaPSD(D0=1.0, Nw=1e3, mu=4)
        s.psd_integrator.init_scatter_table(s)

    def rs_fn():
        s = Scatterer(**kwargs)
        s.set_geometry(geom_horiz_back)
        s.psd_integrator = rs_psd.PSDIntegrator()
        s.psd_integrator.num_points = num_points
        s.psd_integrator.D_max = 10.0
        s.psd = rs_psd.GammaPSD(D0=1.0, Nw=1e3, mu=4)
        s.psd_integrator.init_scatter_table(s)

    return _time(py_fn, 3), _time(rs_fn, 3)


def bench_tmatrix_only():
    """Just the T-matrix computation (CALCTMAT), no amplitude evaluation."""
    radius, wl = 1.5, 6.283185307
    m = complex(1.5, 0.01)

    # pytmatrix wraps CALCTMAT inside Scatterer._init_tmatrix; easiest apples-
    # to-apples is to build a Scatterer and never evaluate SZ.
    def py_fn():
        s = py_tm.Scatterer(radius=radius, wavelength=wl, axis_ratio=1.5,
                            m=m, ddelt=1e-4, ndgs=2)
        s._init_tmatrix()

    def rs_fn():
        rupytmatrix.calctmat(radius, 1.0, wl, m.real, m.imag, 1.5, -1, 1e-4, 2)

    return _time(py_fn, 20), _time(rs_fn, 20)


def main():
    cases = [
        ("calctmat only (spheroid ax=1.5)", bench_tmatrix_only),
        ("single orient, cold (fresh Scatterer)", bench_single_orientation),
        ("cached re-eval (warm T-matrix)", bench_cached_evaluate),
        ("orient-averaged fixed (4×8 = 32)", bench_orient_averaged_fixed),
        ("PSD init_scatter_table, 32 points", lambda: bench_psd_tabulate(32)),
        ("PSD init_scatter_table, 64 points", lambda: bench_psd_tabulate(64)),
    ]

    print(f"{'case':<44} {'pytmatrix':>12} {'rupytmatrix':>14} {'speedup':>10}")
    print("-" * 82)
    for name, fn in cases:
        t_py, t_rs = fn()
        ratio = t_py / t_rs if t_rs > 0 else float("inf")
        arrow = "faster" if ratio >= 1 else "slower"
        print(f"{name:<44} {t_py*1000:>10.2f}ms {t_rs*1000:>12.2f}ms "
              f"{ratio:>7.2f}× {arrow}")


if __name__ == "__main__":
    main()
