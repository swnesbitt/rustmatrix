"""Thread-parallelism tests.

`calctmat` / `calcampl_py` release the GIL (detach from the Python
runtime) for their heavy compute, so Python threads can build T-matrices
and evaluate amplitudes truly in parallel — on both GIL-enabled and
free-threaded (3.13t+) CPython. These tests check correctness under
concurrency; wall-clock scaling is exercised in `benches/` rather than
asserted here, since CI runners have unpredictable load.
"""

from __future__ import annotations

import sys
import sysconfig
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from rustmatrix import _core

# A spread of drop sizes: enough work per call for threads to overlap.
DIAMETERS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
LAM = 8.6  # C band, mm
M = complex(7.718, 2.473)


def _build(d):
    return _core.calctmat(d / 2.0, 1.0, LAM, M.real, M.imag, 1.0 / 0.6, -1, 1e-3, 2)


def _ampl(handle, geom):
    return _core.calcampl_py(handle, LAM, *geom)


def test_parallel_calctmat_matches_serial():
    """T-matrices built concurrently in threads equal serially built ones."""
    serial = [_build(d) for d in DIAMETERS]
    with ThreadPoolExecutor(max_workers=4) as tpe:
        parallel = list(tpe.map(_build, DIAMETERS))

    geom = (90.0, 90.0, 0.0, 180.0, 0.0, 0.0)
    for (hs, nmax_s), (hp, nmax_p) in zip(serial, parallel):
        assert nmax_s == nmax_p
        s_s, z_s = _ampl(hs, geom)
        s_p, z_p = _ampl(hp, geom)
        np.testing.assert_array_equal(s_s, s_p)
        np.testing.assert_array_equal(z_s, z_p)


def test_concurrent_calcampl_shared_handle():
    """Many threads may evaluate against the SAME (frozen) handle at once."""
    handle, _ = _build(2.0)
    geoms = [
        (90.0, 90.0, 0.0, 180.0, 0.0, 0.0),
        (90.0, 90.0, 0.0, 0.0, 0.0, 0.0),
        (60.0, 120.0, 0.0, 180.0, 30.0, 10.0),
        (30.0, 150.0, 45.0, 225.0, 0.0, 20.0),
    ] * 8

    serial = [_ampl(handle, g) for g in geoms]
    with ThreadPoolExecutor(max_workers=8) as tpe:
        parallel = list(tpe.map(lambda g: _ampl(handle, g), geoms))

    for (s_s, z_s), (s_p, z_p) in zip(serial, parallel):
        np.testing.assert_array_equal(s_s, s_p)
        np.testing.assert_array_equal(z_s, z_p)


@pytest.mark.skipif(
    not sysconfig.get_config_var("Py_GIL_DISABLED"),
    reason="only meaningful on free-threaded (t) builds",
)
def test_import_keeps_gil_disabled():
    """The module declares gil_used=false, so importing it must not
    re-enable the GIL on a free-threaded interpreter."""
    assert not sys._is_gil_enabled()
