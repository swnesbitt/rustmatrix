"""Type stubs for the Rust extension module."""

from __future__ import annotations

import numpy as np

RADIUS_EQUAL_VOLUME: float
RADIUS_EQUAL_AREA: float
RADIUS_MAXIMUM: float
SHAPE_SPHEROID: int
SHAPE_CYLINDER: int
SHAPE_CHEBYSHEV: int

class TMatrixHandle:
    nmax: int
    ngauss: int
    def __repr__(self) -> str: ...

def calctmat(
    axi: float,
    rat: float,
    lam: float,
    mrr: float,
    mri: float,
    eps: float,
    np: int,
    ddelt: float,
    ndgs: int,
) -> tuple[TMatrixHandle, int]: ...

def calcampl_py(
    handle: TMatrixHandle,
    lam: float,
    thet0: float,
    thet: float,
    phi0: float,
    phi: float,
    alpha: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]: ...

def mie_qsca(x: float, mrr: float, mri: float) -> float: ...
def mie_qext(x: float, mrr: float, mri: float) -> float: ...

def tabulate_scatter_table(
    diameters: np.ndarray,
    axis_ratios: np.ndarray,
    ms_real: np.ndarray,
    ms_imag: np.ndarray,
    geometries: list[tuple[float, float, float, float, float, float]],
    rat: float,
    lam: float,
    np: int,
    ddelt: float,
    ndgs: int,
) -> tuple[np.ndarray, np.ndarray]: ...
