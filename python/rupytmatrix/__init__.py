"""rupytmatrix — Rust-backed T-matrix scattering.

Drop-in replacement for the numerical core of
`pytmatrix <https://github.com/jleinonen/pytmatrix>`_. The public
``Scatterer`` class mirrors pytmatrix's own so existing code that only
touches ``Scatterer``, ``get_S``, ``get_Z``, etc. can be adapted by
swapping the import.
"""

from __future__ import annotations

from ._core import (  # noqa: F401
    RADIUS_EQUAL_AREA,
    RADIUS_EQUAL_VOLUME,
    RADIUS_MAXIMUM,
    SHAPE_CHEBYSHEV,
    SHAPE_CYLINDER,
    SHAPE_SPHEROID,
    calctmat,
    calcampl_py as _calcampl,
    mie_qext,
    mie_qsca,
)
from . import orientation, quadrature  # noqa: F401
from .scatterer import Scatterer, TMatrix  # noqa: F401

__version__ = "0.1.0"
__all__ = [
    "Scatterer",
    "TMatrix",
    "calctmat",
    "mie_qext",
    "mie_qsca",
    "orientation",
    "quadrature",
    "RADIUS_EQUAL_VOLUME",
    "RADIUS_EQUAL_AREA",
    "RADIUS_MAXIMUM",
    "SHAPE_SPHEROID",
    "SHAPE_CYLINDER",
    "SHAPE_CHEBYSHEV",
]
