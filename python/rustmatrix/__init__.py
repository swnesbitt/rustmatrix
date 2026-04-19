"""rustmatrix — Rust-backed T-matrix scattering for nonspherical particles.

Drop-in replacement for the numerical core of
`pytmatrix <https://github.com/jleinonen/pytmatrix>`_. The public
:class:`Scatterer` class mirrors :class:`pytmatrix.tmatrix.Scatterer` exactly,
so existing code that touches ``Scatterer``, ``get_S``, ``get_Z``,
``PSDIntegrator``, or any of the radar/scatter helpers can be ported by
changing the imports.

Top-level symbols
-----------------
Scatterer, TMatrix
    Core scatterer class (``TMatrix`` is a deprecated alias).
calctmat
    Low-level Rust entry point that builds a T-matrix and returns an opaque
    handle. Prefer :class:`Scatterer` for regular use.
mie_qsca, mie_qext
    Closed-form Mie efficiencies for a homogeneous sphere. Used as a
    sanity check in the ``axis_ratio = 1`` limit.
SHAPE_SPHEROID, SHAPE_CYLINDER, SHAPE_CHEBYSHEV
    Particle-shape codes (integer flags, identical to pytmatrix).
RADIUS_EQUAL_VOLUME, RADIUS_EQUAL_AREA, RADIUS_MAXIMUM
    Interpretations of the ``radius`` attribute.

Submodules
----------
hd_mix
    :class:`HydroMix` and :class:`MixtureComponent` for combining
    multiple hydrometeor species into a single Scatterer-shaped object.
orientation
    Orientation-averaging strategies and PDFs.
psd
    Particle-size-distribution classes and :class:`PSDIntegrator`.
radar
    Polarimetric radar observables (Z_dr, K_dp, ρ_hv, …).
scatter
    Angular-integrated scattering helpers (σ_sca, σ_ext, ω, g, LDR).
spectra
    Doppler and polarimetric spectra — :class:`SpectralIntegrator` with
    fall-speed presets, turbulence models (including Zeng 2023 particle
    inertia), and beam broadening.
refractive
    Refractive-index helpers (Maxwell-Garnett, Bruggeman, tabulated
    water/ice indices across S–W band).
quadrature
    Gautschi quadrature against arbitrary weighting functions (used by
    :func:`orientation.orient_averaged_fixed`).
tmatrix_aux
    Radar-band wavelength / dielectric presets, canned geometries, and
    drop-shape relationships (Thurai, Pruppacher-Beard, Beard-Chuang).

Examples
--------
>>> from rustmatrix import Scatterer
>>> s = Scatterer(radius=1.0, wavelength=33.3, m=complex(7.99, 2.21),
...               axis_ratio=1.0, ddelt=1e-4, ndgs=2)
>>> s.set_geometry((90.0, 90.0, 0.0, 180.0, 0.0, 0.0))
>>> S, Z = s.get_SZ()

References
----------
Mishchenko, M. I., & Travis, L. D. (1998). Capabilities and limitations of a
current FORTRAN implementation of the T-matrix method for randomly oriented,
rotationally symmetric scatterers. *JQSRT*, 60(3), 309–324.

Leinonen, J. (2014). High-level interface to T-matrix scattering calculations:
architecture, capabilities and limitations. *Optics Express*, 22, 1655.
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
from . import (  # noqa: F401
    hd_mix,
    orientation,
    psd,
    quadrature,
    radar,
    refractive,
    scatter,
    spectra,
    tmatrix_aux,
)
from .hd_mix import HydroMix, MixtureComponent  # noqa: F401
from .scatterer import Scatterer, TMatrix  # noqa: F401
from .spectra import SpectralIntegrator, SpectralResult  # noqa: F401

__version__ = "0.1.0"
__all__ = [
    "Scatterer",
    "TMatrix",
    "HydroMix",
    "MixtureComponent",
    "SpectralIntegrator",
    "SpectralResult",
    "calctmat",
    "mie_qext",
    "mie_qsca",
    "hd_mix",
    "orientation",
    "psd",
    "quadrature",
    "radar",
    "refractive",
    "scatter",
    "spectra",
    "tmatrix_aux",
    "RADIUS_EQUAL_VOLUME",
    "RADIUS_EQUAL_AREA",
    "RADIUS_MAXIMUM",
    "SHAPE_SPHEROID",
    "SHAPE_CYLINDER",
    "SHAPE_CHEBYSHEV",
]
