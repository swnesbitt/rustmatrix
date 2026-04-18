"""Python-side `Scatterer` class.

Mirrors the API of `pytmatrix.tmatrix.Scatterer` so downstream code can
use this module as a drop-in replacement. Orientation averaging is
provided by :mod:`rupytmatrix.orientation` (pure Python, calls the Rust
single-orientation evaluator in a loop).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from . import _core
from . import orientation as _orientation
from .quadrature import get_points_and_weights as _qpw


@dataclass
class Scatterer:
    """T-Matrix scattering from nonspherical particles (Rust backend).

    API-compatible with :class:`pytmatrix.tmatrix.Scatterer`. See pytmatrix's
    documentation for the meaning of each attribute.
    """

    radius: float = 1.0
    radius_type: float = 1.0  # RADIUS_EQUAL_VOLUME
    wavelength: float = 1.0
    m: complex = 2.0 + 0.0j
    axis_ratio: float = 1.0
    shape: int = -1  # SHAPE_SPHEROID
    ddelt: float = 1e-3
    ndgs: int = 2
    alpha: float = 0.0
    beta: float = 0.0
    thet0: float = 90.0
    thet: float = 90.0
    phi0: float = 0.0
    phi: float = 180.0
    Kw_sqr: float = 0.93
    suppress_warning: bool = False

    # Constants for convenience (mirror pytmatrix).
    RADIUS_EQUAL_VOLUME = _core.RADIUS_EQUAL_VOLUME
    RADIUS_EQUAL_AREA = _core.RADIUS_EQUAL_AREA
    RADIUS_MAXIMUM = _core.RADIUS_MAXIMUM
    SHAPE_SPHEROID = _core.SHAPE_SPHEROID
    SHAPE_CYLINDER = _core.SHAPE_CYLINDER
    SHAPE_CHEBYSHEV = _core.SHAPE_CHEBYSHEV

    # Internal state (not part of the public API).
    _handle: Optional[object] = field(default=None, repr=False)
    _tm_signature: Tuple = field(default_factory=tuple, repr=False)
    _scatter_signature: Tuple = field(default_factory=tuple, repr=False)
    _orient_signature: Tuple = field(default_factory=tuple, repr=False)
    _S_single: Optional[np.ndarray] = field(default=None, repr=False)
    _Z_single: Optional[np.ndarray] = field(default=None, repr=False)
    _S: Optional[np.ndarray] = field(default=None, repr=False)
    _Z: Optional[np.ndarray] = field(default=None, repr=False)

    def __init__(self, **kwargs):
        defaults = {
            "radius": 1.0,
            "radius_type": _core.RADIUS_EQUAL_VOLUME,
            "wavelength": 1.0,
            "m": complex(2, 0),
            "axis_ratio": 1.0,
            "shape": _core.SHAPE_SPHEROID,
            "ddelt": 1e-3,
            "ndgs": 2,
            "alpha": 0.0,
            "beta": 0.0,
            "thet0": 90.0,
            "thet": 90.0,
            "phi0": 0.0,
            "phi": 180.0,
            "Kw_sqr": 0.93,
            "suppress_warning": False,
            # Orientation averaging defaults (match pytmatrix).
            "orient": _orientation.orient_single,
            "or_pdf": _orientation.gaussian_pdf(),
            "n_alpha": 5,
            "n_beta": 10,
        }
        deprecated = {
            "axi": "radius",
            "lam": "wavelength",
            "eps": "axis_ratio",
            "rat": "radius_type",
            "np": "shape",
            "scatter": "orient",
        }
        for key, new in deprecated.items():
            if key in kwargs:
                if not kwargs.get("suppress_warning", False):
                    warnings.warn(
                        f"'{key}' is deprecated; use '{new}'.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                defaults[new] = kwargs.pop(key)
        for k, v in kwargs.items():
            defaults[k] = v
        for k, v in defaults.items():
            object.__setattr__(self, k, v)
        object.__setattr__(self, "_handle", None)
        object.__setattr__(self, "_tm_signature", ())
        object.__setattr__(self, "_scatter_signature", ())
        object.__setattr__(self, "_orient_signature", ())
        object.__setattr__(self, "_S_single", None)
        object.__setattr__(self, "_Z_single", None)
        object.__setattr__(self, "_S", None)
        object.__setattr__(self, "_Z", None)

    # ---------- convenience helpers ----------

    def set_geometry(self, geom):
        (self.thet0, self.thet, self.phi0, self.phi, self.alpha, self.beta) = geom

    def get_geometry(self):
        return (self.thet0, self.thet, self.phi0, self.phi, self.alpha, self.beta)

    def equal_volume_from_maximum(self):
        if self.shape == _core.SHAPE_SPHEROID:
            if self.axis_ratio > 1.0:
                return self.radius / self.axis_ratio ** (1.0 / 3.0)
            return self.radius * self.axis_ratio ** (2.0 / 3.0)
        if self.shape == _core.SHAPE_CYLINDER:
            if self.axis_ratio > 1.0:
                return self.radius * (1.5 / self.axis_ratio) ** (1.0 / 3.0)
            return self.radius * (1.5 * self.axis_ratio ** 2) ** (1.0 / 3.0)
        raise AttributeError("Unsupported shape for maximum radius.")

    # ---------- core computation ----------

    def _init_tmatrix(self):
        if self.radius_type == _core.RADIUS_MAXIMUM:
            radius_type = _core.RADIUS_EQUAL_VOLUME
            radius = self.equal_volume_from_maximum()
        else:
            radius_type = self.radius_type
            radius = self.radius
        handle, nmax = _core.calctmat(
            radius,
            radius_type,
            self.wavelength,
            self.m.real,
            self.m.imag,
            self.axis_ratio,
            int(self.shape),
            self.ddelt,
            int(self.ndgs),
        )
        object.__setattr__(self, "_handle", handle)
        object.__setattr__(self, "nmax", nmax)
        object.__setattr__(
            self,
            "_tm_signature",
            (
                self.radius,
                self.radius_type,
                self.wavelength,
                self.m,
                self.axis_ratio,
                self.shape,
                self.ddelt,
                self.ndgs,
            ),
        )

    def _init_orient(self):
        """Build (beta_p, beta_w) quadrature against ``or_pdf`` if needed."""
        if self.orient is _orientation.orient_averaged_fixed:
            beta_p, beta_w = _qpw(self.or_pdf, 0, 180, self.n_beta)
            object.__setattr__(self, "beta_p", beta_p)
            object.__setattr__(self, "beta_w", beta_w)
        object.__setattr__(
            self,
            "_orient_signature",
            (self.orient, self.or_pdf, self.n_alpha, self.n_beta),
        )

    def get_SZ_single(self, alpha=None, beta=None):
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            beta = self.beta
        tm_outdated = self._tm_signature != (
            self.radius,
            self.radius_type,
            self.wavelength,
            self.m,
            self.axis_ratio,
            self.shape,
            self.ddelt,
            self.ndgs,
        )
        if tm_outdated or self._handle is None:
            self._init_tmatrix()
        sig = (self.thet0, self.thet, self.phi0, self.phi, alpha, beta)
        scatter_outdated = self._scatter_signature != sig
        if tm_outdated or scatter_outdated:
            S, Z = _core.calcampl_py(
                self._handle,
                self.wavelength,
                self.thet0,
                self.thet,
                self.phi0,
                self.phi,
                alpha,
                beta,
            )
            object.__setattr__(self, "_S_single", np.asarray(S))
            object.__setattr__(self, "_Z_single", np.asarray(Z))
            object.__setattr__(self, "_scatter_signature", sig)
        return self._S_single, self._Z_single

    def get_SZ_orient(self):
        """S and Z using ``self.orient`` (orientation-averaging dispatcher)."""
        orient_outdated = self._orient_signature != (
            self.orient,
            self.or_pdf,
            self.n_alpha,
            self.n_beta,
        )
        if orient_outdated:
            self._init_orient()
        S, Z = self.orient(self)
        object.__setattr__(self, "_S", np.asarray(S))
        object.__setattr__(self, "_Z", np.asarray(Z))
        return self._S, self._Z

    def get_SZ(self):
        return self.get_SZ_orient()

    def get_S(self):
        return self.get_SZ()[0]

    def get_Z(self):
        return self.get_SZ()[1]


# pytmatrix compatibility alias
class TMatrix(Scatterer):
    def __init__(self, **kwargs):
        if not kwargs.get("suppress_warning", False):
            warnings.warn(
                "'TMatrix' is deprecated; use 'Scatterer'.",
                DeprecationWarning,
                stacklevel=2,
            )
        super().__init__(**kwargs)
