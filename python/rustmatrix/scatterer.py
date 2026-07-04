"""Python-side :class:`Scatterer` class.

Mirrors the API of :class:`pytmatrix.tmatrix.Scatterer` so downstream code
can use this module as a drop-in replacement. Orientation averaging is
provided by :mod:`rustmatrix.orientation` (pure Python; calls into the
Rust single-orientation evaluator in a loop). PSD integration is routed
through :mod:`rustmatrix.psd`, whose fast paths run inside Rust with the
GIL released and parallelise across diameters via rayon.

All Rust entrypoints — including the single-particle T-matrix build and
amplitude evaluation used by this class — release the GIL for their
heavy compute, so independent ``Scatterer`` instances can be driven from
Python threads (``concurrent.futures.ThreadPoolExecutor``, dask's
threaded scheduler) with near-linear scaling. The cached ``_handle`` is
immutable and safe to share across threads. Free-threaded CPython
(3.13t+) is fully supported.

Notes
-----
The Rust extension caches the built T-matrix on the ``_handle`` attribute
and reuses it whenever the signature ``(radius, radius_type, wavelength,
m, axis_ratio, shape, ddelt, ndgs)`` is unchanged. Switching orientation
or scattering geometry costs only an amplitude-matrix rotation.
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

    API-compatible with :class:`pytmatrix.tmatrix.Scatterer`. Construct one
    with the size / material / geometry attributes below, then call
    :meth:`get_SZ` (or :meth:`get_S`, :meth:`get_Z`) to obtain the amplitude
    and phase matrices.

    Attributes
    ----------
    radius : float
        Particle "equivalent" radius in mm. Interpreted according to
        :attr:`radius_type`. Default: 1.0.
    radius_type : float
        One of :data:`RADIUS_EQUAL_VOLUME` (default), :data:`RADIUS_EQUAL_AREA`,
        or :data:`RADIUS_MAXIMUM`. Controls how ``radius`` is converted to
        the Mishchenko-code equal-volume radius the solver expects.
    wavelength : float
        Incident-radiation wavelength in mm. Use the ``wl_*`` presets in
        :mod:`rustmatrix.tmatrix_aux` for standard radar bands.
    m : complex
        Refractive index of the particle material. Use
        :mod:`rustmatrix.refractive` for tabulated water/ice values.
    axis_ratio : float
        Horizontal over vertical axis ratio. ``axis_ratio > 1`` is oblate
        (flattened raindrop); ``< 1`` is prolate (columnar ice); ``= 1`` is
        a sphere.
    shape : int
        Particle shape code. :data:`SHAPE_SPHEROID` (-1, default),
        :data:`SHAPE_CYLINDER` (-2), or :data:`SHAPE_CHEBYSHEV` (1).
    ddelt : float
        Convergence tolerance for the T-matrix solver. ``1e-3`` is usually
        fine; tighten to ``1e-4`` for high-accuracy work. Default: 1e-3.
    ndgs : int
        Quadrature density factor. Increase for elongated particles or
        large size parameters if the solver fails to converge. Default: 2.
    alpha, beta : float
        Particle Euler angles in degrees. For a single-orientation
        evaluation both default to 0.
    thet0, thet : float
        Incident and scattering zenith angles in degrees (0 = north pole,
        180 = south). Defaults: both 90 (horizontal propagation).
    phi0, phi : float
        Incident and scattering azimuth angles in degrees. Defaults:
        ``phi0 = 0``, ``phi = 180`` (backscatter).
    Kw_sqr : float
        |K_w|² dielectric factor used by :func:`radar.refl`. Standard
        radar-band values are in :data:`tmatrix_aux.K_w_sqr`. Default: 0.93.
    orient : callable
        Orientation-averaging strategy from :mod:`rustmatrix.orientation`
        (``orient_single``, ``orient_averaged_fixed``, or
        ``orient_averaged_adaptive``). Default: ``orient_single``.
    or_pdf : callable
        Orientation PDF returning weight given β in degrees. See
        :func:`orientation.gaussian_pdf`, :func:`orientation.uniform_pdf`.
    n_alpha, n_beta : int
        Number of α / β samples for ``orient_averaged_fixed``. Defaults:
        5 and 10.
    psd_integrator : PSDIntegrator, optional
        When set, :meth:`get_SZ` integrates S and Z against
        :attr:`psd`. See :class:`rustmatrix.psd.PSDIntegrator`.
    psd : PSD, optional
        Particle-size distribution instance (e.g.
        :class:`rustmatrix.psd.GammaPSD`).
    suppress_warning : bool
        Silence the ``DeprecationWarning`` emitted when legacy pytmatrix
        kwargs (``axi``, ``lam``, ``eps``, ``rat``, ``np``, ``scatter``)
        are used.

    Examples
    --------
    >>> from rustmatrix import Scatterer
    >>> s = Scatterer(radius=1.0, wavelength=33.3, m=complex(7.99, 2.21),
    ...               axis_ratio=1.5, ddelt=1e-4, ndgs=2)
    >>> s.set_geometry((90, 90, 0, 180, 0, 0))  # horizontal backscatter
    >>> S, Z = s.get_SZ()
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
    _psd_signature: Tuple = field(default_factory=tuple, repr=False)
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
            # PSD integration (disabled by default).
            "psd_integrator": None,
            "psd": None,
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
        object.__setattr__(self, "_psd_signature", ())
        object.__setattr__(self, "_S_single", None)
        object.__setattr__(self, "_Z_single", None)
        object.__setattr__(self, "_S", None)
        object.__setattr__(self, "_Z", None)

    # ---------- convenience helpers ----------

    def set_geometry(self, geom):
        """Assign ``(thet0, thet, phi0, phi, alpha, beta)`` in one call.

        ``geom`` is a 6-tuple of degrees — use the ``geom_*`` presets from
        :mod:`rustmatrix.tmatrix_aux` for the common radar cases.
        """
        (self.thet0, self.thet, self.phi0, self.phi, self.alpha, self.beta) = geom

    def get_geometry(self):
        """Return the current ``(thet0, thet, phi0, phi, alpha, beta)`` tuple."""
        return (self.thet0, self.thet, self.phi0, self.phi, self.alpha, self.beta)

    def equal_volume_from_maximum(self):
        """Equal-volume-sphere radius given ``self.radius`` as a maximum radius.

        Only defined for :data:`SHAPE_SPHEROID` and :data:`SHAPE_CYLINDER`.
        Used internally when :attr:`radius_type` is
        :data:`RADIUS_MAXIMUM`.
        """
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
        """Amplitude and phase matrices at a single Euler orientation.

        Parameters
        ----------
        alpha, beta : float, optional
            Euler angles in degrees. Default to ``self.alpha``, ``self.beta``.

        Returns
        -------
        S : ndarray (2, 2) complex
            Amplitude scattering matrix.
        Z : ndarray (4, 4) float
            Phase (Stokes) matrix.

        Notes
        -----
        Cached on the handle; re-building the T-matrix only happens when the
        size/material signature changes. Advancing only ``(alpha, beta)`` or
        the scattering geometry reuses the existing T-matrix.
        """
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
        """S and Z using ``self.orient`` (orientation-averaging dispatcher).

        Dispatches to the callable in :attr:`orient` —
        :func:`orientation.orient_single` by default.
        """
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
        """Amplitude + phase matrices, PSD-integrated if configured.

        Returns
        -------
        S : ndarray (2, 2) complex
        Z : ndarray (4, 4) float

        Notes
        -----
        If :attr:`psd_integrator` is ``None`` this is equivalent to
        :meth:`get_SZ_orient`. Otherwise it returns the N(D)-weighted
        average over the scatter table built by
        :meth:`PSDIntegrator.init_scatter_table`.
        """
        if self.psd_integrator is None:
            return self.get_SZ_orient()

        scatter_sig = (
            self.thet0,
            self.thet,
            self.phi0,
            self.phi,
            self.alpha,
            self.beta,
            self.orient,
        )
        psd_sig = (self.psd,)
        if self._scatter_signature != scatter_sig or self._psd_signature != psd_sig:
            S, Z = self.psd_integrator(self.psd, self.get_geometry())
            object.__setattr__(self, "_S", np.asarray(S))
            object.__setattr__(self, "_Z", np.asarray(Z))
            object.__setattr__(self, "_scatter_signature", scatter_sig)
            object.__setattr__(self, "_psd_signature", psd_sig)
        return self._S, self._Z

    def get_S(self):
        """Amplitude matrix only — convenience wrapper around :meth:`get_SZ`."""
        return self.get_SZ()[0]

    def get_Z(self):
        """Phase matrix only — convenience wrapper around :meth:`get_SZ`."""
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
