"""Particle size distributions (PSDs) and size-distribution integration.

Port of ``pytmatrix.psd``. The :class:`PSDIntegrator` builds a lookup
table ``S(D), Z(D)`` at a fixed set of diameters and scattering
geometries, then integrates it against a PSD ``N(D)`` using the
trapezoidal rule. The per-diameter evaluation calls through to the Rust
T-matrix core.

Typical usage::

    sca = Scatterer(wavelength=wl_C, m=complex(7.99, 2.21), axis_ratio=1/0.9)
    sca.psd_integrator = PSDIntegrator()
    sca.psd_integrator.D_max = 8.0
    sca.psd_integrator.num_points = 256
    sca.psd_integrator.init_scatter_table(sca)
    sca.psd = GammaPSD(D0=2.0, Nw=1e3, mu=4)
    S, Z = sca.get_SZ()
"""

from __future__ import annotations

import pickle
import warnings
from datetime import datetime
from typing import Callable, Optional

import numpy as np
from scipy.special import gamma

from . import _core, tmatrix_aux
from . import orientation as _orientation

# numpy 2.0 renamed trapz -> trapezoid; scipy >= 1.14 dropped trapz from
# scipy.integrate. Cover both.
try:
    from numpy import trapezoid as _trapezoid
except ImportError:  # numpy < 2.0
    from numpy import trapz as _trapezoid


class PSD:
    """Abstract PSD base class; default behaviour is identically zero."""

    def __call__(self, D):
        if np.shape(D) == ():
            return 0.0
        return np.zeros_like(D)

    def __eq__(self, other):
        return False


class ExponentialPSD(PSD):
    """Exponential PSD ``N(D) = N0 exp(-Lambda D)``.

    Truncates at ``D_max`` (default ``11/Lambda ~ 3*D0``).
    """

    def __init__(self, N0: float = 1.0, Lambda: float = 1.0, D_max: Optional[float] = None):
        self.N0 = float(N0)
        self.Lambda = float(Lambda)
        self.D_max = 11.0 / Lambda if D_max is None else D_max

    def __call__(self, D):
        psd = self.N0 * np.exp(-self.Lambda * D)
        if np.shape(D) == ():
            if D > self.D_max:
                return 0.0
        else:
            psd[D > self.D_max] = 0.0
        return psd

    def __eq__(self, other):
        try:
            return (
                isinstance(other, ExponentialPSD)
                and self.N0 == other.N0
                and self.Lambda == other.Lambda
                and self.D_max == other.D_max
            )
        except AttributeError:
            return False


class UnnormalizedGammaPSD(ExponentialPSD):
    """``N(D) = N0 * D^mu * exp(-Lambda D)`` (unnormalised gamma)."""

    def __init__(
        self,
        N0: float = 1.0,
        Lambda: float = 1.0,
        mu: float = 0.0,
        D_max: Optional[float] = None,
    ):
        super().__init__(N0=N0, Lambda=Lambda, D_max=D_max)
        self.mu = mu

    def __call__(self, D):
        # log-space is numerically safer than D**mu for large mu
        psd = self.N0 * np.exp(self.mu * np.log(D) - self.Lambda * D)
        if np.shape(D) == ():
            if D > self.D_max or D == 0:
                return 0.0
        else:
            psd[(D > self.D_max) | (D == 0)] = 0.0
        return psd

    def __eq__(self, other):
        try:
            return super().__eq__(other) and self.mu == other.mu
        except AttributeError:
            return False


class GammaPSD(PSD):
    """Normalised gamma PSD (Bringi/Chandrasekar convention).

    ``N(D) = Nw * f(mu) * (D/D0)^mu * exp(-(3.67+mu) D/D0)``,
    ``f(mu) = 6/3.67^4 * (3.67+mu)^(mu+4) / Gamma(mu+4)``.
    """

    def __init__(
        self,
        D0: float = 1.0,
        Nw: float = 1.0,
        mu: float = 0.0,
        D_max: Optional[float] = None,
    ):
        self.D0 = float(D0)
        self.mu = float(mu)
        self.D_max = 3.0 * D0 if D_max is None else D_max
        self.Nw = float(Nw)
        self.nf = Nw * 6.0 / 3.67 ** 4 * (3.67 + mu) ** (mu + 4) / gamma(mu + 4)

    def __call__(self, D):
        d = D / self.D0
        psd = self.nf * np.exp(self.mu * np.log(d) - (3.67 + self.mu) * d)
        if np.shape(D) == ():
            if D > self.D_max or D == 0.0:
                return 0.0
        else:
            psd[(D > self.D_max) | (D == 0.0)] = 0.0
        return psd

    def __eq__(self, other):
        try:
            return (
                isinstance(other, GammaPSD)
                and self.D0 == other.D0
                and self.Nw == other.Nw
                and self.mu == other.mu
                and self.D_max == other.D_max
            )
        except AttributeError:
            return False


class BinnedPSD(PSD):
    """Step-function PSD from ``bin_edges`` (n+1 values) and ``bin_psd`` (n)."""

    def __init__(self, bin_edges, bin_psd):
        if len(bin_edges) != len(bin_psd) + 1:
            raise ValueError("There must be n+1 bin edges for n bins.")
        self.bin_edges = bin_edges
        self.bin_psd = bin_psd

    def psd_for_D(self, D):
        if not (self.bin_edges[0] < D <= self.bin_edges[-1]):
            return 0.0
        # Binary search to locate bin.
        start = 0
        end = len(self.bin_edges)
        while end - start > 1:
            half = (start + end) // 2
            if self.bin_edges[start] < D <= self.bin_edges[half]:
                end = half
            else:
                start = half
        return self.bin_psd[start]

    def __call__(self, D):
        if np.shape(D) == ():
            return self.psd_for_D(D)
        return np.array([self.psd_for_D(d) for d in D])

    def __eq__(self, other):
        if other is None:
            return False
        return (
            len(self.bin_edges) == len(other.bin_edges)
            and (np.asarray(self.bin_edges) == np.asarray(other.bin_edges)).all()
            and (np.asarray(self.bin_psd) == np.asarray(other.bin_psd)).all()
        )


class PSDIntegrator:
    """Integrates scattering properties over a particle-size distribution.

    Attach an instance to ``Scatterer.psd_integrator`` and set
    ``Scatterer.psd``; then :meth:`Scatterer.get_SZ` returns the
    PSD-averaged ``(S, Z)``.

    Attributes:
        num_points: number of diameters used to sample the lookup table.
        m_func: optional callable ``m(D)`` to vary refractive index with
            diameter. If ``None`` the scatterer's single ``m`` is used.
        axis_ratio_func: optional callable ``eps(D)`` for drop-shape
            relationships.
        D_max: largest diameter to tabulate (usually the largest ``D_max``
            of any PSD that will be passed in).
        geometries: tuple of ``(thet0, thet, phi0, phi, alpha, beta)``
            scattering geometries to precompute.
    """

    attrs = {"num_points", "m_func", "axis_ratio_func", "D_max", "geometries"}

    def __init__(self, **kwargs):
        self.num_points = 1024
        self.m_func: Optional[Callable[[float], complex]] = None
        self.axis_ratio_func: Optional[Callable[[float], float]] = None
        self.D_max: Optional[float] = None
        self.geometries = (tmatrix_aux.geom_horiz_back,)

        for k, v in kwargs.items():
            if k in self.attrs:
                setattr(self, k, v)

        self._S_table = None
        self._Z_table = None
        self._angular_table = None
        self._previous_psd = None

    def __call__(self, psd, geometry):
        return self.get_SZ(psd, geometry)

    def get_SZ(self, psd, geometry):
        """PSD-integrated ``(S, Z)`` for the given scattering geometry."""
        if self._S_table is None or self._Z_table is None:
            raise AttributeError("Initialize or load the scattering table first.")

        if not isinstance(psd, PSD) or self._previous_psd != psd:
            self._S_dict = {}
            self._Z_dict = {}
            psd_w = psd(self._psd_D)
            for geom in self.geometries:
                # _S_table[geom] has shape (2, 2, num_points); trapezoid
                # integrates along axis=-1 by default.
                self._S_dict[geom] = _trapezoid(
                    self._S_table[geom] * psd_w, self._psd_D
                )
                self._Z_dict[geom] = _trapezoid(
                    self._Z_table[geom] * psd_w, self._psd_D
                )
            self._previous_psd = psd

        return self._S_dict[geometry], self._Z_dict[geometry]

    def get_angular_integrated(self, psd, geometry, property_name, h_pol=True):
        """PSD-integrated angular quantity (sca_xsect / ext_xsect / asym)."""
        if self._angular_table is None:
            raise AttributeError(
                "Initialize or load the table of angular-integrated quantities first."
            )

        pol = "h_pol" if h_pol else "v_pol"
        psd_w = psd(self._psd_D)

        def sca_xsect(geom):
            return _trapezoid(
                self._angular_table["sca_xsect"][pol][geom] * psd_w, self._psd_D
            )

        if property_name == "sca_xsect":
            return sca_xsect(geometry)
        if property_name == "ext_xsect":
            return _trapezoid(
                self._angular_table["ext_xsect"][pol][geometry] * psd_w, self._psd_D
            )
        if property_name == "asym":
            sca_int = sca_xsect(geometry)
            if sca_int <= 0:
                return 0.0
            num = _trapezoid(
                self._angular_table["asym"][pol][geometry]
                * self._angular_table["sca_xsect"][pol][geometry]
                * psd_w,
                self._psd_D,
            )
            return num / sca_int
        raise ValueError(f"Unknown property_name {property_name!r}")

    def init_scatter_table(self, tm, angular_integration: bool = False, verbose: bool = False):
        """Populate the diameter-indexed lookup tables.

        Walks ``self.num_points`` equally-spaced diameters from
        ``D_max/num_points`` to ``D_max``, evaluates ``tm.get_SZ_orient()``
        at each of the registered ``self.geometries``, and caches the
        amplitude/phase matrices. If ``angular_integration=True`` also
        tabulates polarised scattering and extinction cross-sections and
        the asymmetry parameter at each diameter.
        """
        if self.D_max is None:
            raise AttributeError("PSDIntegrator.D_max must be set before init_scatter_table.")

        # Deferred to avoid a module-level cycle (scatter imports from here).
        from . import scatter

        self._psd_D = np.linspace(
            self.D_max / self.num_points, self.D_max, self.num_points
        )

        self._S_table = {}
        self._Z_table = {}
        self._previous_psd = None
        self._m_table = np.empty(self.num_points, dtype=complex)

        if angular_integration:
            self._angular_table = {
                "sca_xsect": {"h_pol": {}, "v_pol": {}},
                "ext_xsect": {"h_pol": {}, "v_pol": {}},
                "asym": {"h_pol": {}, "v_pol": {}},
            }
        else:
            self._angular_table = None

        old_m = tm.m
        old_axis_ratio = tm.axis_ratio
        old_radius = tm.radius
        old_geom = tm.get_geometry()
        old_psd_integrator = tm.psd_integrator

        # Evaluate per-diameter m and axis_ratio up front so the Rust
        # tabulator (which can't call back into Python) has plain arrays.
        if self.m_func is not None:
            for i, D in enumerate(self._psd_D):
                self._m_table[i] = self.m_func(D)
        else:
            self._m_table[:] = tm.m
        if self.axis_ratio_func is not None:
            axis_ratios = np.array(
                [self.axis_ratio_func(D) for D in self._psd_D], dtype=float
            )
        else:
            axis_ratios = np.full(self.num_points, float(tm.axis_ratio))

        # Rust fast paths: single orientation and fixed-quadrature
        # orientation averaging, both parallelised across diameters.
        # Adaptive orientation averaging (scipy.dblquad) and the
        # ``angular_integration`` path still fall through to Python,
        # because they rely on per-sample Python callbacks.
        orient_fn = getattr(tm, "orient", _orientation.orient_single)
        use_rust_single = (
            not angular_integration and orient_fn is _orientation.orient_single
        )
        use_rust_orient_avg = (
            not angular_integration
            and orient_fn is _orientation.orient_averaged_fixed
        )

        try:
            # Disable PSD integration on the scatterer to avoid recursion
            # through get_SZ.
            tm.psd_integrator = None

            for geom in self.geometries:
                self._S_table[geom] = np.empty((2, 2, self.num_points), dtype=complex)
                self._Z_table[geom] = np.empty((4, 4, self.num_points))
                if angular_integration:
                    for key in ("sca_xsect", "ext_xsect", "asym"):
                        for pol in ("h_pol", "v_pol"):
                            self._angular_table[key][pol][geom] = np.empty(self.num_points)

            if use_rust_single or use_rust_orient_avg:
                geoms = [tuple(g) for g in self.geometries]
                common = (
                    np.ascontiguousarray(self._psd_D, dtype=float),
                    np.ascontiguousarray(axis_ratios, dtype=float),
                    np.ascontiguousarray(self._m_table.real, dtype=float),
                    np.ascontiguousarray(self._m_table.imag, dtype=float),
                    geoms,
                )
                extras = (
                    float(tm.radius_type),
                    float(tm.wavelength),
                    int(tm.shape),
                    float(tm.ddelt),
                    int(tm.ndgs),
                )
                if use_rust_orient_avg:
                    # Build the (alpha, beta) quadrature the same way the
                    # Python orient_averaged_fixed does, then hand the nodes
                    # to the Rust averager.
                    tm._init_orient()
                    alphas = np.linspace(0, 360, tm.n_alpha + 1)[:-1]
                    S_batch, Z_batch = _core.tabulate_scatter_table_orient_avg(
                        *common,
                        np.ascontiguousarray(alphas, dtype=float),
                        np.ascontiguousarray(tm.beta_p, dtype=float),
                        np.ascontiguousarray(tm.beta_w, dtype=float),
                        *extras,
                    )
                else:
                    S_batch, Z_batch = _core.tabulate_scatter_table(
                        *common, *extras,
                    )
                # S_batch: (num_points, num_geoms, 2, 2); Z_batch: (..., 4, 4).
                # Our on-disk layout is (2, 2, num_points) per geom — reshape.
                for g_idx, geom in enumerate(self.geometries):
                    self._S_table[geom] = np.moveaxis(S_batch[:, g_idx, :, :], 0, -1)
                    self._Z_table[geom] = np.moveaxis(Z_batch[:, g_idx, :, :], 0, -1)
            else:
                # Fallback: Python loop (orientation-averaged or angular
                # integration). Keeps callbacks like ``tm.orient`` working.
                for i, D in enumerate(self._psd_D):
                    if verbose:
                        print(f"Computing point {i} at D={D}...")
                    tm.m = self._m_table[i]
                    tm.axis_ratio = axis_ratios[i]
                    tm.radius = D / 2.0
                    for geom in self.geometries:
                        tm.set_geometry(geom)
                        S, Z = tm.get_SZ_orient()
                        self._S_table[geom][:, :, i] = S
                        self._Z_table[geom][:, :, i] = Z
                        if angular_integration:
                            for pol in ("h_pol", "v_pol"):
                                h_pol = pol == "h_pol"
                                self._angular_table["sca_xsect"][pol][geom][i] = (
                                    scatter.sca_xsect(tm, h_pol=h_pol)
                                )
                                self._angular_table["ext_xsect"][pol][geom][i] = (
                                    scatter.ext_xsect(tm, h_pol=h_pol)
                                )
                                self._angular_table["asym"][pol][geom][i] = (
                                    scatter.asym(tm, h_pol=h_pol)
                                )
        finally:
            tm.m = old_m
            tm.axis_ratio = old_axis_ratio
            tm.radius = old_radius
            tm.psd_integrator = old_psd_integrator
            tm.set_geometry(old_geom)

    def save_scatter_table(self, fn: str, description: str = "") -> None:
        """Pickle the lookup tables to disk."""
        data = {
            "description": description,
            "time": datetime.now(),
            "psd_scatter": (
                self.num_points,
                self.D_max,
                self._psd_D,
                self._S_table,
                self._Z_table,
                self._angular_table,
                self._m_table,
                self.geometries,
            ),
            "version": tmatrix_aux.VERSION,
        }
        with open(fn, "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def load_scatter_table(self, fn: str):
        """Load a pickled lookup table saved by :meth:`save_scatter_table`."""
        with open(fn, "rb") as f:
            data = pickle.load(f)
        if "version" not in data or data["version"] != tmatrix_aux.VERSION:
            warnings.warn("Loading data saved with another version.", Warning)
        (
            self.num_points,
            self.D_max,
            self._psd_D,
            self._S_table,
            self._Z_table,
            self._angular_table,
            self._m_table,
            self.geometries,
        ) = data["psd_scatter"]
        return data["time"], data["description"]
