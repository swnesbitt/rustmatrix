"""Hydrometeor mixtures.

Combine multiple hydrometeor species — each with its own PSD, scatterer
configuration, orientation PDF, refractive index, and shape — and read
the *combined* polarimetric radar observables through the existing
:mod:`rustmatrix.radar` and :mod:`rustmatrix.scatter` helpers.

Why summing S and Z works
-------------------------
Both the amplitude matrix ``S`` (forward, used by K_dp and A_i) and the
phase matrix ``Z`` (backscatter intensities, used by Z_h, Z_dr, rho_hv,
delta_hv, LDR) are *linear functionals of the number concentration*
``N(D)``. For a mixture of independent species,

    S_mix(geom) = sum_i int S_i(D, geom) N_i(D) dD
    Z_mix(geom) = sum_i int Z_i(D, geom) N_i(D) dD

Every non-linear observable (Z_dr, rho_hv, delta_hv, LDR) is a rational
function of the *total* ``S`` and ``Z``. Summing across species and
passing the combined matrices to :func:`radar.Zdr`, :func:`radar.rho_hv`,
etc. is therefore the physically correct incoherent-mixture recipe —
not an approximation.

Mixing fractions live inside each component's ``N_i(D)``: scale ``Nw``
in a :class:`~rustmatrix.psd.GammaPSD`, ``N0`` in an
:class:`~rustmatrix.psd.ExponentialPSD`, etc. There is no separate
weight scalar.

Example
-------
>>> from rustmatrix import Scatterer, HydroMix, MixtureComponent
>>> from rustmatrix.psd import PSDIntegrator, GammaPSD, ExponentialPSD
>>> # Configure each species' scatterer with an initialised PSDIntegrator
>>> # that registers every geometry the mixture will query, then:
>>> mix = HydroMix([
...     MixtureComponent(rain_scatterer,  GammaPSD(D0=1.5, Nw=8e3, mu=4), "rain"),
...     MixtureComponent(ice_scatterer,   ExponentialPSD(N0=5e3, Lambda=2.5), "ice"),
... ])
>>> mix.set_geometry(geom_horiz_back)
>>> from rustmatrix import radar
>>> Zh, Zdr, rho = radar.refl(mix), radar.Zdr(mix), radar.rho_hv(mix)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .psd import PSD
from .scatterer import Scatterer


@dataclass
class MixtureComponent:
    """One hydrometeor species inside a :class:`HydroMix`.

    Attributes
    ----------
    scatterer : Scatterer
        Species-specific scatterer. Must have ``psd_integrator`` attached
        and already initialised (``init_scatter_table(...)`` called) with
        every geometry the mixture will query.
    psd : PSD
        Number concentration ``N_i(D)`` for this species. Scale the PSD
        parameters (``Nw``, ``N0``, ...) to express the species' share
        of the mixture.
    label : str, optional
        Human-readable name used in error messages.
    """

    scatterer: Scatterer
    psd: PSD
    label: Optional[str] = None


class HydroMix:
    """Mixture of hydrometeor species with a Scatterer-shaped API.

    The instance exposes the same attributes and methods that the
    :mod:`radar` and :mod:`scatter` helpers read on a
    :class:`~rustmatrix.Scatterer` — ``wavelength``, ``Kw_sqr``,
    ``thet0/thet/phi0/phi/alpha/beta``, ``set_geometry``,
    ``get_geometry``, ``get_S``, ``get_Z``, ``get_SZ`` — so existing
    helpers work on a :class:`HydroMix` unchanged.

    Parameters
    ----------
    components : list of MixtureComponent, optional
        Initial components. More may be added later via :meth:`add`.
    Kw_sqr : float, optional
        Reference |K_w|^2 used by :func:`radar.refl` to normalise
        reflectivity. Defaults to the first component's ``Kw_sqr`` —
        typically the liquid-water value.

    Notes
    -----
    * All components must share ``wavelength``; a mismatch raises
      ``ValueError`` on :meth:`add`.
    * Each component's :class:`~rustmatrix.psd.PSDIntegrator` must have
      been initialised with every geometry the mixture will query.
      Missing geometries raise ``ValueError`` on :meth:`get_SZ`.
    * :attr:`psd_integrator` is always ``None`` on the mixture so that
      :func:`scatter.ext_xsect` falls back to its optical-theorem path,
      which reads the summed forward ``S`` matrix via :meth:`get_S`.
    """

    def __init__(
        self,
        components: Optional[List[MixtureComponent]] = None,
        Kw_sqr: Optional[float] = None,
    ):
        self._components: List[MixtureComponent] = []
        self._wavelength: Optional[float] = None
        self.Kw_sqr = Kw_sqr
        self.thet0 = 90.0
        self.thet = 90.0
        self.phi0 = 0.0
        self.phi = 180.0
        self.alpha = 0.0
        self.beta = 0.0
        self.psd_integrator = None

        if components:
            for c in components:
                self.add(c)

    @property
    def components(self) -> Tuple[MixtureComponent, ...]:
        return tuple(self._components)

    @property
    def wavelength(self) -> float:
        if self._wavelength is None:
            raise ValueError("HydroMix has no components; wavelength is undefined.")
        return self._wavelength

    def add(self, component: MixtureComponent) -> "HydroMix":
        if not isinstance(component, MixtureComponent):
            raise TypeError(
                "HydroMix.add expects a MixtureComponent, got "
                f"{type(component).__name__}."
            )
        sc = component.scatterer
        if sc.psd_integrator is None:
            raise ValueError(
                f"Component {component.label!r} has no psd_integrator. "
                "Attach a PSDIntegrator and call init_scatter_table before "
                "adding to a HydroMix."
            )
        if getattr(sc.psd_integrator, "_S_table", None) is None:
            raise ValueError(
                f"Component {component.label!r} has an uninitialised "
                "psd_integrator. Call init_scatter_table(...) first."
            )
        if self._wavelength is None:
            self._wavelength = sc.wavelength
            if self.Kw_sqr is None:
                self.Kw_sqr = sc.Kw_sqr
        elif sc.wavelength != self._wavelength:
            raise ValueError(
                f"Wavelength mismatch: component {component.label!r} has "
                f"wavelength={sc.wavelength}, but mixture is locked at "
                f"{self._wavelength}."
            )
        self._components.append(component)
        return self

    def set_geometry(self, geom):
        """Set ``(thet0, thet, phi0, phi, alpha, beta)`` on the mixture
        and propagate to every component."""
        (
            self.thet0,
            self.thet,
            self.phi0,
            self.phi,
            self.alpha,
            self.beta,
        ) = geom
        for c in self._components:
            c.scatterer.set_geometry(geom)

    def get_geometry(self):
        return (
            self.thet0,
            self.thet,
            self.phi0,
            self.phi,
            self.alpha,
            self.beta,
        )

    def get_SZ(self):
        """Summed ``(S, Z)`` across components at the current geometry.

        Returns
        -------
        S : ndarray (2, 2) complex
        Z : ndarray (4, 4) float
        """
        if not self._components:
            raise ValueError("HydroMix has no components.")
        geom = self.get_geometry()
        S = np.zeros((2, 2), dtype=complex)
        Z = np.zeros((4, 4), dtype=float)
        for c in self._components:
            integ = c.scatterer.psd_integrator
            if geom not in integ.geometries:
                raise ValueError(
                    f"Geometry {geom!r} is not registered on component "
                    f"{c.label!r}. Include it in PSDIntegrator.geometries "
                    "before init_scatter_table."
                )
            S_i, Z_i = integ.get_SZ(c.psd, geom)
            S = S + S_i
            Z = Z + Z_i
        return S, Z

    def get_S(self):
        return self.get_SZ()[0]

    def get_Z(self):
        return self.get_SZ()[1]
