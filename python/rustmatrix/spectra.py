"""Doppler and polarimetric spectra.

Evaluate Doppler-velocity-resolved scattering matrices ``S_spec(v)`` and
``Z_spec(v)`` for a hydrometeor PSD (single species) or a
:class:`~rustmatrix.hd_mix.HydroMix`, then derive the spectral radar
observables (sZ_h, sZ_dr, sK_dp, sρ_hv, sδ_hv, sLDR) by applying the
bulk :mod:`rustmatrix.radar` formulas bin-by-bin.

Physics
-------
Line-of-sight velocity for a vertical-pointing radar is
``v = v_t(D) + w``, where ``v_t(D)`` is terminal fall speed and ``w``
is the mean vertical air motion.

Sign convention: **positive velocity = toward a downward-pointing
radar = fall direction**. For an up-looking radar flip the sign of
``w`` and ``v_bins`` (it is a pure sign flip — the kernels are even).

Finite beamwidth combined with horizontal wind broadens the spectrum
by a deterministic Gaussian of width
``σ_beam = |u_h| · θ_b / (2 √(2 ln 2))`` (Doviak & Zrnić 1993, small-θ_b
vertical-pointing limit). This ``σ_beam`` adds in quadrature to the
diameter-dependent turbulence width ``σ_t(D)`` when the spectral
Gaussian kernel is built.

``S_spec`` and ``Z_spec`` are linear in N(D), so a
:class:`~rustmatrix.hd_mix.HydroMix` is handled by summing per-component
spectra on a shared velocity grid.

System noise
------------
Radar receivers have a thermal noise floor that appears in every
velocity bin. The noise is uncorrelated between H and V channels, so
it adds incoherently to ``sZ_h`` and ``sZ_v`` but does **not** feed the
underlying ``S_spec`` / ``Z_spec`` matrices — those remain the
signal-only scattering matrices and continue to round-trip to the bulk
``radar.*`` helpers exactly. Noise biases ``sZ_dr``, ``sρ_hv``,
``sLDR`` through the per-bin SNR.

Pass ``noise=`` to :class:`SpectralIntegrator`:

* ``None`` (default) — no noise, the spectrum is signal-only.
* ``"realistic"`` or ``True`` — use :func:`realistic_noise_floor` for the
  scatterer's wavelength (a -20 dBZ total reflectivity by default).
* a ``float`` — total noise reflectivity [mm⁶ m⁻³] distributed uniformly
  across the velocity grid and applied equally to H and V.
* a 2-tuple ``(noise_h, noise_v)`` [mm⁶ m⁻³] — separate H and V noise.

References
----------
Doviak, R. J. & Zrnić, D. S. (1993). *Doppler Radar and Weather
Observations*, 2nd ed., Academic Press. (§5.3 beam-broadening.)

Zeng, Y., Janjić, T., de Lozar, A., Blahak, U., Reich, H., Keil, C.,
Seifert, A., Hagen, M., Tridon, F., Kneifel, S. (2023). "Interpreting
Doppler spectra measured by cloud radars using a particle inertia
model."  *Atmos. Meas. Tech.*, 16, 3727–3757. doi:10.5194/amt-16-3727-2023
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Dict, Mapping, Optional, Tuple, Union
import warnings

import numpy as np

try:
    from numpy import trapezoid as _trapezoid
except ImportError:  # numpy < 2.0
    from numpy import trapz as _trapezoid

from . import radar as _radar
from . import scatter as _scatter
from .hd_mix import HydroMix, MixtureComponent
from .psd import PSD
from .scatterer import Scatterer
from .tmatrix_aux import geom_vert_back


# ---------------------------------------------------------------------------
# Fall-speed presets  (spectra.fall_speed.*)
# ---------------------------------------------------------------------------

def _to_array(D):
    return np.atleast_1d(np.asarray(D, dtype=float))


def atlas_srivastava_sekhon_1973(D, rho_ratio: float = 1.0):
    """Atlas–Srivastava–Sekhon (1973) rain terminal velocity [m/s].

    ``v_t(D) = (9.65 − 10.3 · exp(−0.6 · D)) · rho_ratio``

    Parameters
    ----------
    D : array_like
        Equivalent drop diameter [mm].
    rho_ratio : float
        ``(ρ_0 / ρ_air)^0.4`` air-density correction (1.0 at sea level).
    """
    Dv = _to_array(D)
    return rho_ratio * (9.65 - 10.3 * np.exp(-0.6 * Dv))


def brandes_et_al_2002(D):
    """Brandes et al. (2002) 4th-order polynomial rain fall speed [m/s].

    ``v_t(D) = −0.1021 + 4.932 D − 0.9551 D² + 0.07934 D³ − 0.002362 D⁴``
    (valid 0.1 ≤ D ≤ 8 mm; clamped to zero below 0.1 mm).
    """
    Dv = _to_array(D)
    v = (
        -0.1021
        + 4.932 * Dv
        - 0.9551 * Dv ** 2
        + 0.07934 * Dv ** 3
        - 0.002362 * Dv ** 4
    )
    v[Dv < 0.1] = 0.0
    return np.maximum(v, 0.0)


def beard_1976(D, T: float = 293.15, P: float = 101325.0):
    """Beard-style rain terminal velocity with explicit T, P dependence [m/s].

    Uses the Brandes et al. (2002) sea-level fit as the reference fall
    speed and applies Beard's (1977) density correction
    ``(ρ₀ / ρ)^0.5`` to shift to arbitrary (T, P). This is the practical
    form used in most cloud microphysics codes when "Beard 1976" is
    requested with a non-standard ambient state — the explicit
    three-regime Beard (1976) formulation is notoriously sensitive to
    coefficient transcription and gains little accuracy over the
    much-simpler Brandes fit below 7 mm.

    Parameters
    ----------
    D : array_like
        Equivalent drop diameter [mm].
    T : float
        Air temperature [K]. Default 293.15 K (20 °C).
    P : float
        Air pressure [Pa]. Default 101325 Pa.

    References
    ----------
    Beard, K. V. (1977). Terminal velocity adjustment for cloud and
    precipitation drops aloft. *J. Atmos. Sci.*, 34, 1293–1298.
    Brandes, E. A., Zhang, G., & Vivekanandan, J. (2002). Experiments in
    rainfall estimation with a polarimetric radar in a subtropical
    environment. *J. Appl. Meteor.*, 41, 674–685.
    """
    Dv = _to_array(D)
    R_d = 287.05
    rho0 = 1.2041  # reference dry-air density at 20 °C, 1 atm
    rho = P / (R_d * T)
    density_correction = np.sqrt(rho0 / rho)
    return brandes_et_al_2002(Dv) * density_correction


def locatelli_hobbs_1974_aggregates(D):
    """Locatelli & Hobbs (1974) unrimed aggregates (mixed habits) [m/s].

    ``v_t(D) = 0.69 · D^0.41`` (D in mm, valid 1 ≤ D ≤ 10 mm).
    """
    Dv = _to_array(D)
    return 0.69 * np.power(np.maximum(Dv, 1e-6), 0.41)


def locatelli_hobbs_1974_graupel_hex(D):
    """Locatelli & Hobbs (1974) hexagonal graupel [m/s].

    ``v_t(D) = 1.1 · D^0.57`` (D in mm, valid 0.5 ≤ D ≤ 2 mm).
    """
    Dv = _to_array(D)
    return 1.1 * np.power(np.maximum(Dv, 1e-6), 0.57)


def power_law(a: float, b: float, D_ref: float = 1.0, c: float = 0.0) -> Callable:
    """Return a user-parametrised power-law fall-speed callable.

    ``v_t(D) = a · (D / D_ref)^b + c``     [m/s, D in mm]

    Parameters
    ----------
    a, b : float
        Prefactor and exponent of the power law.
    D_ref : float
        Reference diameter used to normalise ``D`` (mm). Default 1.0.
    c : float
        Additive offset [m/s]. Default 0.0.
    """

    def _f(D):
        Dv = _to_array(D)
        return a * np.power(np.maximum(Dv, 1e-12) / D_ref, b) + c

    _f.__name__ = f"power_law(a={a}, b={b}, D_ref={D_ref}, c={c})"
    return _f


class fall_speed:  # noqa: N801 — used as a namespace, not a class
    """Namespace of fall-speed presets. Call any as ``fall_speed.preset(D)``."""

    atlas_srivastava_sekhon_1973 = staticmethod(atlas_srivastava_sekhon_1973)
    brandes_et_al_2002 = staticmethod(brandes_et_al_2002)
    beard_1976 = staticmethod(beard_1976)
    locatelli_hobbs_1974_aggregates = staticmethod(locatelli_hobbs_1974_aggregates)
    locatelli_hobbs_1974_graupel_hex = staticmethod(locatelli_hobbs_1974_graupel_hex)
    power_law = staticmethod(power_law)


# ---------------------------------------------------------------------------
# Turbulence models  (spectra.turbulence.*)
# ---------------------------------------------------------------------------

class _TurbulenceModel:
    """Base class: ``__call__(D) -> σ_t(D)`` [m/s]."""

    def __call__(self, D):
        raise NotImplementedError


class NoTurbulence(_TurbulenceModel):
    """``σ_t(D) = 0`` everywhere (delta-function binning)."""

    def __call__(self, D):
        return np.zeros_like(_to_array(D))

    def __repr__(self):
        return "NoTurbulence()"


class GaussianTurbulence(_TurbulenceModel):
    """Constant Gaussian broadening: ``σ_t(D) = σ_t``.

    The canonical spectral-polarimetry assumption. Matches classical
    papers that parameterise broadening with a single bulk σ_t.
    """

    def __init__(self, sigma_t: float):
        if sigma_t < 0:
            raise ValueError("sigma_t must be non-negative.")
        self.sigma_t = float(sigma_t)

    def __call__(self, D):
        return np.full_like(_to_array(D), self.sigma_t)

    def __repr__(self):
        return f"GaussianTurbulence(sigma_t={self.sigma_t})"


class InertialZeng2023(_TurbulenceModel):
    """Particle-inertia-corrected turbulence response (Zeng et al. 2023).

    Heavy drops cannot follow small-scale eddies, so their effective
    turbulence broadening is a low-pass-filtered version of the ambient
    air turbulence. We use a first-order response:

        σ_t(D) = σ_air / √(1 + St(D)²)

    with Stokes number ``St(D) = τ_p(D) / τ_eddy``, particle relaxation
    time ``τ_p(D) = v_t_ref(D) / g`` (g = 9.80665 m/s²), and eddy
    turnover time ``τ_eddy = (L_o² / ε)^(1/3)``.

    Limits:
    * Small drops (``St → 0``) → ``GaussianTurbulence(σ_air)``.
    * Large drops (``St → ∞``) → ``NoTurbulence()``.

    Parameters
    ----------
    sigma_air : float
        Ambient turbulence intensity [m/s].
    epsilon : float
        Turbulent kinetic energy dissipation rate [m²/s³].
    L_o : float, optional
        Outer (integral) length scale [m]. Default 100 m.
    v_t_ref : callable, optional
        ``D -> v_t(D)`` used to estimate ``τ_p``. Defaults to
        :func:`atlas_srivastava_sekhon_1973`. Pass the same fall-speed
        model used in the integrator for consistency.

    References
    ----------
    Zeng et al. (2023), *Atmos. Meas. Tech.*, 16, 3727.
    """

    _g = 9.80665  # m/s²

    def __init__(
        self,
        sigma_air: float,
        epsilon: float,
        L_o: float = 100.0,
        v_t_ref: Optional[Callable] = None,
    ):
        if sigma_air < 0:
            raise ValueError("sigma_air must be non-negative.")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        if L_o <= 0:
            raise ValueError("L_o must be positive.")
        self.sigma_air = float(sigma_air)
        self.epsilon = float(epsilon)
        self.L_o = float(L_o)
        self.v_t_ref = v_t_ref if v_t_ref is not None else atlas_srivastava_sekhon_1973
        self.tau_eddy = (self.L_o ** 2 / self.epsilon) ** (1.0 / 3.0)

    def __call__(self, D):
        Dv = _to_array(D)
        tau_p = self.v_t_ref(Dv) / self._g
        St = tau_p / self.tau_eddy
        return self.sigma_air / np.sqrt(1.0 + St ** 2)

    def __repr__(self):
        return (
            f"InertialZeng2023(sigma_air={self.sigma_air}, "
            f"epsilon={self.epsilon}, L_o={self.L_o})"
        )


def _normalise_turbulence(t) -> Callable:
    """Accept a _TurbulenceModel, or any `D -> σ_t(D)` callable."""
    if t is None:
        return NoTurbulence()
    if isinstance(t, _TurbulenceModel):
        return t
    if callable(t):
        return t
    raise TypeError(
        "turbulence must be a TurbulenceModel, a callable D->σ_t, or None."
    )


class turbulence:  # noqa: N801 — namespace
    """Namespace for turbulence models. ``turbulence.from_params(...)``
    picks the right concrete model from simple scalar inputs."""

    NoTurbulence = NoTurbulence
    GaussianTurbulence = GaussianTurbulence
    InertialZeng2023 = InertialZeng2023

    @staticmethod
    def from_params(
        sigma: Optional[float] = None,
        epsilon: Optional[float] = None,
        L_o: float = 100.0,
        v_t_ref: Optional[Callable] = None,
    ) -> _TurbulenceModel:
        """Smart constructor.

        * ``epsilon`` supplied ⇒ :class:`InertialZeng2023` (with
          ``sigma_air = sigma`` if given, else a default of 0.3 m/s).
        * only ``sigma`` supplied ⇒ :class:`GaussianTurbulence`.
        * both ``None`` (or ``sigma == 0`` and no ε) ⇒ :class:`NoTurbulence`.
        """
        if epsilon is not None:
            return InertialZeng2023(
                sigma_air=sigma if sigma is not None else 0.3,
                epsilon=epsilon,
                L_o=L_o,
                v_t_ref=v_t_ref,
            )
        if sigma is None or sigma == 0:
            return NoTurbulence()
        return GaussianTurbulence(sigma_t=sigma)


# ---------------------------------------------------------------------------
# System noise
# ---------------------------------------------------------------------------

#: Default "realistic" total noise reflectivity [dBZ] when the user asks
#: for a realistic default. Picked to be a sensible mid-range value that
#: is representative of research radars without being wavelength-specific
#: (operational S-band is quieter, cloud radars noisier in absolute dBZ
#: but equivalent per-range-gate).
REALISTIC_NOISE_DBZ = -20.0


def realistic_noise_floor(
    wavelength_mm: float = 33.3, Z_dBZ: float = REALISTIC_NOISE_DBZ
) -> float:
    """Return a realistic total noise reflectivity in mm⁶ m⁻³.

    Converts a noise-equivalent reflectivity in dBZ to linear units. The
    default (-20 dBZ) is a realistic per-range-gate noise floor for a
    research radar; operational S-band systems are typically quieter
    (-30 to -40 dBZ), while cloud radars at Ka/W with short integration
    times can be noisier.

    The wavelength argument is accepted for signature-compatibility so
    callers can write ``realistic_noise_floor(scatterer.wavelength)``
    even though the default is currently wavelength-independent.

    Parameters
    ----------
    wavelength_mm : float
        Radar wavelength [mm]. Currently unused by the default model but
        reserved for future band-dependent presets.
    Z_dBZ : float
        Total noise reflectivity [dBZ] across the Nyquist range.

    Returns
    -------
    float
        Total noise reflectivity in linear mm⁶ m⁻³.
    """
    _ = wavelength_mm  # reserved; kept for future band-dependent behaviour
    return 10.0 ** (Z_dBZ / 10.0)


def _resolve_noise(noise, wavelength_mm: float) -> Tuple[float, float]:
    """Return ``(noise_h, noise_v)`` in mm⁶ m⁻³ from the user spec.

    ``None`` or 0 ⇒ ``(0, 0)`` (no noise).
    ``"realistic"`` / ``"default"`` / ``True`` ⇒ realistic default on both channels.
    ``float`` ⇒ same value on H and V.
    ``(nh, nv)`` ⇒ separate H and V values.
    """
    if noise is None or noise is False:
        return 0.0, 0.0
    if noise is True or (isinstance(noise, str) and noise.lower() in ("realistic", "default")):
        n = realistic_noise_floor(wavelength_mm)
        return n, n
    if isinstance(noise, (tuple, list)):
        if len(noise) != 2:
            raise ValueError("`noise` tuple must be (noise_h, noise_v).")
        nh, nv = float(noise[0]), float(noise[1])
        if nh < 0 or nv < 0:
            raise ValueError("noise values must be non-negative.")
        return nh, nv
    try:
        n = float(noise)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "`noise` must be None, True, 'realistic', a float, or a (noise_h, noise_v) tuple."
        ) from exc
    if n < 0:
        raise ValueError("noise value must be non-negative.")
    return n, n


# ---------------------------------------------------------------------------
# Spectral result container
# ---------------------------------------------------------------------------

@dataclass
class SpectralResult:
    """Doppler-resolved scattering matrices and derived observables.

    Attributes
    ----------
    v : ndarray (M,)
        Doppler-velocity bin centres [m/s].
    S_spec : ndarray (2, 2, M) complex
        Spectral amplitude matrix.
    Z_spec : ndarray (4, 4, M) float
        Spectral phase matrix.
    sZ_h, sZ_v : ndarray (M,)
        Spectral linear reflectivity [mm⁶ m⁻³ / (m/s)], horizontal /
        vertical polarisation.
    sZ_dr : ndarray (M,)
        Spectral differential reflectivity (linear ratio).
    sKdp : ndarray (M,) or None
        Spectral specific differential phase [° / km]. ``None`` when no
        forward-scatter geometry was supplied.
    srho_hv : ndarray (M,)
        Spectral copolar correlation coefficient (0–1).
    sdelta_hv : ndarray (M,)
        Spectral backscatter differential phase [rad].
    sLDR : ndarray (M,)
        Spectral linear depolarisation ratio (linear).
    wavelength : float
        Wavelength [mm] (copied from the scatterer for `collapse_to_bulk`).
    Kw_sqr : float
        |K_w|² (copied for `collapse_to_bulk`).
    """

    v: np.ndarray
    S_spec: np.ndarray
    Z_spec: np.ndarray
    sZ_h: np.ndarray
    sZ_v: np.ndarray
    sZ_dr: np.ndarray
    sKdp: Optional[np.ndarray]
    srho_hv: np.ndarray
    sdelta_hv: np.ndarray
    sLDR: np.ndarray
    wavelength: float
    Kw_sqr: float
    noise_h: float = 0.0
    noise_v: float = 0.0
    _S_spec_forward: Optional[np.ndarray] = None  # per-bin forward S for sKdp re-derivation

    def collapse_to_bulk(self) -> SimpleNamespace:
        """Integrate S_spec / Z_spec over v and return a Scatterer-shaped
        shim so bulk :mod:`radar` helpers can be called on the integrated
        matrices.

        The returned object has ``.get_S()``, ``.get_Z()``,
        ``.wavelength``, ``.Kw_sqr``, and a backscatter geometry — enough
        for ``radar.refl``, ``radar.Zdr``, ``radar.rho_hv``,
        ``radar.delta_hv`` to work. If the spectrum was built with a
        forward geometry, ``.get_S_forward()`` returns the forward-summed
        S so you can hand-construct ``radar.Kdp``-style quantities.
        """
        Z_sum = _trapezoid(self.Z_spec, self.v, axis=-1)
        S_sum_back = _trapezoid(self.S_spec, self.v, axis=-1)
        if self._S_spec_forward is not None:
            S_sum_forward = _trapezoid(self._S_spec_forward, self.v, axis=-1)
        else:
            S_sum_forward = None

        shim = SimpleNamespace()
        shim.wavelength = self.wavelength
        shim.Kw_sqr = self.Kw_sqr
        shim.thet0, shim.thet, shim.phi0, shim.phi = 0.0, 180.0, 0.0, 0.0
        shim.alpha, shim.beta = 0.0, 0.0
        shim.psd_integrator = None
        shim.get_S = lambda _S=S_sum_back: _S
        shim.get_Z = lambda _Z=Z_sum: _Z
        shim.get_SZ = lambda _S=S_sum_back, _Z=Z_sum: (_S, _Z)
        if S_sum_forward is not None:
            shim.get_S_forward = lambda _S=S_sum_forward: _S
        return shim


# ---------------------------------------------------------------------------
# SpectralIntegrator
# ---------------------------------------------------------------------------

class SpectralIntegrator:
    """Build spectral S and Z matrices on a Doppler-velocity grid.

    Parameters
    ----------
    source : Scatterer or HydroMix
        The scattering target. A :class:`~rustmatrix.Scatterer` must have
        an initialised ``psd_integrator`` and a ``psd`` attached. A
        :class:`~rustmatrix.HydroMix` carries its own components (each
        with ``scatterer.psd``).
    fall_speed : callable, optional
        ``D_mm -> v_t_m_per_s`` for the **single-species** case. Ignored
        when ``source`` is a ``HydroMix``.
    turbulence : _TurbulenceModel or callable or None, optional
        Turbulence model for the single-species case. Ignored when
        ``source`` is a ``HydroMix``.
    component_kinematics : mapping, optional
        For ``HydroMix`` inputs only. Maps component label (or index)
        to ``(fall_speed, turbulence)``. Every component in the mixture
        must be represented.
    v_bins : array_like, optional
        Explicit 1-D velocity grid [m/s]. Mutually exclusive with
        ``(v_min, v_max, n_bins)``.
    v_min, v_max, n_bins : float, float, int, optional
        Convenience triple that builds ``v_bins = linspace(v_min, v_max, n_bins)``.
    w : float
        Mean vertical air motion [m/s], positive-downward. Default 0.
    u_h : float
        Horizontal wind speed magnitude [m/s]. Drives beam broadening.
        Default 0.
    beamwidth : float
        One-way half-power beamwidth θ_b [rad]. Default 0 (pencil beam).
    geometry_backscatter : 6-tuple
        Backscatter geometry. Default :data:`geom_vert_back`.
    geometry_forward : 6-tuple, optional
        Forward-scatter geometry. Required if you want sK_dp.
    noise : None / "realistic" / True / float / (float, float), optional
        System noise floor. ``None`` (default) disables noise and the
        spectrum is signal-only — :meth:`SpectralResult.collapse_to_bulk`
        then round-trips to the bulk radar observables exactly. Pass
        ``"realistic"`` or ``True`` to use :func:`realistic_noise_floor`
        for the scatterer's wavelength; a scalar for equal H/V noise in
        mm⁶ m⁻³; or a 2-tuple ``(noise_h, noise_v)``.
        Noise is distributed uniformly across ``v_bins``, added to
        ``sZ_h`` / ``sZ_v``, and biases ``sZ_dr`` / ``sρ_hv`` / ``sLDR``
        through per-bin SNR. The underlying ``S_spec`` and ``Z_spec``
        stay signal-only.
    """

    _sigma_beam_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))

    def __init__(
        self,
        source: Union[Scatterer, HydroMix],
        fall_speed: Optional[Callable] = None,
        turbulence=None,
        *,
        component_kinematics: Optional[Mapping] = None,
        v_bins: Optional[np.ndarray] = None,
        v_min: Optional[float] = None,
        v_max: Optional[float] = None,
        n_bins: Optional[int] = None,
        w: float = 0.0,
        u_h: float = 0.0,
        beamwidth: float = 0.0,
        geometry_backscatter: Tuple = geom_vert_back,
        geometry_forward: Optional[Tuple] = None,
        noise=None,
    ):
        self.source = source
        self.w = float(w)
        self.u_h = float(abs(u_h))
        self.beamwidth = float(beamwidth)
        self.geometry_backscatter = tuple(geometry_backscatter)
        self.geometry_forward = (
            tuple(geometry_forward) if geometry_forward is not None else None
        )
        self._noise_spec = noise  # resolved lazily in run() once λ is known

        # --- velocity grid ---
        v_triple = (v_min, v_max, n_bins)
        triple_given = any(x is not None for x in v_triple)
        if v_bins is None and not triple_given:
            raise ValueError(
                "Pass either `v_bins` or all three of `v_min`, `v_max`, `n_bins`."
            )
        if v_bins is not None and triple_given:
            raise ValueError(
                "Pass only one of `v_bins` or `(v_min, v_max, n_bins)`."
            )
        if v_bins is not None:
            self.v_bins = np.asarray(v_bins, dtype=float).ravel()
        else:
            if None in v_triple:
                raise ValueError(
                    "`v_min`, `v_max`, `n_bins` must all be supplied together."
                )
            self.v_bins = np.linspace(float(v_min), float(v_max), int(n_bins))
        if self.v_bins.size < 2:
            raise ValueError("v_bins must have at least 2 points.")

        # --- beam-broadening variance ---
        self.sigma_beam = self.u_h * self.beamwidth / self._sigma_beam_fwhm

        # --- resolve per-component kinematics ---
        if isinstance(source, HydroMix):
            if component_kinematics is None:
                raise ValueError(
                    "HydroMix sources require `component_kinematics` mapping "
                    "component label -> (fall_speed, turbulence)."
                )
            if fall_speed is not None or turbulence is not None:
                raise ValueError(
                    "Pass `component_kinematics` for HydroMix; do not also "
                    "pass single-species `fall_speed`/`turbulence`."
                )
            self._component_kin = self._resolve_component_kinematics(
                source, component_kinematics
            )
            self._single_kin = None
        else:
            if fall_speed is None:
                raise ValueError("Single-species source requires `fall_speed`.")
            self._single_kin = (fall_speed, _normalise_turbulence(turbulence))
            self._component_kin = None

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_component_kinematics(mix: HydroMix, mapping: Mapping):
        by_key: Dict[int, Tuple[Callable, Callable]] = {}
        labels = [c.label for c in mix.components]
        # Accept either label-keyed or index-keyed mappings.
        for key, val in mapping.items():
            if not (isinstance(val, tuple) and len(val) == 2):
                raise ValueError(
                    f"component_kinematics[{key!r}] must be (fall_speed, turbulence)."
                )
            fs, turb = val
            turb = _normalise_turbulence(turb)
            if isinstance(key, int):
                idx = key
            else:
                if key not in labels:
                    raise ValueError(
                        f"component_kinematics refers to label {key!r} not "
                        f"in mixture labels {labels!r}."
                    )
                idx = labels.index(key)
            by_key[idx] = (fs, turb)
        missing = set(range(len(mix.components))) - set(by_key)
        if missing:
            missing_labels = [labels[i] for i in sorted(missing)]
            raise ValueError(
                "component_kinematics missing entries for: "
                f"{missing_labels!r}."
            )
        return [by_key[i] for i in range(len(mix.components))]

    # ------------------------------------------------------------------
    def _spectra_for_component(
        self,
        scatterer: Scatterer,
        psd: PSD,
        fall_speed_fn: Callable,
        turb_fn: Callable,
        geom: Tuple,
        table_key: str,
    ) -> np.ndarray:
        """Build the spectral matrix (complex or real) for one component.

        Returns an array of shape ``(P, Q, M)`` where ``(P, Q)`` is
        ``(2, 2)`` for the S table and ``(4, 4)`` for Z.
        """
        integ = scatterer.psd_integrator
        if integ is None or getattr(integ, "_S_table", None) is None:
            raise ValueError(
                "Component scatterer lacks an initialised PSDIntegrator."
            )
        if geom not in integ.geometries:
            raise ValueError(
                f"Geometry {geom!r} is not registered on this component. "
                "Include it in PSDIntegrator.geometries before "
                "init_scatter_table."
            )
        D = integ._psd_D  # shape (N_D,)
        table = getattr(integ, f"_{table_key}_table")[geom]  # (P, Q, N_D)
        N_D = np.atleast_1d(psd(D))  # shape (N_D,)
        v_t = np.atleast_1d(np.asarray(fall_speed_fn(D), dtype=float))
        sigma_t = np.atleast_1d(np.asarray(turb_fn(D), dtype=float))
        sigma_eff = np.sqrt(sigma_t ** 2 + self.sigma_beam ** 2)  # (N_D,)

        # Expected velocity for each diameter (positive-downward).
        v_exp = v_t + self.w  # (N_D,)

        v = self.v_bins  # (M,)
        # Per-bin kernel K(v_k, D_i), shape (M, N_D).
        diff = v[:, None] - v_exp[None, :]
        # σ_eff == 0 → delta: accumulate into nearest bin with weight
        # 1 / local bin width so the integral is preserved.
        K = np.zeros((v.size, D.size), dtype=float)
        zero_sigma = sigma_eff <= 0
        if np.any(~zero_sigma):
            s = sigma_eff[~zero_sigma]
            K[:, ~zero_sigma] = (
                1.0 / (np.sqrt(2.0 * np.pi) * s)
            ) * np.exp(-0.5 * (diff[:, ~zero_sigma] / s) ** 2)
        if np.any(zero_sigma):
            # Bin-centered delta: find nearest v_bin, compute local width.
            # Use np.clip to handle edges; local width = half-distance sum.
            dv_edge = np.diff(v)
            local_width = np.empty_like(v)
            local_width[1:-1] = 0.5 * (dv_edge[:-1] + dv_edge[1:])
            local_width[0] = dv_edge[0]
            local_width[-1] = dv_edge[-1]
            for idx_D in np.where(zero_sigma)[0]:
                k = int(np.argmin(np.abs(v - v_exp[idx_D])))
                if local_width[k] > 0:
                    K[k, idx_D] = 1.0 / local_width[k]

        # Trapezoidal weights along D.
        w_D = np.empty_like(D)
        dD = np.diff(D)
        w_D[1:-1] = 0.5 * (dD[:-1] + dD[1:])
        w_D[0] = 0.5 * dD[0]
        w_D[-1] = 0.5 * dD[-1]

        # Build combined weight[k, D] = N(D) * K(k, D) * trap_w(D).
        weight = K * (N_D * w_D)[None, :]  # (M, N_D)

        # Contract over the diameter axis.
        # table has shape (P, Q, N_D); weight has shape (M, N_D).
        # Result shape (P, Q, M).
        return np.einsum("pqd,md->pqm", table, weight)

    # ------------------------------------------------------------------
    def _range_warning(self):
        """Emit UserWarning if the v_bins grid is too narrow to hold the
        bulk of the spectral power."""
        # Determine expected min/max of v_exp ± 3 σ_eff across all components.
        lo_all, hi_all = [], []
        if self._single_kin is not None:
            sc = self.source
            integ = sc.psd_integrator
            D = integ._psd_D if integ is not None else np.array([1.0])
            fs, turb = self._single_kin
            v_exp = fs(D) + self.w
            sigma_eff = np.sqrt(turb(D) ** 2 + self.sigma_beam ** 2)
            lo_all.append(np.min(v_exp - 3 * sigma_eff))
            hi_all.append(np.max(v_exp + 3 * sigma_eff))
        else:
            for comp, (fs, turb) in zip(self.source.components, self._component_kin):
                integ = comp.scatterer.psd_integrator
                D = integ._psd_D
                v_exp = fs(D) + self.w
                sigma_eff = np.sqrt(turb(D) ** 2 + self.sigma_beam ** 2)
                lo_all.append(np.min(v_exp - 3 * sigma_eff))
                hi_all.append(np.max(v_exp + 3 * sigma_eff))
        lo, hi = min(lo_all), max(hi_all)
        v_lo, v_hi = self.v_bins[0], self.v_bins[-1]
        if lo < v_lo or hi > v_hi:
            warnings.warn(
                "Spectral power extends beyond v_bins: expected range "
                f"[{lo:.3g}, {hi:.3g}] m/s, v_bins covers "
                f"[{v_lo:.3g}, {v_hi:.3g}] m/s. Bulk-sum identity will be "
                "degraded by the leakage.",
                UserWarning,
                stacklevel=3,
            )

    # ------------------------------------------------------------------
    def run(self) -> SpectralResult:
        """Evaluate the spectrum and derived observables."""
        self._range_warning()

        geom_b = self.geometry_backscatter
        geom_f = self.geometry_forward
        M = self.v_bins.size

        if self._single_kin is not None:
            sc = self.source
            fs, turb = self._single_kin
            S_spec = self._spectra_for_component(
                sc, sc.psd, fs, turb, geom_b, "S"
            )
            Z_spec = self._spectra_for_component(
                sc, sc.psd, fs, turb, geom_b, "Z"
            )
            if geom_f is not None:
                S_spec_f = self._spectra_for_component(
                    sc, sc.psd, fs, turb, geom_f, "S"
                )
            else:
                S_spec_f = None
            wavelength = sc.wavelength
            Kw_sqr = sc.Kw_sqr
        else:
            mix: HydroMix = self.source
            S_spec = np.zeros((2, 2, M), dtype=complex)
            Z_spec = np.zeros((4, 4, M), dtype=float)
            S_spec_f = (
                np.zeros((2, 2, M), dtype=complex) if geom_f is not None else None
            )
            for comp, (fs, turb) in zip(mix.components, self._component_kin):
                S_spec += self._spectra_for_component(
                    comp.scatterer, comp.psd, fs, turb, geom_b, "S"
                )
                Z_spec += self._spectra_for_component(
                    comp.scatterer, comp.psd, fs, turb, geom_b, "Z"
                )
                if geom_f is not None:
                    S_spec_f += self._spectra_for_component(
                        comp.scatterer, comp.psd, fs, turb, geom_f, "S"
                    )
            wavelength = mix.wavelength
            Kw_sqr = mix.Kw_sqr

        # Derive per-bin observables.
        sZ_h = np.empty(M)
        sZ_v = np.empty(M)
        sZ_dr = np.empty(M)
        srho_hv = np.empty(M)
        sdelta_hv = np.empty(M)
        sLDR = np.empty(M)
        sKdp = np.empty(M) if geom_f is not None else None

        pref = wavelength ** 4 / (np.pi ** 5 * Kw_sqr)

        # Resolve system-noise spec now that λ is known, and spread it
        # uniformly across the velocity grid (units: mm⁶ m⁻³ / (m/s)).
        noise_h_total, noise_v_total = _resolve_noise(self._noise_spec, wavelength)
        v_span = float(self.v_bins[-1] - self.v_bins[0])
        noise_psd_h = noise_h_total / v_span if v_span > 0 else 0.0
        noise_psd_v = noise_v_total / v_span if v_span > 0 else 0.0

        for k in range(M):
            Z = Z_spec[:, :, k]
            sigma_h = 2.0 * np.pi * (Z[0, 0] - Z[0, 1] - Z[1, 0] + Z[1, 1])
            sigma_v = 2.0 * np.pi * (Z[0, 0] + Z[0, 1] + Z[1, 0] + Z[1, 1])
            signal_h = pref * sigma_h
            signal_v = pref * sigma_v
            sZ_h[k] = signal_h + noise_psd_h
            sZ_v[k] = signal_v + noise_psd_v
            # Z_dr including noise bias (symmetric if noise_h == noise_v).
            if sZ_v[k] != 0:
                sZ_dr[k] = sZ_h[k] / sZ_v[k]
            else:
                sZ_dr[k] = np.nan
            a = (Z[2, 2] + Z[3, 3]) ** 2 + (Z[3, 2] - Z[2, 3]) ** 2
            b = Z[0, 0] - Z[0, 1] - Z[1, 0] + Z[1, 1]
            c = Z[0, 0] + Z[0, 1] + Z[1, 0] + Z[1, 1]
            if (b * c) > 0:
                rho_signal = np.sqrt(a / (b * c))
            else:
                rho_signal = np.nan
            # Noise biases ρ_hv by the per-bin SNR: ρ_obs = ρ_sig /
            # sqrt((1+1/SNR_h)(1+1/SNR_v)). With zero noise this reduces
            # to ρ_signal exactly.
            if noise_psd_h > 0 or noise_psd_v > 0:
                snr_h = signal_h / noise_psd_h if noise_psd_h > 0 else np.inf
                snr_v = signal_v / noise_psd_v if noise_psd_v > 0 else np.inf
                bias = np.sqrt((1 + 1.0 / snr_h) * (1 + 1.0 / snr_v))
                srho_hv[k] = rho_signal / bias if np.isfinite(bias) else rho_signal
            else:
                srho_hv[k] = rho_signal
            sdelta_hv[k] = np.arctan2(Z[2, 3] - Z[3, 2], -Z[2, 2] - Z[3, 3])
            num_ldr = Z[0, 0] - Z[0, 1] + Z[1, 0] - Z[1, 1]
            den_ldr = Z[0, 0] - Z[0, 1] - Z[1, 0] + Z[1, 1]
            # LDR: noise adds to the cross-pol (num) and co-pol (den) in
            # proportion to each channel's noise — same formula as sZ_dr.
            ldr_num = pref * 2.0 * np.pi * num_ldr + noise_psd_v
            ldr_den = pref * 2.0 * np.pi * den_ldr + noise_psd_h
            sLDR[k] = ldr_num / ldr_den if ldr_den != 0 else np.nan
            if sKdp is not None:
                Sf = S_spec_f[:, :, k]
                sKdp[k] = 1e-3 * (180.0 / np.pi) * wavelength * (Sf[1, 1] - Sf[0, 0]).real

        return SpectralResult(
            v=self.v_bins.copy(),
            S_spec=S_spec,
            Z_spec=Z_spec,
            sZ_h=sZ_h,
            sZ_v=sZ_v,
            sZ_dr=sZ_dr,
            sKdp=sKdp,
            srho_hv=srho_hv,
            sdelta_hv=sdelta_hv,
            sLDR=sLDR,
            wavelength=wavelength,
            Kw_sqr=Kw_sqr,
            noise_h=noise_h_total,
            noise_v=noise_v_total,
            _S_spec_forward=S_spec_f,
        )


__all__ = [
    "fall_speed",
    "turbulence",
    "SpectralIntegrator",
    "SpectralResult",
    "NoTurbulence",
    "GaussianTurbulence",
    "InertialZeng2023",
    "atlas_srivastava_sekhon_1973",
    "brandes_et_al_2002",
    "beard_1976",
    "locatelli_hobbs_1974_aggregates",
    "locatelli_hobbs_1974_graupel_hex",
    "power_law",
    "realistic_noise_floor",
    "REALISTIC_NOISE_DBZ",
]
