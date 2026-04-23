"""Radar beam patterns and scene-integrated spectra.

When a radar beam has non-zero width the observed Doppler spectrum is a
solid-angle weighted sum of per-pixel spectra. For a homogeneous scene
across the beam, turbulence and fall speeds independent of position,
and horizontal wind in the beam this reduces to the closed-form

    σ_beam = |u_h| · θ_b / (2 √(2 ln 2))

that :class:`~rustmatrix.spectra.SpectralIntegrator` already builds in
through its ``u_h`` / ``beamwidth`` arguments. For a **heterogeneous**
scene — isolated convective cells, updraft/downdraft couplets,
horizontal reflectivity gradients — the analytic formula fails: the
beam samples regions with different reflectivities, vertical motions,
and wind speeds, and the observed spectrum is a genuine spatial
integral.

This module provides the pieces needed for that integral:

* :class:`BeamPattern` base and three concrete subclasses:

  - :class:`GaussianBeam` — canonical well-tapered main-lobe; no
    sidelobes.
  - :class:`AiryBeam` — uniform circular aperture pattern
    ``[2 J₁(x)/x]²`` with realistic sidelobes (first at −17.6 dB).
  - :class:`TabulatedBeam` — user-supplied ``(θ, gain)`` samples, for
    measured patterns or Taylor tapers.

* :class:`Scene` — spatial fields ``Z_dBZ(x,y,z)``, ``w(x,y,z)``,
  ``u_h(x,y,z)`` that the integrator evaluates at each beam sample.

* :class:`BeamIntegrator` — drives a scatterer + beam + scene + PSD
  factory through a Doppler-velocity grid and returns a
  :class:`~rustmatrix.spectra.SpectralResult`.

Coordinate conventions
----------------------
Radar at the origin, boresight along ``-ẑ`` (down-looking) is the
primary use case. A beam sample offset by ``(θ, φ)`` from boresight
points along ``(sin θ cos φ, sin θ sin φ, -cos θ)``. The line-of-sight
Doppler of a particle with velocity ``(u_x, u_y, -v_fall)`` — where
``v_fall = v_t(D) + w`` is the downward fall rate — is

    v_obs = (v_t + w) cos θ + u_h sin θ cos(φ − φ_wind)

which collapses to ``v_t + w`` at boresight (``θ = 0``). Positive
``v_obs`` means the particle is moving **away from a down-looking
radar**, i.e. in the fall direction — same convention as the rest of
:mod:`rustmatrix.spectra`.

References
----------
Doviak, R. J. & Zrnić, D. S. (1993). *Doppler Radar and Weather
Observations*, 2nd ed., Academic Press. §3.2, §5.3.

Balanis, C. A. (2016). *Antenna Theory: Analysis and Design*, 4th ed.,
Wiley. §15.4 (circular aperture Airy pattern).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

try:
    from numpy import trapezoid as _trapezoid
except ImportError:  # numpy < 2.0
    from numpy import trapz as _trapezoid

from ..scatterer import Scatterer
from ..tmatrix_aux import geom_vert_back


# ---------------------------------------------------------------------------
# Bessel J1 (polynomial approximation, A&S 9.4.4 / 9.4.6)
# ---------------------------------------------------------------------------

def _j1(x):
    """Bessel function of the first kind, order 1.

    Uses the Abramowitz & Stegun polynomial approximations (9.4.4 for
    |x| ≤ 3, 9.4.6 for |x| > 3), accurate to ~1e-7. Implemented in pure
    numpy so the beam module carries no scipy dependency.
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) <= 3.0
    # --- |x| ≤ 3: J1(x) = x * polynomial in (x/3)² (A&S 9.4.4) ---
    xs = x[small]
    t = (xs / 3.0) ** 2
    poly_small = (
        0.5
        - 0.56249985 * t
        + 0.21093573 * t ** 2
        - 0.03954289 * t ** 3
        + 0.00443319 * t ** 4
        - 0.00031761 * t ** 5
        + 0.00001109 * t ** 6
    )
    out[small] = xs * poly_small
    # --- |x| > 3: asymptotic form (A&S 9.4.6) ---
    xl = x[~small]
    sign = np.sign(xl)
    xa = np.abs(xl)
    y = 3.0 / xa
    f1 = (
        0.79788456
        + 0.00000156 * y
        + 0.01659667 * y ** 2
        + 0.00017105 * y ** 3
        - 0.00249511 * y ** 4
        + 0.00113653 * y ** 5
        - 0.00020033 * y ** 6
    )
    theta1 = (
        xa - 0.75 * np.pi
        + 0.12499612 * y
        + 0.00005650 * y ** 2
        - 0.00637879 * y ** 3
        + 0.00074348 * y ** 4
        + 0.00079824 * y ** 5
        - 0.00029166 * y ** 6
    )
    out[~small] = sign * f1 * np.cos(theta1) / np.sqrt(xa)
    return out


# ---------------------------------------------------------------------------
# Beam patterns
# ---------------------------------------------------------------------------

class BeamPattern:
    """Base class for circularly symmetric one-way power patterns.

    Subclasses define :meth:`gain`, the normalized one-way power
    pattern ``G(θ)`` with ``G(0) = 1``. :meth:`sample` returns a set of
    ``(θ, φ, weight)`` triples suitable for driving a
    :class:`BeamIntegrator`; default weights are proportional to
    ``G²(θ) sin θ`` (two-way pattern × solid-angle element) and sum to
    1 over the sampled cone.

    Attributes
    ----------
    hpbw : float
        One-way half-power full-width [rad].
    """

    hpbw: float

    def gain(self, theta):
        raise NotImplementedError

    def sample(
        self,
        n_theta: int = 32,
        n_phi: int = 24,
        max_theta: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(theta, phi, weight)`` flat arrays on the beam cone.

        Parameters
        ----------
        n_theta, n_phi : int
            Number of polar / azimuthal samples. The default 32 × 24 is
            generally enough to recover the analytic σ_beam to ~1 %.
        max_theta : float, optional
            Outer cone half-angle [rad]. Default ``3 × hpbw`` —
            captures >99.9 % of a Gaussian two-way pattern and a few
            Airy sidelobes.

        Notes
        -----
        Weights are two-way, ``G²(θ) sin θ``, normalized so they sum
        to 1. The line-of-sight velocity projection is the *only*
        thing that knows about φ, so for a circularly symmetric beam
        the azimuth integration is exactly Simpson's rule on a
        periodic grid.
        """
        if max_theta is None:
            max_theta = 3.0 * self.hpbw
        theta_1d = np.linspace(0.0, max_theta, n_theta)
        phi_1d = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
        TH, PH = np.meshgrid(theta_1d, phi_1d, indexing="ij")
        G = self.gain(TH)
        dtheta = theta_1d[1] - theta_1d[0]
        dphi = 2.0 * np.pi / n_phi
        w = (G ** 2) * np.sin(TH) * dtheta * dphi
        total = w.sum()
        if total > 0:
            w = w / total
        return TH.ravel(), PH.ravel(), w.ravel()

    def __repr__(self):
        return f"{type(self).__name__}(hpbw={self.hpbw:.4g} rad)"


class GaussianBeam(BeamPattern):
    """Gaussian one-way power pattern.

    ``G(θ) = exp(-θ² / (2 σ²))`` with ``σ = hpbw / (2 √(2 ln 2))``.
    No sidelobes; well-tapered aperture illumination. This is the
    pattern implicit in :class:`~rustmatrix.spectra.SpectralIntegrator`
    when ``beamwidth > 0`` is supplied.
    """

    _FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))

    def __init__(self, hpbw: float):
        self.hpbw = float(hpbw)
        self.sigma = self.hpbw / self._FWHM

    def gain(self, theta):
        theta = np.asarray(theta, dtype=float)
        return np.exp(-0.5 * (theta / self.sigma) ** 2)


class AiryBeam(BeamPattern):
    """Uniformly illuminated circular aperture (Airy) pattern.

    ``G(θ) = [2 J₁(x) / x]²`` with ``x = α sin θ`` and ``α`` chosen so
    the pattern crosses half-power at ``θ = hpbw / 2``. Gives the
    canonical parabolic-dish response — a main lobe slightly narrower
    than a matched-HPBW Gaussian, followed by the first sidelobe at
    **−17.6 dB** and nulls between lobes.

    Parameters
    ----------
    hpbw : float
        One-way half-power beamwidth [rad].
    """

    #: Argument x where ``[2 J₁(x)/x]² = 0.5``.
    X_HALFPOWER = 1.6163399

    def __init__(self, hpbw: float):
        self.hpbw = float(hpbw)
        self.alpha = self.X_HALFPOWER / np.sin(self.hpbw / 2.0)

    def gain(self, theta):
        theta = np.asarray(theta, dtype=float)
        x = self.alpha * np.sin(theta)
        out = np.ones_like(x)
        nz = x != 0.0
        out[nz] = (2.0 * _j1(x[nz]) / x[nz]) ** 2
        return out


class TabulatedBeam(BeamPattern):
    """Beam pattern interpolated from a user-supplied table.

    Parameters
    ----------
    theta : array_like
        Polar-angle samples [rad], strictly increasing and starting at 0.
    gain : array_like
        One-way power pattern at those angles, ``gain[0] = 1`` assumed.
    hpbw : float, optional
        Advertised one-way half-power beamwidth [rad]. If omitted it is
        inferred from the table by linear interpolation at ``G = 0.5``.
    """

    def __init__(
        self,
        theta,
        gain,
        hpbw: Optional[float] = None,
    ):
        theta = np.asarray(theta, dtype=float).ravel()
        gain_a = np.asarray(gain, dtype=float).ravel()
        if theta.shape != gain_a.shape:
            raise ValueError("theta and gain must have the same length.")
        if theta[0] != 0.0 or np.any(np.diff(theta) <= 0):
            raise ValueError("theta must start at 0 and be strictly increasing.")
        self._theta = theta
        self._gain = gain_a
        if hpbw is None:
            below = np.where(gain_a < 0.5)[0]
            if below.size == 0:
                raise ValueError(
                    "Cannot infer HPBW — gain table never crosses 0.5."
                )
            i = below[0]
            # linear interpolate between i-1 and i
            th0, th1 = theta[i - 1], theta[i]
            g0, g1 = gain_a[i - 1], gain_a[i]
            frac = (0.5 - g0) / (g1 - g0)
            hpbw = 2.0 * (th0 + frac * (th1 - th0))
        self.hpbw = float(hpbw)

    def gain(self, theta):
        theta = np.asarray(theta, dtype=float)
        g = np.interp(np.abs(theta), self._theta, self._gain,
                      left=self._gain[0], right=0.0)
        return g


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

Field = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


@dataclass
class Scene:
    """Spatial fields evaluated at beam-sample positions.

    Each field is a callable ``f(x, y, z) -> array`` that accepts
    numpy arrays of shape ``(N,)`` (positions of a batch of beam
    samples) and returns an ``(N,)`` array.

    Attributes
    ----------
    Z_dBZ : callable
        Equivalent reflectivity field [dBZ].
    w : callable
        Mean vertical air motion [m/s], positive-downward.
    u_h : callable
        Horizontal wind speed magnitude [m/s].
    u_h_azimuth : float or callable
        Horizontal wind azimuth [rad], measured from +x axis. If a
        scalar, treated as constant; if callable, evaluated like the
        other fields. Default 0 (wind blows in +x direction).
    """

    Z_dBZ: Field
    w: Field
    u_h: Field
    u_h_azimuth: object = 0.0  # float or callable

    def evaluate(self, x, y, z):
        """Return ``(Z_dBZ, w, u_h, u_h_azimuth)`` at each (x, y, z).

        All inputs and outputs are 1-D arrays of the same length.
        """
        Z = np.asarray(self.Z_dBZ(x, y, z), dtype=float)
        w = np.asarray(self.w(x, y, z), dtype=float)
        u = np.asarray(self.u_h(x, y, z), dtype=float)
        if callable(self.u_h_azimuth):
            phi_w = np.asarray(self.u_h_azimuth(x, y, z), dtype=float)
        else:
            phi_w = np.full_like(x, float(self.u_h_azimuth))
        return Z, w, u, phi_w


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def marshall_palmer_psd_factory(
    N0: float = 8000.0,
    D_max: float = 6.0,
    Z_floor_dBZ: float = -30.0,
):
    """Return a PSD factory that maps dBZ to a Marshall–Palmer exponential PSD.

    Uses the Rayleigh-equivalent Z-Λ relation for an exponential PSD,
    ``Z ≈ 720 · N₀ / Λ⁷`` (mm⁶ m⁻³ with D in mm). This is a
    self-consistent analytic inversion — accurate at rain-band
    wavelengths where Rayleigh scattering dominates — and avoids a
    root-finding step at each pixel. Below ``Z_floor_dBZ`` the factory
    returns a PSD with effectively zero concentration (``Λ`` very
    large), so empty pixels contribute nothing.

    Parameters
    ----------
    N0 : float
        Intercept parameter [mm⁻¹ m⁻³]. Default 8000 (Marshall–Palmer).
    D_max : float
        Upper integration cutoff [mm].
    Z_floor_dBZ : float
        Reflectivities below this are treated as "empty".

    Returns
    -------
    factory : callable
        ``factory(Z_dBZ) -> ExponentialPSD`` suitable as the
        :class:`BeamIntegrator` PSD mapper.
    """
    from ..psd import ExponentialPSD

    def _factory(Z_dBZ):
        Z_dBZ = float(Z_dBZ)
        if Z_dBZ < Z_floor_dBZ:
            return ExponentialPSD(N0=N0, Lambda=1e6, D_max=D_max)
        Z_lin = 10.0 ** (Z_dBZ / 10.0)
        Lambda = (720.0 * N0 / Z_lin) ** (1.0 / 7.0)
        return ExponentialPSD(N0=N0, Lambda=Lambda, D_max=D_max)

    _factory.N0 = N0
    _factory.D_max = D_max
    return _factory


# ---------------------------------------------------------------------------
# BeamIntegrator
# ---------------------------------------------------------------------------

class BeamIntegrator:
    """Integrate a spectral S/Z response over a beam pattern + scene.

    Parameters
    ----------
    scatterer : Scatterer
        Single-species scatterer with a :class:`PSDIntegrator` already
        populated for the backscatter (and, optionally, forward)
        geometry. The per-diameter ``_S_table`` / ``_Z_table`` are
        reused at every beam sample — only the PSD weights change.
    beam : BeamPattern
        Beam pattern whose :meth:`~BeamPattern.sample` method produces
        the angular quadrature grid.
    scene : Scene
        Scene fields evaluated at each beam-sample pixel.
    psd_factory : callable
        ``Z_dBZ -> PSD`` mapping used to build the per-pixel PSD.
    fall_speed : callable
        ``D_mm -> v_t_m_per_s``, same contract as
        :class:`~rustmatrix.spectra.SpectralIntegrator`.
    turbulence : callable or _TurbulenceModel, optional
        Per-pixel turbulence. Default zero turbulence.
    radar_position : 3-tuple
        ``(x, y, z)`` of the radar [m]. Default ``(0, 0, 0)``.
    boresight : 3-tuple
        Unit vector along the beam axis. Default ``(0, 0, -1)``
        (down-looking).
    range_m : float
        Slant range to the range-gate centre [m].
    v_bins : array_like, optional
        Explicit 1-D velocity grid [m/s]. Mutually exclusive with
        ``(v_min, v_max, n_bins)``.
    v_min, v_max, n_bins : float, float, int, optional
        Convenience triple.
    n_theta, n_phi : int
        Beam-sample counts. Defaults 32 × 24.
    max_theta_over_hpbw : float
        Outer sampling cone in multiples of ``hpbw``. Default 3.
    geometry_backscatter, geometry_forward : 6-tuples
        Same meaning as :class:`~rustmatrix.spectra.SpectralIntegrator`.

    Notes
    -----
    This integrator is **single-species**. For a mixture, run one
    integrator per species and add the resulting ``S_spec`` / ``Z_spec``
    on the shared velocity grid (the same linearity argument as
    :class:`~rustmatrix.hd_mix.HydroMix`).
    """

    def __init__(
        self,
        scatterer: Scatterer,
        beam: BeamPattern,
        scene: Scene,
        psd_factory: Callable,
        fall_speed: Callable,
        turbulence=None,
        *,
        radar_position: Sequence[float] = (0.0, 0.0, 0.0),
        boresight: Sequence[float] = (0.0, 0.0, -1.0),
        range_m: float = 1.0,
        v_bins=None,
        v_min: Optional[float] = None,
        v_max: Optional[float] = None,
        n_bins: Optional[int] = None,
        n_theta: int = 32,
        n_phi: int = 24,
        max_theta_over_hpbw: float = 3.0,
        geometry_backscatter: Tuple = geom_vert_back,
        geometry_forward: Optional[Tuple] = None,
    ):
        # Defer import of sibling module to avoid circular import.
        from . import (
            SpectralResult,
            _normalise_turbulence,
            _resolve_noise,
        )
        self._SpectralResult = SpectralResult
        self._resolve_noise = _resolve_noise
        self._turb = _normalise_turbulence(turbulence)
        self.scatterer = scatterer
        self.beam = beam
        self.scene = scene
        self.psd_factory = psd_factory
        self.fall_speed = fall_speed
        self.radar_position = np.asarray(radar_position, dtype=float)
        bs = np.asarray(boresight, dtype=float)
        bs = bs / np.linalg.norm(bs)
        self.boresight = bs
        self.range_m = float(range_m)
        self.n_theta = int(n_theta)
        self.n_phi = int(n_phi)
        self.max_theta = max_theta_over_hpbw * beam.hpbw
        self.geometry_backscatter = tuple(geometry_backscatter)
        self.geometry_forward = (
            tuple(geometry_forward) if geometry_forward is not None else None
        )

        # velocity grid
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
            self.v_bins = np.linspace(float(v_min), float(v_max), int(n_bins))

        # Pre-compute a basis for off-boresight rotation. For boresight
        # along -ẑ the basis is (x̂, ŷ, -ẑ); otherwise we pick any
        # perpendicular pair.
        self._e3 = bs  # along-axis
        if abs(bs[2]) > 0.999:
            self._e1 = np.array([1.0, 0.0, 0.0])
        else:
            self._e1 = np.cross(bs, np.array([0.0, 0.0, 1.0]))
            self._e1 /= np.linalg.norm(self._e1)
        self._e2 = np.cross(self._e3, self._e1)

    # ------------------------------------------------------------------
    def _ray_direction(self, theta, phi):
        """Unit vector along a beam sample `(theta, phi)` in world coords."""
        st, ct = np.sin(theta), np.cos(theta)
        cp, sp = np.cos(phi), np.sin(phi)
        # rotation: n = sin θ cos φ · ê1 + sin θ sin φ · ê2 + cos θ · ê3
        e1 = self._e1
        e2 = self._e2
        e3 = self._e3
        nx = st * cp * e1[0] + st * sp * e2[0] + ct * e3[0]
        ny = st * cp * e1[1] + st * sp * e2[1] + ct * e3[1]
        nz = st * cp * e1[2] + st * sp * e2[2] + ct * e3[2]
        return nx, ny, nz

    # ------------------------------------------------------------------
    def run(self):
        """Evaluate the beam-integrated spectrum.

        Returns
        -------
        SpectralResult
            Same container used by
            :class:`~rustmatrix.spectra.SpectralIntegrator`. All
            bulk-sum identities still hold (the spectrum is linear in
            the beam weights and per-pixel PSD).
        """
        sc = self.scatterer
        integ = sc.psd_integrator
        if integ is None or getattr(integ, "_S_table", None) is None:
            raise ValueError(
                "Scatterer lacks an initialised PSDIntegrator with S/Z tables."
            )
        geom_b = self.geometry_backscatter
        geom_f = self.geometry_forward
        if geom_b not in integ.geometries:
            raise ValueError(f"Geometry {geom_b!r} not on scatterer.")
        if geom_f is not None and geom_f not in integ.geometries:
            raise ValueError(f"Forward geometry {geom_f!r} not on scatterer.")

        S_tab_b = integ._S_table[geom_b]  # (2, 2, N_D)
        Z_tab_b = integ._Z_table[geom_b]  # (4, 4, N_D)
        S_tab_f = integ._S_table[geom_f] if geom_f is not None else None
        D = integ._psd_D

        theta_s, phi_s, weights = self.beam.sample(
            n_theta=self.n_theta,
            n_phi=self.n_phi,
            max_theta=self.max_theta,
        )

        # Pixel positions for all samples.
        nx, ny, nz = self._ray_direction(theta_s, phi_s)
        rp = self.radar_position
        px = rp[0] + self.range_m * nx
        py = rp[1] + self.range_m * ny
        pz = rp[2] + self.range_m * nz

        Z_pix, w_pix, u_pix, phi_w_pix = self.scene.evaluate(px, py, pz)

        v_t = np.asarray(self.fall_speed(D), dtype=float)
        sigma_t = np.asarray(self._turb(D), dtype=float)

        M = self.v_bins.size
        S = theta_s.size

        # --- per-pixel PSD values ----------------------------------------
        # Factory call still loops, but ExponentialPSD evaluation is cheap.
        N_D_pix = np.empty((S, D.size), dtype=float)
        active = weights > 0.0
        for i in np.where(active)[0]:
            N_D_pix[i] = self.psd_factory(Z_pix[i])(D)
        N_D_pix[~active] = 0.0

        # --- per-sample expected velocity --------------------------------
        ct = np.cos(theta_s)[:, None]          # (S, 1)
        st = np.sin(theta_s)[:, None]          # (S, 1)
        cos_dphi = np.cos(phi_s - phi_w_pix)[:, None]
        v_exp = (v_t[None, :] + w_pix[:, None]) * ct + \
                u_pix[:, None] * st * cos_dphi      # (S, N_D)

        # --- shared D-direction trapezoidal weights ----------------------
        w_D = np.empty_like(D)
        dD = np.diff(D)
        w_D[1:-1] = 0.5 * (dD[:-1] + dD[1:])
        w_D[0] = 0.5 * dD[0]
        w_D[-1] = 0.5 * dD[-1]

        # --- velocity-bin local widths ------------------------------------
        v = self.v_bins
        dv_edge = np.diff(v)
        local_width = np.empty_like(v)
        local_width[1:-1] = 0.5 * (dv_edge[:-1] + dv_edge[1:])
        local_width[0] = dv_edge[0]
        local_width[-1] = dv_edge[-1]
        dv_median = float(np.median(local_width))

        # --- vectorized Gaussian kernel ----------------------------------
        # K has shape (S, M, N_D). For S=256, M=384, N_D=48 this is ~37 MB.
        diff_v = v[None, :, None] - v_exp[:, None, :]   # (S, M, N_D)
        sigma_eff = sigma_t                              # (N_D,) — pos-indep
        narrow = sigma_eff < 0.5 * dv_median
        wide = ~narrow

        K = np.zeros_like(diff_v)
        if wide.any():
            s_w = sigma_eff[wide]
            K[:, :, wide] = (1.0 / (np.sqrt(2.0 * np.pi) * s_w)) * np.exp(
                -0.5 * (diff_v[:, :, wide] / s_w) ** 2
            )
        if narrow.any():
            # For narrow-σ diameters, bin power into the nearest v cell
            # per sample. Loop only over the narrow diameters; expected
            # to be few when turbulence is present.
            for d_idx in np.where(narrow)[0]:
                nearest = np.argmin(
                    np.abs(v[None, :] - v_exp[:, d_idx:d_idx + 1]), axis=1
                )   # (S,)
                for i, k in enumerate(nearest):
                    if local_width[k] > 0 and weights[i] > 0:
                        K[i, k, d_idx] = 1.0 / local_width[k]

        # weight[s, k, d] = K[s,k,d] * N_D[s,d] * w_D[d] * beam_weight[s]
        # Collapse over s to get a shared (M, N_D) weight, then apply tables.
        sample_weight = (N_D_pix * w_D[None, :]) * weights[:, None]  # (S, N_D)
        # Combined = sum_s K[s,:,:] * sample_weight[s, :]
        combined = np.einsum("smd,sd->md", K, sample_weight)   # (M, N_D)

        S_spec = np.einsum("pqd,md->pqm", S_tab_b, combined)
        Z_spec = np.einsum("pqd,md->pqm", Z_tab_b, combined)
        if S_tab_f is not None:
            S_spec_f = np.einsum("pqd,md->pqm", S_tab_f, combined)
        else:
            S_spec_f = None

        # Derive observables (copied from SpectralIntegrator.run()).
        wavelength = sc.wavelength
        Kw_sqr = sc.Kw_sqr
        pref = wavelength ** 4 / (np.pi ** 5 * Kw_sqr)
        sZ_h = np.empty(M)
        sZ_v = np.empty(M)
        sZ_dr = np.empty(M)
        srho_hv = np.empty(M)
        sdelta_hv = np.empty(M)
        sLDR = np.empty(M)
        sKdp = np.empty(M) if S_spec_f is not None else None

        for k in range(M):
            Z = Z_spec[:, :, k]
            sigma_h = 2.0 * np.pi * (Z[0, 0] - Z[0, 1] - Z[1, 0] + Z[1, 1])
            sigma_v = 2.0 * np.pi * (Z[0, 0] + Z[0, 1] + Z[1, 0] + Z[1, 1])
            sZ_h[k] = pref * sigma_h
            sZ_v[k] = pref * sigma_v
            sZ_dr[k] = sZ_h[k] / sZ_v[k] if sZ_v[k] != 0 else np.nan
            a = (Z[2, 2] + Z[3, 3]) ** 2 + (Z[3, 2] - Z[2, 3]) ** 2
            b = Z[0, 0] - Z[0, 1] - Z[1, 0] + Z[1, 1]
            c = Z[0, 0] + Z[0, 1] + Z[1, 0] + Z[1, 1]
            srho_hv[k] = np.sqrt(a / (b * c)) if (b * c) > 0 else np.nan
            sdelta_hv[k] = np.arctan2(Z[2, 3] - Z[3, 2], -Z[2, 2] - Z[3, 3])
            num_ldr = Z[0, 0] - Z[0, 1] + Z[1, 0] - Z[1, 1]
            den_ldr = Z[0, 0] - Z[0, 1] - Z[1, 0] + Z[1, 1]
            sLDR[k] = num_ldr / den_ldr if den_ldr != 0 else np.nan
            if sKdp is not None:
                Sf = S_spec_f[:, :, k]
                sKdp[k] = 1e-3 * (180.0 / np.pi) * wavelength * (
                    Sf[1, 1] - Sf[0, 0]
                ).real

        return self._SpectralResult(
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
            _S_spec_forward=S_spec_f,
        )


__all__ = [
    "BeamPattern",
    "GaussianBeam",
    "AiryBeam",
    "TabulatedBeam",
    "Scene",
    "BeamIntegrator",
    "marshall_palmer_psd_factory",
]
