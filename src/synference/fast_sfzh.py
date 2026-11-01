"""Fast, vectorised construction of parametric SFZH grids.

Building a :class:`synthesizer.parametric.Stars` object from an SFH function
and a metallicity distribution is dominated by ``Stars._get_sfzh``, which calls
:func:`scipy.integrate.quad` once per age bin (and once per metallicity bin for
non-delta metallicity distributions). For the 1e5-1e7 galaxies Synference
builds this is the bottleneck.

This module reproduces exactly the same binning as Synthesizer but computes the
mass in each age bin with a single vectorised SFR evaluation:

* The integration domain ``[0, ages[-1]]`` (shifted by ``age_offset``) is split
  at every Synthesizer bin edge, at a cached set of log-spaced sub-nodes, and at
  SFH specific nodes (``min_age``/``max_age`` truncations, internal step
  breakpoints of bin based SFHs, interpolation knots of tabulated SFHs and
  geometrically refined nodes around sharp features such as the peak of a
  narrow log-normal or a short e-folding time close to ``max_age``).
* Every sub-interval is integrated with an ``n``-point Gauss-Legendre rule.
  Gauss-Legendre nodes are strictly interior so step discontinuities placed at
  interval boundaries are integrated exactly, piecewise constant / linear SFHs
  are integrated exactly by the 1-point (midpoint) rule and smooth SFHs to
  ~1e-8 relative accuracy.
* Sub-interval integrals are summed per age bin with ``np.add.reduceat``.

Metallicity weights reproduce Synthesizer bit-for-bit for ``DeltaConstant`` (and
float metallicities) - the same cloud-in-cell assignment as the C extension -
and use the analytic error function integral for ``ZDist.Normal`` (agreeing with
Synthesizer's ``quad`` to ~1e-10). Unknown ZDist classes use the same ``quad``
loop as Synthesizer.

Anything not explicitly supported (instantaneous float SFHs, array SFHs,
unknown/user defined SFH classes, SFH subclasses not registered here, divergent
integrals, non-finite results) automatically falls back to Synthesizer's own
``Stars(sf_hist=..., metal_dist=...)`` path, so :func:`make_parametric_stars`
can be used unconditionally as a drop-in replacement.

Example:
    >>> stars = make_parametric_stars(
    ...     SFH.LogNormal(
    ...         tau=0.3,
    ...         peak_age=200
    ...         * Myr,
    ...         max_age=1
    ...         * Gyr,
    ...     ),
    ...     ZDist.DeltaConstant(
    ...         log10metallicity=-2.0
    ...     ),
    ...     grid.log10ages,
    ...     grid.metallicities,
    ...     initial_mass=1e9
    ...     * Msun,
    ... )
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Optional, Tuple

import numpy as np
from scipy import integrate
from scipy.special import erf, hyp2f1
from synthesizer.parametric import SFH, ZDist
from synthesizer.parametric import Stars as ParametricStars
from unyt import unyt_array, unyt_quantity

__all__ = [
    "sfh_bin_masses",
    "zdist_weights",
    "fast_sfzh",
    "make_parametric_stars",
    "is_supported_sfh",
    "is_supported_zdist",
    "FastParametricStars",
    "age_bin_edges",
]

# Gauss-Legendre order used for smooth SFHs. 6 points integrate polynomials of
# degree 11 exactly; with the node placement below this gives ~1e-9 relative
# accuracy per bin for every smooth parametrisation tested.
GL_ORDER = 6

# Number of log-spaced sub-intervals per Synthesizer age bin in the cached base
# node set (on top of SFH specific feature nodes).
SUB_PER_BIN = 3

# Number of geometric sub-nodes used to resolve the first age bin [0, e_1],
# spanning N_HEAD_DECADES decades below e_1.
N_HEAD_DECADES = 8
N_HEAD_PER_DECADE = 3

_GL_CACHE: dict = {}
_NODE_CACHE: "OrderedDict" = OrderedDict()
_NODE_CACHE_MAX = 32


class _Unsupported(Exception):
    """Raised internally when the fast path cannot handle an input."""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _gl(order: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (cached) Gauss-Legendre nodes and weights on [-1, 1]."""
    if order not in _GL_CACHE:
        _GL_CACHE[order] = np.polynomial.legendre.leggauss(order)
    return _GL_CACHE[order]


def _to_yr(value) -> float:
    """Convert a float (assumed yr) or unyt quantity to a float in yr."""
    if value is None:
        return 0.0
    if isinstance(value, (unyt_quantity, unyt_array)):
        return float(value.to("yr").value)
    return float(value)


def age_bin_edges(log10ages) -> np.ndarray:
    """Return the age bin edges (yr) Synthesizer integrates the SFH over.

    Matches ``synthesizer.parametric.Stars._get_sfzh``: ``[0, midpoints of
    10**log10ages (linear), ages[-1]]``.

    Args:
        log10ages (np.ndarray): log10(age / yr) of the grid.

    Returns:
        np.ndarray: ``len(log10ages) + 1`` bin edges in yr.
    """
    ages = 10 ** np.asarray(log10ages, dtype=np.float64)
    edges = np.empty(ages.size + 1)
    edges[0] = 0.0
    if ages.size > 1:
        # np.mean([a, b]) == (a + b) / 2 == 0.5 * (a + b) bit-for-bit
        edges[1:-1] = 0.5 * (ages[1:] + ages[:-1])
    edges[-1] = ages[-1]
    return edges


def _base_nodes(log10ages) -> Tuple[np.ndarray, np.ndarray]:
    """Cached (edges, base_nodes) for an age grid (relative, no offset)."""
    log10ages = np.asarray(log10ages, dtype=np.float64)
    key = log10ages.tobytes()
    hit = _NODE_CACHE.get(key)
    if hit is not None:
        _NODE_CACHE.move_to_end(key)
        return hit

    edges = age_bin_edges(log10ages)
    pieces = [edges]
    # Log-spaced sub-nodes inside each bin except the first
    lo, hi = edges[1:-1], edges[2:]
    if lo.size and SUB_PER_BIN > 1:
        frac = np.arange(1, SUB_PER_BIN) / SUB_PER_BIN
        good = (lo > 0) & (hi > lo)
        sub = np.exp(np.log(lo[good, None]) + frac[None, :] * np.log(hi[good] / lo[good])[:, None])
        pieces.append(sub.ravel())
    # Geometric refinement of the first bin [0, e_1] down towards 0
    if edges.size > 1 and edges[1] > 0:
        n = N_HEAD_DECADES * N_HEAD_PER_DECADE
        pieces.append(edges[1] * np.logspace(-N_HEAD_DECADES, 0, n + 1))
    nodes = np.unique(np.concatenate(pieces))
    nodes = nodes[(nodes >= edges[0]) & (nodes <= edges[-1])]

    _NODE_CACHE[key] = (edges, nodes)
    if len(_NODE_CACHE) > _NODE_CACHE_MAX:
        _NODE_CACHE.popitem(last=False)
    return edges, nodes


# Precomputed dimensionless node patterns
_GEOM = np.concatenate([[0.0], np.geomspace(1e-4, 60.0, 25)])  # in units of tau
_U_LOGNORMAL = np.linspace(-12.0, 12.0, 49)  # in units of tau (ln t)
_U_GAUSS = np.linspace(-10.0, 10.0, 41)  # in units of sigma
_DPL_LOG = np.logspace(-6.0, 6.0, 97)  # in units of peak_age


def _geom(scale: float) -> np.ndarray:
    """Geometric offsets ``|scale| * [0, geomspace(1e-4, 60)]``."""
    s = abs(scale)
    if not np.isfinite(s) or s == 0:
        return _GEOM[:1]
    return s * _GEOM


# --------------------------------------------------------------------------
# SFH handlers: each returns (extra_nodes, vectorised_sfr, gl_order,
# head_integral) where head_integral is an optional callable giving the exact
# integral over [0, x] for SFHs singular at age 0.
# --------------------------------------------------------------------------
def _h_constant(s, span):
    return np.array([s.min_age, s.max_age]), s._sfrs, 1, None


def _h_gaussian(s, span):
    nodes = np.concatenate([[s.min_age, s.max_age], s.peak_age + s.sigma * _U_GAUSS])
    return nodes, s._sfrs, GL_ORDER, None


def _h_exponential(s, span):
    g = _geom(s.tau)
    nodes = np.concatenate([[s.min_age, s.max_age], s.max_age - g, s.min_age + g])
    return nodes, s._sfrs, GL_ORDER, None


def _h_delayed(s, span):
    g = _geom(s.tau)
    nodes = np.concatenate([[s.min_age, s.max_age], s.max_age - g, s.min_age + g])
    return nodes, s._sfrs, GL_ORDER, None


def _h_lognormal(s, span):
    # In t = max_age - age the SFR is a Gaussian in ln(t) of width tau
    # centred on t_0 (it is the d ln t density); resolve +-12 sigma.
    if not (np.isfinite(s.t_0) and s.tau > 0):
        raise _Unsupported("degenerate LogNormal")
    t = np.exp(s.t_0 + s.tau * _U_LOGNORMAL)
    nodes = np.concatenate([[s.min_age, s.max_age], s.max_age - t])
    return nodes, s._sfrs, GL_ORDER, None


def _h_doublepowerlaw(s, span):
    if not (s.peak_age > 0 and np.isfinite(s.alpha) and np.isfinite(s.beta)):
        raise _Unsupported("degenerate DoublePowerLaw")
    m = min(s.alpha, s.beta)
    head = None
    if s.min_age <= 0:
        if m >= 1:
            # Non-integrable singularity at age = 0: quad's answer is
            # ill-defined, so defer to Synthesizer for identical behaviour.
            raise _Unsupported("DoublePowerLaw diverges at age 0")
        if m > 0:
            p, d = s.peak_age, abs(s.alpha - s.beta)

            def head(x, p=p, m=m, d=d):
                # Exact int_0^x da / ((a/p)^alpha + (a/p)^beta)
                #   = p int_0^Y y^-m / (1 + y^d) dy,  Y = x / p
                y = x / p
                if d == 0:
                    return 0.5 * p * y ** (1 - m) / (1 - m)
                c = (1 - m) / d
                return p * y ** (1 - m) / (1 - m) * hyp2f1(1.0, c, 1.0 + c, -(y**d))

    turn = _GEOM / max(abs(s.alpha), abs(s.beta), 1.0)
    turn = turn[turn < 1.0]  # beyond this the log-spaced nodes suffice
    nodes = np.concatenate(
        [
            [s.min_age, s.max_age, s.peak_age],
            s.peak_age * _DPL_LOG,
            # resolve the turnover at the peak, whose width in ln(age) is
            # ~1/max(alpha, beta) (sharp for steep power laws)
            s.peak_age * np.exp(turn),
            s.peak_age * np.exp(-turn),
        ]
    )
    return nodes, s._sfrs, GL_ORDER, head


def _step_eval_lower(s):
    """Vectorised Continuity/ContinuityFlex/Dirichlet ``_sfr``."""
    upper = np.asarray(s.bin_edges[1:], dtype=np.float64)
    rates = np.asarray(s.sfrs, dtype=np.float64)
    n = rates.size

    def f(age):
        idx = np.searchsorted(upper, age)
        out = rates[np.minimum(idx, n - 1)]
        bad = (idx >= n) | (age < s.min_age) | (age > s.max_age)
        return np.where(bad, 0.0, out)

    return f


def _h_continuity(s, span):
    nodes = np.concatenate([[s.min_age, s.max_age], np.asarray(s.bin_edges, dtype=np.float64)])
    return nodes, _step_eval_lower(s), 1, None


def _h_continuity_psb(s, span):
    edges = np.asarray(s.bin_edges, dtype=np.float64)
    rates = np.asarray(s.sfrs, dtype=np.float64)
    n = rates.size

    def f(age):
        idx = np.searchsorted(edges, age, side="right") - 1
        out = rates[np.clip(idx, 0, n - 1)]
        bad = (idx < 0) | (idx >= n) | (age < s.min_age) | (age >= s.max_age)
        return np.where(bad, 0.0, out)

    return np.concatenate([[s.min_age, s.max_age], edges]), f, 1, None


def _h_interp(s, span):
    # DenseBasis / Stochastic: np.interp on (finegrid, intsfh) -> piecewise
    # linear, integrated exactly by the midpoint rule with knots as nodes.
    return np.asarray(s.finegrid, dtype=np.float64), s._sfrs, 1, None


def _h_combined(s, span):
    parts = [_handler_for(m)(m, span) for m in s.sfhs]
    if any(p[3] is not None for p in parts):
        raise _Unsupported("CombinedSFH with singular component")
    weights = np.asarray(s.weights, dtype=np.float64)
    funcs = [p[1] for p in parts]

    def f(age):
        out = np.zeros_like(age, dtype=np.float64)
        for w, fn in zip(weights, funcs):
            out = out + fn(age) * w
        return out

    nodes = np.concatenate([p[0] for p in parts])
    return nodes, f, max(p[2] for p in parts), None


_HANDLERS = {
    SFH.Constant: _h_constant,
    SFH.Gaussian: _h_gaussian,
    SFH.Exponential: _h_exponential,
    SFH.TruncatedExponential: _h_exponential,
    SFH.DecliningExponential: _h_exponential,
    SFH.DelayedExponential: _h_delayed,
    SFH.LogNormal: _h_lognormal,
    SFH.DoublePowerLaw: _h_doublepowerlaw,
    SFH.DenseBasis: _h_interp,
}
for _name, _h in (
    ("Continuity", _h_continuity),
    ("ContinuityFlex", _h_continuity),
    ("Dirichlet", _h_continuity),
    ("ContinuityPSB", _h_continuity_psb),
    ("CombinedSFH", _h_combined),
    ("Stochastic", _h_interp),
):
    if hasattr(SFH, _name):
        _HANDLERS[getattr(SFH, _name)] = _h


def _handler_for(sfh) -> Callable:
    # Exact type match only: subclasses may override _sfr.
    h = _HANDLERS.get(type(sfh))
    if h is None:
        raise _Unsupported(f"unsupported SFH type {type(sfh).__name__}")
    return h


def is_supported_sfh(sfh) -> bool:
    """Return True if ``sfh`` has a vectorised fast path."""
    combined = getattr(SFH, "CombinedSFH", None)
    if combined is not None and type(sfh) is combined:
        return all(is_supported_sfh(m) for m in sfh.sfhs)
    return type(sfh) in _HANDLERS


def is_supported_zdist(metal_dist) -> bool:
    """Return True if ``metal_dist`` has an exact/vectorised fast path."""
    return isinstance(metal_dist, (ZDist.DeltaConstant, ZDist.Normal, float, np.floating)) or (
        isinstance(metal_dist, unyt_quantity) and metal_dist.size == 1
    )


# --------------------------------------------------------------------------
# Age axis
# --------------------------------------------------------------------------
def _quad_bin_masses(sfh, edges, offset):
    """Synthesizer's own per-bin quad loop (reference / fallback)."""
    out = np.zeros(edges.size - 1)
    for i in range(out.size):
        out[i] = integrate.quad(sfh.get_sfr, edges[i] + offset, edges[i + 1] + offset)[0]
    return out


def _fast_bin_masses(sfh, log10ages, offset) -> np.ndarray:
    edges, base = _base_nodes(log10ages)
    span = edges[-1] - edges[0]
    extra, func, order, head = _handler_for(sfh)(sfh, span)

    if offset != 0.0:
        edges = edges + offset
        base = base + offset
    lo, hi = edges[0], edges[-1]
    extra = np.asarray(extra, dtype=np.float64)
    extra = extra[np.isfinite(extra) & (extra > lo) & (extra < hi)]
    nodes = np.concatenate([base, extra])
    nodes.sort()
    keep = np.empty(nodes.size, dtype=bool)
    keep[0] = True
    np.greater(nodes[1:], nodes[:-1], out=keep[1:])
    nodes = nodes[keep]

    a = nodes[:-1]
    h = 0.5 * (nodes[1:] - a)
    m = a + h
    if order == 1:
        seg = 2.0 * h * func(m)
    else:
        x, w = _gl(order)
        pts = (m[:, None] + h[:, None] * x[None, :]).ravel()
        seg = h * (func(pts).reshape(-1, order) @ w)

    if head is not None and lo == 0.0:
        # Replace the first (singular) interval by its analytic integral
        seg[0] = head(nodes[1])

    idx = np.searchsorted(nodes, edges)
    return np.add.reduceat(seg, idx[:-1])


def sfh_bin_masses(sfh, log10ages, age_offset=0.0, fallback: bool = True) -> np.ndarray:
    """Mass formed per Synthesizer age bin for an SFH function object.

    Matches ``Stars._get_sfzh``: bin ``i`` spans
    ``[edges[i] + age_offset, edges[i + 1] + age_offset]`` with
    ``edges = [0, midpoints of 10**log10ages, 10**log10ages[-1]]`` (yr), and
    the result is the integral of ``sfh.get_sfr`` over each bin (un-normalised).

    Args:
        sfh (SFH.Common): The SFH function object.
        log10ages (np.ndarray): log10(age / yr) of the age grid.
        age_offset (float/unyt_quantity): Offset (yr if float) added to edges.
        fallback (bool): If True, unsupported SFHs (or non-finite fast
            results) are integrated with Synthesizer's quad loop instead of
            raising.

    Returns:
        np.ndarray: Mass per age bin (same units as ``sfh.get_sfr`` * yr).

    Raises:
        TypeError: If ``fallback=False`` and the SFH is unsupported.
    """
    offset = _to_yr(age_offset)
    try:
        with np.errstate(all="ignore"):
            out = _fast_bin_masses(sfh, log10ages, offset)
        if not np.all(np.isfinite(out)):
            raise _Unsupported("non-finite result")
        return out
    except _Unsupported as e:
        if not fallback:
            raise TypeError(str(e)) from None
    return _quad_bin_masses(sfh, age_bin_edges(log10ages), offset)


# --------------------------------------------------------------------------
# Metallicity axis
# --------------------------------------------------------------------------
_UNIT_MASS = None


def _unit_mass() -> float:
    """Internal value of the 1 Msun dummy particle mass Synthesizer uses.

    Synthesizer converts ``1.0 * Msun`` to its internal (galactic) unit
    system, giving ``1.0000000000000002``; reproducing it keeps the
    metallicity weights bit-for-bit identical.
    """
    global _UNIT_MASS
    if _UNIT_MASS is None:
        from synthesizer.particle import Stars as ParticleStars
        from unyt import Msun, yr

        p = ParticleStars(
            initial_masses=np.array([1.0]) * Msun,
            ages=np.array([0.0]) * yr,
            metallicities=np.array([0.01]),
        )
        _UNIT_MASS = float(np.asarray(p._initial_masses, dtype=np.float64)[0])
    return _UNIT_MASS


def _cic_weights(z: float, metallicities: np.ndarray) -> np.ndarray:
    """Replicates the C cloud-in-cell (ParticleStars.get_metal_dist, mass 1)."""
    grid = metallicities
    n = grid.size
    m = _unit_mass()
    w = np.zeros(n)
    if n == 1:
        w[0] = m
        return w
    if z <= grid[0]:
        w[0] = 1.0 * m
    elif z >= grid[-1]:
        w[-1] = 1.0 * m
    else:
        up = int(np.searchsorted(grid, z, side="right"))
        lo = up - 1
        frac = (z - grid[lo]) / (grid[up] - grid[lo])
        w[lo] = (1.0 - frac) * m
        w[up] = frac * m
    return w


def _particle_metal_weights(z: float, metallicities) -> np.ndarray:
    """Synthesizer's own dummy particle path (used for non-float64 grids)."""
    from synthesizer.particle import Stars as ParticleStars
    from unyt import Msun, yr

    p = ParticleStars(
        initial_masses=np.array([1.0]) * Msun,
        ages=np.array([0.0]) * yr,
        metallicities=np.array([z]),
    )
    return p.get_metal_dist(metallicities)


def _metal_edges(metallicities: np.ndarray) -> np.ndarray:
    z = metallicities
    edges = np.empty(z.size + 1)
    edges[0] = 0.0
    edges[1:-1] = 0.5 * (z[1:] + z[:-1])
    edges[-1] = z[-1]
    return edges


def zdist_weights(metal_dist, metallicities) -> np.ndarray:
    """Weights per metallicity bin, matching ``Stars._get_sfzh``.

    * ``ZDist.DeltaConstant`` / float metallicity: cloud-in-cell assignment in
      linear Z, bit-for-bit identical to Synthesizer (clamped to grid ends).
    * ``ZDist.Normal``: analytic integral of the Gaussian over Synthesizer's
      metallicity bins ``[0, midpoints, Z_max]`` (matches quad to ~1e-10).
    * Any other ``ZDist`` object: Synthesizer's quad loop.

    Args:
        metal_dist: ZDist object or a float metallicity.
        metallicities (np.ndarray): grid metallicities (linear).

    Returns:
        np.ndarray: weight per metallicity bin (un-normalised).
    """
    metallicities = np.asarray(metallicities)
    if isinstance(metal_dist, ZDist.DeltaConstant) or isinstance(
        metal_dist, (float, np.floating, unyt_quantity)
    ):
        if isinstance(metal_dist, ZDist.DeltaConstant):
            z = metal_dist.get_metallicity()
        else:
            z = metal_dist
        z = float(np.asarray(z))
        if metallicities.dtype != np.float64 or metallicities.ndim != 1:
            return _particle_metal_weights(z, metallicities)
        return _cic_weights(z, metallicities)

    if type(metal_dist) is ZDist.Normal:
        mu, sig = float(metal_dist.mean), float(metal_dist.sigma)
        if sig > 0 and np.isfinite(mu) and np.isfinite(sig):
            e = _metal_edges(metallicities.astype(np.float64))
            c = 0.5 * erf((e - mu) / (sig * np.sqrt(2.0)))
            out = np.diff(c)
            # erf differences lose precision deep in the same tail; fall
            # back to quad only in that (rare) case.
            if np.all(np.isfinite(out)):
                return out

    if hasattr(metal_dist, "get_dist_weight"):
        e = _metal_edges(metallicities.astype(np.float64))
        return np.array(
            [
                integrate.quad(metal_dist.get_dist_weight, e[i], e[i + 1])[0]
                for i in range(e.size - 1)
            ]
        )
    raise TypeError(f"Unsupported metallicity distribution {type(metal_dist)}")


# --------------------------------------------------------------------------
# SFZH and Stars
# --------------------------------------------------------------------------
def fast_sfzh(sfh, metal_dist, log10ages, metallicities, age_offset=0.0) -> Optional[np.ndarray]:
    """Un-normalised SFZH ``sfh_bins[:, None] * z_weights[None, :]``.

    Returns None if either input has no fast path (callers should then use
    Synthesizer's own machinery).
    """
    if not is_supported_sfh(sfh):
        return None
    if not (is_supported_zdist(metal_dist) or isinstance(metal_dist, ZDist.Common)):
        return None
    try:
        with np.errstate(all="ignore"):
            sf = _fast_bin_masses(sfh, log10ages, _to_yr(age_offset))
    except _Unsupported:
        return None
    if not np.all(np.isfinite(sf)):
        return None
    zw = zdist_weights(metal_dist, metallicities)
    return sf[:, np.newaxis] * zw


def make_parametric_stars(
    sfh,
    metal_dist,
    log10ages,
    metallicities,
    initial_mass=None,
    subclass: bool = False,
    **kwargs,
) -> ParametricStars:
    """Drop-in fast replacement for ``Stars(sf_hist=sfh, metal_dist=...)``.

    The returned object is constructed through Synthesizer's normal
    ``Stars.__init__`` with ``sf_hist``/``metal_dist`` passed as usual (so
    ``sf_hist_func``, ``metal_dist_func``, ``_instant_sf`` and
    ``_instant_metallicity`` are set exactly as before) plus a precomputed
    ``sfzh=``, which Synthesizer uses instead of its quad integration.
    Normalisation (``initial_mass`` or ``surviving_mass`` + ``grid``),
    ``sfzh_normalisation``, ``sf_hist``/``metal_dist`` projections and all
    other attributes are therefore produced by Synthesizer itself.

    Falls back to the original quad path transparently whenever the SFH or
    ZDist is not supported.

    Args:
        sfh: SFH function object (or anything Stars accepts as ``sf_hist``).
        metal_dist: ZDist object or float (or anything Stars accepts).
        log10ages (np.ndarray): Grid log10 ages.
        metallicities (np.ndarray): Grid metallicities.
        initial_mass (unyt_quantity): Total initial stellar mass.
        subclass (bool): If True return a :class:`FastParametricStars`, whose
            ``_get_sfzh`` (used by ``get_at_earlier_time`` and
            ``calculate_*_mass_at_age``) also uses the fast path. Default
            False returns a plain ``synthesizer.parametric.Stars``.
        **kwargs: Passed to ``Stars`` (e.g. ``surviving_mass``, ``grid``,
            ``fesc``, emitter parameters).

    Returns:
        synthesizer.parametric.Stars
    """
    cls = FastParametricStars if subclass else ParametricStars
    if subclass:
        # The subclass computes its SFZH lazily in _get_sfzh.
        return cls(
            log10ages=log10ages,
            metallicities=metallicities,
            sf_hist=sfh,
            metal_dist=metal_dist,
            initial_mass=initial_mass,
            **kwargs,
        )

    sfzh = kwargs.pop("sfzh", None)
    if sfzh is None:
        sfzh = fast_sfzh(sfh, metal_dist, log10ages, metallicities)

    stars = ParametricStars(
        log10ages=log10ages,
        metallicities=metallicities,
        sf_hist=sfh,
        metal_dist=metal_dist,
        initial_mass=initial_mass,
        sfzh=sfzh,
        **kwargs,
    )
    # Synthesizer's _get_sfzh sets this side effect for DeltaConstant
    if sfzh is not None and isinstance(metal_dist, ZDist.Common):
        if metal_dist.name == "DeltaConstant":
            stars._instant_metallicity = metal_dist.get_metallicity()
    return stars


class FastParametricStars(ParametricStars):
    """``synthesizer.parametric.Stars`` with a vectorised ``_get_sfzh``.

    Identical to the parent except that SFZH integration (at construction and
    in ``get_at_earlier_time``/``calculate_*_mass_at_age``) uses
    :func:`fast_sfzh` when possible, falling back to the parent otherwise.
    """

    def _get_sfzh(self, age_offset=None):
        """Compute the SFZH, using the fast path when supported."""
        sfh = self.sf_hist_func
        zd = self.metal_dist_func
        if sfh is not None and self._instant_sf is None:
            if zd is not None:
                metal = zd
            elif self._instant_metallicity is not None:
                metal = self._instant_metallicity
            else:
                metal = None
            if metal is not None:
                sfzh = fast_sfzh(
                    sfh,
                    metal,
                    self.log10ages,
                    self.metallicities,
                    age_offset=0.0 if age_offset is None else age_offset,
                )
                if sfzh is not None:
                    if zd is not None and zd.name == "DeltaConstant":
                        self._instant_metallicity = zd.get_metallicity()
                    return sfzh
        return super()._get_sfzh(age_offset=age_offset)
