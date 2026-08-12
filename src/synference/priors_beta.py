# -*- coding: utf-8 -*-
"""Prospector-Beta-style conditional priors for library generation.

This module ports the physically-motivated, parameter-dependent priors of
Prospector-Beta (Wang, Leja, et al. 2023, ApJL 944, L58) from
``prospect.models.priors_beta`` into plain numpy/scipy so they can be used to
generate ``synference`` training libraries. Four pieces are implemented:

1.  **Stellar mass function (SMF)** mass prior, ``p(log_mass | zred)``:
    Leja et al. (2020) continuity-model double-Schechter fit, clamped to its
    [0.2, 3.0] validity range (matches Prospector-Beta's default
    ``TemplateLibrary["beta"]`` configuration; no high-z blend is applied).
2.  **Mass-metallicity relation (MZR)**, ``p(log_zmet | log_mass)``: a
    truncated Gaussian whose mean/sigma are interpolated from the tabulated
    Gallazzi et al. (2005) relation.
3.  **Dynamic redshift prior**, ``p(zred) ~ N(zred) * dV_c/dz``, where
    ``N(z)`` is the SMF integrated above a mass floor.
4.  **SFH prior**, ``p(logsfr_ratios | zred, log_mass)``: a Student-t(df=2)
    prior on adjacent Continuity-SFH bin ratios, centered on the expected
    ratio implied by the Behroozi et al. (2019) cosmic SFRD, time-shifted by
    a mass-dependent "downsizing" offset.

Unlike upstream Prospector (which hardcodes WMAP9), every age/lookback-time
calculation here takes an explicit ``cosmo`` (default ``Planck18``), so
libraries generated with this module stay internally consistent with the
rest of the ``synference`` pipeline.

``sample_prospector_beta_prior`` is the main entry point: it draws a joint
Latin Hypercube over the unit cube and pushes each row through the sequential
conditional quantile transform above (redshift -> mass|z -> met|mass ->
SFH|mass,z), mirroring ``BetaPrior.unit_transform`` in ``prospect``.
"""

import inspect

import astropy.units as u
import numpy as np
from astropy.cosmology import Planck18
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from scipy.stats import qmc, t, truncnorm
from tqdm import tqdm
from unyt import Myr, unyt_array

from .prior_data import MASSMET, SPL_TL_SFRD

__all__ = [
    "loc_massmet",
    "scale_massmet",
    "mass_func_at_z",
    "cdf_mass_func_at_z",
    "delta_t_dex",
    "expected_logsfr_ratios",
    "beta_agebins",
    "setup_mass_normalization",
    "setup_dynamic_z_prior",
    "sample_prospector_beta_prior",
]

# Leja et al. (2020) continuity-model median double-Schechter parameters,
# tabulated at three anchor redshifts (z=0.2, 1.6, 3.0).
_LEJA_20_PARS = {
    "logphi1": (-2.44, -3.08, -4.14),
    "logphi2": (-2.89, -3.29, -3.51),
    "logmstar": (10.79, 10.88, 10.84),
    "alpha1": -0.28,
    "alpha2": -1.48,
}


# ---------------------------------------------------------------------------
# Mass-metallicity relation (Gallazzi et al. 2005)
# ---------------------------------------------------------------------------


def loc_massmet(log_mass):
    """Mean log(Z/Zsun) of the Gallazzi et al. (2005) mass-metallicity relation."""
    return np.interp(log_mass, MASSMET[:, 0], MASSMET[:, 1])


def scale_massmet(log_mass):
    """Width (P84 - P16) of the Gallazzi et al. (2005) mass-metallicity relation."""
    p84 = np.interp(log_mass, MASSMET[:, 0], MASSMET[:, 3])
    p16 = np.interp(log_mass, MASSMET[:, 0], MASSMET[:, 2])
    return p84 - p16


# ---------------------------------------------------------------------------
# Stellar mass function: Leja et al. (2020)
# ---------------------------------------------------------------------------


def _schechter(logm, logphi, logmstar, alpha):
    """Schechter (1976) function in dlogm."""
    return (
        (10**logphi)
        * np.log(10)
        * 10 ** ((logm - logmstar) * (alpha + 1))
        * np.exp(-(10 ** (logm - logmstar)))
    )


def _parameter_at_z0(y, z0, z1=0.2, z2=1.6, z3=3.0):
    """Quadratic interpolation of a Leja+20 parameter between 3 anchor redshifts."""
    y1, y2, y3 = y
    a = ((y3 - y1) + (y2 - y1) / (z2 - z1) * (z1 - z3)) / (
        z3**2 - z1**2 + (z2**2 - z1**2) / (z2 - z1) * (z1 - z3)
    )
    b = ((y2 - y1) - a * (z2**2 - z1**2)) / (z2 - z1)
    c = y1 - a * z1**2 - b * z1
    return a * z0**2 + b * z0 + c


def _low_z_mass_func(z0, logm):
    """Leja et al. (2020) double-Schechter mass function, valid for z <~ 3."""
    z0_clamped = np.clip(z0, 0.2, 3.0)
    logphi1 = _parameter_at_z0(_LEJA_20_PARS["logphi1"], z0_clamped)
    logphi2 = _parameter_at_z0(_LEJA_20_PARS["logphi2"], z0_clamped)
    logmstar = _parameter_at_z0(_LEJA_20_PARS["logmstar"], z0_clamped)
    phi1 = _schechter(logm, logphi1, logmstar, _LEJA_20_PARS["alpha1"])
    phi2 = _schechter(logm, logphi2, logmstar, _LEJA_20_PARS["alpha2"])
    return phi1 + phi2


def mass_func_at_z(z, log_mass, bounds=(6.0, 12.5)):
    """GSMF at redshift z, evaluated on a sorted mass grid.

    Composite stellar mass function Phi(log_mass, z) (Leja+20, clamped to its
    [0.2, 3.0] validity range rather than extrapolated, matching
    Prospector-Beta's default ``TemplateLibrary["beta"]`` configuration).
    """
    log_mass = np.asarray(log_mass, dtype=float)
    phi = _low_z_mass_func(z, log_mass)
    phi = np.where((log_mass < bounds[0]) | (log_mass > bounds[1]), 0.0, phi)
    return phi


def _pdf_mass_func_at_z(z, log_mass_grid, bounds):
    phi = mass_func_at_z(z, log_mass_grid, bounds=bounds)
    norm = simpson(y=phi, x=log_mass_grid)
    return phi / norm


def cdf_mass_func_at_z(z, log_mass_grid, bounds=(6.0, 12.5)):
    """CDF of the stellar mass function at redshift z, evaluated on a sorted mass grid."""
    pdf = _pdf_mass_func_at_z(z, log_mass_grid, bounds)
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    cdf[cdf < 0] = 0.0
    cdf[0] = 0.0
    cdf[-1] = 1.0
    return cdf


def _ppf_from_grid(u_val, xs, cdf):
    """Invert a tabulated CDF (percent-point function)."""
    return interp1d(cdf, xs, bounds_error=False, fill_value=(xs[0], xs[-1]))(u_val)


# ---------------------------------------------------------------------------
# Mass-function normalization N(z) and dynamic redshift prior p(z) ~ N(z) dV/dz
# ---------------------------------------------------------------------------


def setup_mass_normalization(
    cosmo,
    zred_min,
    zred_max,
    mass_max,
    mass_min=7.0,
    zred_grid_len=4000,
    mass_integ_grid=100,
):
    """Precompute N(z) = integral of Phi(log_mass, z) over [mass_min, mass_max].

    Returns (z_grid, n_gal_z, finterp_mass_norm).
    """
    z_grid = np.linspace(zred_min, zred_max, zred_grid_len)
    n_gal_z = np.zeros_like(z_grid)
    m_grid = np.linspace(mass_min, mass_max, mass_integ_grid)
    for i, z in enumerate(z_grid):
        phi = mass_func_at_z(z, m_grid, bounds=(mass_min, mass_max))
        n_gal_z[i] = simpson(y=phi, x=m_grid)
    finterp_mass_norm = interp1d(z_grid, n_gal_z, bounds_error=False, fill_value=0.0)
    return z_grid, n_gal_z, finterp_mass_norm


def setup_dynamic_z_prior(cosmo, z_grid, n_gal_z, zred_min, zred_max):
    """Build p(z) ~ N(z) * dV_c/dz and its inverse-CDF interpolator.

    Returns finterp_cdf_z: callable mapping a unit-interval value to zred.
    """
    dvol = cosmo.differential_comoving_volume(z_grid).value
    pdf_unnorm = n_gal_z * dvol

    in_range = (z_grid >= zred_min) & (z_grid <= zred_max)
    z_in = z_grid[in_range]
    pdf_in = pdf_unnorm[in_range]
    pdf_in = pdf_in / simpson(y=pdf_in, x=z_in)
    pdf_in[pdf_in < 0] = 0.0

    cdf = np.cumsum(pdf_in)
    cdf /= cdf[-1]
    cdf = np.concatenate(([0.0], cdf))
    z_cdf = np.concatenate(([z_in[0]], z_in))
    cdf[-1] = 1.0

    # scipy interp1d needs a strictly increasing x-array; de-duplicate flat spots.
    cdf_unique, idx_unique = np.unique(cdf, return_index=True)
    z_unique = z_cdf[idx_unique]

    finterp_cdf_z = interp1d(
        cdf_unique, z_unique, bounds_error=False, fill_value=(z_unique[0], z_unique[-1])
    )
    return finterp_cdf_z


# ---------------------------------------------------------------------------
# SFH prior: mass/redshift-conditioned expectation from the Behroozi+19 SFRD
# ---------------------------------------------------------------------------


def delta_t_dex(log_mass, mlims=(9.0, 12.0), dlims=(-0.2, 0.8)):
    """Mass-dependent "downsizing" shift (dex) applied to the age of the universe."""
    a = (dlims[1] - dlims[0]) / (mlims[1] - mlims[0])
    b = dlims[0] - a * mlims[0]
    return np.clip(a * log_mass + b, dlims[0], dlims[1])


def _make_age_to_redshift(cosmo, z_max=30.0, n_grid=2000):
    """Build an interpolator from age of the universe [yr] back to redshift."""
    z_grid = np.concatenate(([0.0], np.logspace(-3, np.log10(z_max), n_grid)))
    age_grid_yr = cosmo.age(z_grid).to(u.yr).value
    # age_grid_yr decreases monotonically with z; interp1d needs increasing x.
    return interp1d(
        age_grid_yr[::-1],
        z_grid[::-1],
        bounds_error=False,
        fill_value=(z_grid[-1], z_grid[0]),
    )


def _agebins_for_expectation(zstart, cosmo, nbins_sfh=7, amin=7.1295):
    """Age bins for the expected SFR ratios, given a starting redshift.

    Age bin edges (linear years) used only to compute the *expected* SFR
    per bin, following the same spacing as :func:`beta_agebins` but starting
    from an already-elapsed lookback time (Prospector's ``z_to_agebins_rescale``).
    """
    agelims = np.zeros(nbins_sfh + 1)
    agelims[0] = cosmo.lookback_time(zstart).to(u.yr).value
    tuniv = cosmo.lookback_time(15).to(u.yr).value
    tbinmax = tuniv - (tuniv - agelims[0]) * 0.10
    agelims[-2] = tbinmax
    agelims[-1] = tuniv

    if zstart <= 3.0:
        agelims[1] = agelims[0] + 3e7
        agelims[2] = agelims[1] + 1e8
        i_age = 3
    else:
        agelims[1] = agelims[0] + 10**amin
        i_age = 2
    nbins_mid = len(agelims) - i_age

    with np.errstate(invalid="ignore", divide="ignore"):
        log_edges = (
            np.log10(agelims[:i_age]).tolist()[:-1]
            + np.linspace(np.log10(agelims[i_age - 1]), np.log10(tbinmax), nbins_mid).tolist()
            + [np.log10(tuniv)]
        )
    if agelims[0] == 0:
        log_edges[0] = 0.0

    log_edges = np.array(log_edges)
    edges = np.column_stack([log_edges[:-1], log_edges[1:]])
    return 10.0**edges


def expected_logsfr_ratios(
    zred,
    log_mass,
    nbins_sfh=7,
    logsfr_ratio_mini=-5.0,
    logsfr_ratio_maxi=5.0,
    cosmo=Planck18,
    age_to_redshift=None,
    amin=7.1295,
):
    """Expected log10(SFR_j / SFR_{j+1}) given (zred, log_mass).

    From the Behroozi et al. (2019) cosmic SFRD shifted in time
    by a mass-dependent "downsizing" offset (:func:`delta_t_dex`).
    """
    if age_to_redshift is None:
        age_to_redshift = _make_age_to_redshift(cosmo)

    age_yr = cosmo.age(zred).to(u.yr).value
    age_shifted_yr = 10.0 ** (np.log10(age_yr) + delta_t_dex(log_mass))
    z_shifted = float(np.clip(age_to_redshift(age_shifted_yr), 0.15, 10.0))

    agebins_shifted = _agebins_for_expectation(z_shifted, cosmo, nbins_sfh=nbins_sfh, amin=amin)

    sfr_shifted = np.array(
        [SPL_TL_SFRD.integral(a, b) / (b - a) for a, b in agebins_shifted]
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        logsfr_ratios = np.log10(sfr_shifted[:-1] / sfr_shifted[1:])
    logsfr_ratios = np.clip(logsfr_ratios, logsfr_ratio_mini, logsfr_ratio_maxi)

    if not np.all(np.isfinite(logsfr_ratios)):
        bad = np.where(~np.isfinite(logsfr_ratios))[0]
        first_bad = bad.min()
        if first_bad > 0:
            logsfr_ratios[first_bad:] = logsfr_ratios[first_bad - 1]
        else:
            logsfr_ratios[~np.isfinite(logsfr_ratios)] = 0.0

    return logsfr_ratios


# ---------------------------------------------------------------------------
# Redshift-dependent Continuity-SFH age bins (for the sampled galaxy itself)
# ---------------------------------------------------------------------------


def beta_agebins(zred, nbins_sfh=7, cosmo=Planck18, amin=7.1295):
    """Continuity-SFH age-bin edges at redshift ``zred``.

    Follows the Prospector-Beta scheme (``zred_to_agebins_pbeta``). Returned as a
    ``unyt_array`` of shape ``(nbins_sfh, 2)`` in Myr, directly usable as
    ``synthesizer.parametric.SFH.Continuity(agebins=...)``.
    """
    tuniv_yr = cosmo.age(zred).to(u.yr).value
    tbinmax = tuniv_yr * 0.9

    if zred <= 3.0:
        agelims = (
            [0.0, 7.47712]
            + np.linspace(8.0, np.log10(tbinmax), nbins_sfh - 2).tolist()
            + [np.log10(tuniv_yr)]
        )
    else:
        agelims = np.linspace(amin, np.log10(tbinmax), nbins_sfh).tolist() + [
            np.log10(tuniv_yr)
        ]
        agelims[0] = 0.0

    edges_yr = 10.0 ** np.array(agelims)
    edges_yr[0] = 0.0  # true zero (age = now), matching this repo's continuity_agebins
    agebins = np.column_stack([edges_yr[:-1], edges_yr[1:]])
    return unyt_array(agebins, "yr").to(Myr)


# ---------------------------------------------------------------------------
# Truncated Student-t(df=2) quantile transform for the SFH ratio prior
# ---------------------------------------------------------------------------


def _truncated_t_unit_transform(u_val, loc, scale, half_width, df=2):
    """Inverse-CDF of a Student-t(df) truncated to [-half_width, half_width].

    shifted by ``loc`` and evaluated at unit-cube coordinate(s) ``u_val``.
    Mirrors the rescaled-CDF trick already used by
    ``synthesizer.parametric.SFH.Continuity.init_from_prior``.
    """
    cdf_min = t.cdf(-half_width, df=df, scale=scale)
    cdf_max = t.cdf(half_width, df=df, scale=scale)
    rescaled = cdf_min + (cdf_max - cdf_min) * u_val
    return t.ppf(rescaled, df=df, scale=scale) + loc


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def sample_prospector_beta_prior(
    N,
    nbins_sfh=7,
    zred_range=(1e-3, 15.0),
    mass_range=(7.0, 12.5),
    met_range=(-1.98, 0.19),
    logsfr_ratio_range=(-5.0, 5.0),
    logsfr_ratio_tscale=0.3,
    cosmo=Planck18,
    rng=None,
    zred_grid_len=4000,
    mass_subgrid_len=1000,
    mass_integ_grid=100,
    mask=None,
    verbose=True,
):
    """Draw N galaxies from the full Prospector-Beta prior.

    Draws a single joint Latin Hypercube over the unit cube (dimensionality
    ``3 + (nbins_sfh - 1)``: zred, log_mass, log_zmet, logsfr_ratios) and
    pushes each row through the sequential conditional quantile transform
    zred -> log_mass | zred -> log_zmet | log_mass -> logsfr_ratios | zred,
    log_mass, mirroring ``BetaPrior.unit_transform`` in
    ``prospect.models.priors_beta``.

    ``mask`` : optional boolean array of length N. When given, the (cheap,
    vectorized) redshift draw is still computed for every galaxy, but the
    expensive per-galaxy mass/metallicity/SFH quantile transform is only run
    for indices where ``mask`` is True (the rest are left as NaN). This
    mirrors the multinode pattern in
    ``examples/library_generation/scripts/final_library_generation_multinode.py``,
    where every MPI rank computes the same full-size, seeded hypercube for
    reproducibility but only pays the per-galaxy cost for its own slice.

    Returns:
    -------
    dict with keys "redshift", "log_mass", "log_zmet",
    "logsfr_ratios" (N, nbins_sfh - 1), and "agebins"
    (N, nbins_sfh, 2) unyt Myr.
    """
    zred_min, zred_max = zred_range
    mass_min, mass_max = mass_range

    ndim = 3 + (nbins_sfh - 1)
    sampler_kwargs = {}
    if rng is not None:
        # scipy >= 1.15 renamed LatinHypercube's `seed` kwarg to `rng`.
        rng_key = "rng" if "rng" in inspect.signature(qmc.LatinHypercube).parameters else "seed"
        sampler_kwargs = {rng_key: rng}
    sampler = qmc.LatinHypercube(d=ndim, **sampler_kwargs)
    u = sampler.random(int(N))

    # --- Precompute the dynamic p(z) and mass-function normalization ---
    z_grid, n_gal_z, _ = setup_mass_normalization(
        cosmo,
        zred_min,
        zred_max,
        mass_max,
        mass_min=mass_min,
        zred_grid_len=zred_grid_len,
        mass_integ_grid=mass_integ_grid,
    )
    finterp_cdf_z = setup_dynamic_z_prior(cosmo, z_grid, n_gal_z, zred_min, zred_max)
    age_to_redshift = _make_age_to_redshift(cosmo)

    # --- 1. Redshift: vectorized, independent of everything else drawn ---
    redshift = finterp_cdf_z(u[:, 0])

    # --- 2, 3, 4: sequential per-galaxy, since each depends on prior draws ---
    log_mass = np.full(N, np.nan)
    log_zmet = np.full(N, np.nan)
    logsfr_ratios = np.full((N, nbins_sfh - 1), np.nan)
    agebins = np.full((N, nbins_sfh, 2), np.nan)

    m_subgrid = np.linspace(mass_min, mass_max, mass_subgrid_len)

    indices = np.arange(N) if mask is None else np.nonzero(mask)[0]
    iterator = tqdm(indices, desc="Sampling Prospector-Beta prior", disable=not verbose)
    for i in iterator:
        z_i = redshift[i]

        # Mass | redshift, via the SMF CDF.
        cdf_mass = cdf_mass_func_at_z(z_i, m_subgrid, bounds=(mass_min, mass_max))
        m_i = float(_ppf_from_grid(u[i, 1], m_subgrid, cdf_mass))
        log_mass[i] = m_i

        # Metallicity | mass, via a truncated normal.
        a = (met_range[0] - loc_massmet(m_i)) / scale_massmet(m_i)
        b = (met_range[1] - loc_massmet(m_i)) / scale_massmet(m_i)
        log_zmet[i] = truncnorm.ppf(
            u[i, 2], a, b, loc=loc_massmet(m_i), scale=scale_massmet(m_i)
        )

        # SFH ratios | mass, redshift, via a shifted truncated Student-t.
        loc = expected_logsfr_ratios(
            z_i,
            m_i,
            nbins_sfh=nbins_sfh,
            logsfr_ratio_mini=logsfr_ratio_range[0],
            logsfr_ratio_maxi=logsfr_ratio_range[1],
            cosmo=cosmo,
            age_to_redshift=age_to_redshift,
        )
        ratios = _truncated_t_unit_transform(
            u[i, 3:],
            loc=loc,
            scale=logsfr_ratio_tscale,
            half_width=logsfr_ratio_range[1],
        )
        logsfr_ratios[i] = np.clip(ratios, logsfr_ratio_range[0], logsfr_ratio_range[1])

        # Age bins for this galaxy's redshift.
        agebins[i] = beta_agebins(z_i, nbins_sfh=nbins_sfh, cosmo=cosmo).to_value(Myr)

    return {
        "redshift": redshift,
        "log_mass": log_mass,
        "log_zmet": log_zmet,
        "logsfr_ratios": logsfr_ratios,
        "agebins": unyt_array(agebins, Myr),
    }
