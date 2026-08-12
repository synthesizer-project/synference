# -*- coding: utf-8 -*-
"""Prospector-alpha-style priors for library generation.

Prospector-alpha (Leja et al. 2017) predates Prospector-Beta
(:mod:`synference.priors_beta`) and, unlike it, has no hierarchical/
parameter-dependent priors: total stellar mass and stellar metallicity are
drawn independently (no mass-metallicity relation), and the non-parametric
Dirichlet-SFH uses a *fixed* set of age bins (not conditioned on redshift).
The one thing worth a dedicated utility here is the Dirichlet-SFH prior
itself: it is not a plain independent-per-parameter draw, since the
``ncomp - 1`` latent ``z_fraction`` variables must each be drawn from a
*different*, position-dependent Beta distribution for the Betancourt (2010)/
Leja et al. (2017) stick-breaking construction to yield a symmetric
Dirichlet(1, ..., 1) prior over the ``ncomp`` SFR fractions
(``prospect.models.templates.adjust_dirichlet_agebins``); a plain
``draw_from_hypercube`` call cannot express that.

``synthesizer.parametric.SFH.Dirichlet`` already implements the matching
stick-breaking transform from ``z_fraction`` to per-bin mass fractions, so
this module only needs to supply correctly-distributed ``z_fraction`` draws.

The other non-``draw_from_hypercube``-able piece is ``dust_ratio``
(``prospect.models.templates._alpha_["dust_ratio"]``, inherited unchanged by
Prospector-Beta): the birth-cloud-to-diffuse dust ratio used to set
``dust1 = dust2 * dust_ratio`` (``transforms.dustratio_to_dust1``), drawn from
a truncated normal (``priors.ClippedNormal`` is ``scipy.stats.truncnorm``
under the hood, not a simple clip-after-draw).
"""

import inspect

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import qmc, truncnorm

__all__ = [
    "dirichlet_zfraction_beta_params",
    "sample_dirichlet_sfr_fractions",
    "sample_dust_ratio",
]


def dirichlet_zfraction_beta_params(ncomp):
    """Draws Dirichlet SFR fractions.

    Per-latent-variable (alpha, beta) shape parameters for the stick-breaking
    construction of a symmetric Dirichlet(1, ..., 1) prior over ``ncomp`` SFR
    fractions via ``ncomp - 1`` independent Beta-distributed ``z_fraction``
    variables (Betancourt 2010; Leja et al. 2017). Matches
    ``prospect.models.templates.adjust_dirichlet_agebins``.
    """
    alpha = np.arange(ncomp - 1, 0, -1, dtype=float)
    beta_shape = np.ones_like(alpha)
    return alpha, beta_shape


def sample_dirichlet_sfr_fractions(N, ncomp, rng=None):
    """Sample Dirichlet SFR fractions for Prospector-alpha/-beta.

    Draw N sets of Dirichlet(1, ..., 1)-distributed SFR fractions over
    ``ncomp`` bins, via the stick-breaking ``z_fraction`` latent variables
    (see :func:`dirichlet_zfraction_beta_params`).

    Each of the ``ncomp - 1`` components is drawn from its own Beta marginal
    via a joint Latin Hypercube over the unit cube pushed through the
    per-component Beta quantile function, preserving space-filling.

    Returns:
    -------
    z_fraction : ndarray of shape (N, ncomp - 1)
        Pass directly as ``synthesizer.parametric.SFH.Dirichlet(z_fraction=...)``.
    """
    alpha, beta_shape = dirichlet_zfraction_beta_params(ncomp)
    ndim = ncomp - 1

    sampler_kwargs = {}
    if rng is not None:
        rng_key = "rng" if "rng" in inspect.signature(qmc.LatinHypercube).parameters else "seed"
        sampler_kwargs = {rng_key: rng}
    sampler = qmc.LatinHypercube(d=ndim, **sampler_kwargs)
    u = sampler.random(int(N))

    z_fraction = beta_dist.ppf(u, alpha, beta_shape)
    return z_fraction


def sample_dust_ratio(N, mean=1.0, sigma=0.3, mini=0.0, maxi=2.0, rng=None):
    """Sample dust ratio prior draws for Prospector-alpha/-beta.

    Draw N samples of Prospector-alpha/-beta's ``dust_ratio`` prior:
    ``priors.ClippedNormal(mean=1.0, sigma=0.3, mini=0.0, maxi=2.0)``
    (``prospect.models.templates._alpha_["dust_ratio"]``), i.e. a normal
    distribution truncated (not clipped) to ``[mini, maxi]``.

    ``dust_ratio`` sets the birth-cloud optical depth relative to the diffuse
    one, ``dust1 = dust2 * dust_ratio``, for stars younger than
    ``dust_tesc`` (10 Myr) in the Charlot & Fall (2000) two-component dust
    model.

    Drawn via inverse-CDF through a 1-D Latin Hypercube (matching the
    space-filling idiom used elsewhere in this module and in
    ``draw_from_hypercube``) rather than ``scipy.stats.truncnorm.rvs``.

    Returns:
    -------
    dust_ratio : ndarray of shape (N,)
    """
    a = (mini - mean) / sigma
    b = (maxi - mean) / sigma

    sampler_kwargs = {}
    if rng is not None:
        rng_key = "rng" if "rng" in inspect.signature(qmc.LatinHypercube).parameters else "seed"
        sampler_kwargs = {rng_key: rng}
    sampler = qmc.LatinHypercube(d=1, **sampler_kwargs)
    u = sampler.random(int(N))[:, 0]

    return truncnorm.ppf(u, a, b, loc=mean, scale=sigma)
