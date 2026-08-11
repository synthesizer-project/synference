# -*- coding: utf-8 -*-
"""Tabulated data backing the Prospector-Beta-style conditional priors.

Data sources
------------
1. Mass-metallicity relation: Gallazzi et al. (2005), MNRAS 362, 41.
2. Cosmic star formation rate density: Behroozi et al. (2019), MNRAS 488, 3143.

Both tables are copied verbatim from the ``prospector`` package
(``prospect/models/prior_data/``), which implements the same priors for the
Prospector-Beta model (Wang et al. 2023, ApJL 944, L58).
"""

from importlib.resources import files

import numpy as np
from scipy.interpolate import UnivariateSpline

__all__ = ["MASSMET", "SPL_TL_SFRD"]

_base_path = files(__name__)


def _load_txt(filename, **kwargs):
    with (_base_path / filename).open("rb") as f:
        return np.loadtxt(f, **kwargs)


# Mass-metallicity relation (Gallazzi et al. 2005), Chabrier IMF.
# Columns: [logMass, P50_logZ, P16_logZ, P84_logZ]
MASSMET = _load_txt("gallazzi_05_massmet.txt")

# Cosmic SFR density (Behroozi et al. 2019).
# Columns: [redshift, lookback_time, sfrd]. The file's header claims the
# lookback-time column is in Gyr, but the raw values are actually in
# *years* (e.g. at z=1.02 the tabulated lookback time is 7.93e9, i.e.
# 7.93 Gyr only if read as years; interpreted literally as Gyr it would be
# ~7.9 billion Gyr, far exceeding the age of the universe). Verified against
# several more (z, lookback) pairs spanning z=1e-8 to z=30 before trusting
# this; downstream code must pass lookback times in years to match.
_z_b19, _tl_b19_yr, _sfrd_b19 = _load_txt("behroozi_19_sfrd.txt", unpack=True)

# Spline of SFRD vs lookback time [yr], used to compute expected SFRs in
# fixed lookback-time bins (see priors_beta.expected_logsfr_ratios).
SPL_TL_SFRD = UnivariateSpline(_tl_b19_yr, _sfrd_b19, s=0, ext=3)
