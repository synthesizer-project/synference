"""Tests for synference.fast_sfzh (vectorised parametric SFZH construction).

The fast path is compared against Synthesizer's own ``Stars(sf_hist=...,
metal_dist=...)`` quad integration. Synthesizer's ``quad`` is itself
inaccurate for sharply peaked SFHs / steps inside wide age bins (we have seen
errors up to ~1e-2), so each case is additionally compared against a refined
reference (quad split at the SFH's natural breakpoints, tight tolerances): the
fast result must match Synthesizer to < 1e-3 OR, where Synthesizer's own error
vs the refined reference exceeds 5e-4, match the refined reference to < 1e-5.

Run with ``pytest -s tests/test_fast_sfzh.py`` to see the error distributions.
"""

import os
import warnings

import numpy as np
import pytest
from scipy import integrate
from synthesizer.grid import Grid
from synthesizer.parametric import SFH, Galaxy, Stars, ZDist
from unyt import Msun, Myr, angstrom, unyt_array, yr

from synference import fast_sfzh as F

GRID_NAME = "bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps.hdf5"
AGE_MAX = 13.5e3  # Myr
TOL_SYNTH = 1e-3
TOL_REF = 1e-5
SUMMARY = {}


def _grid_dir():
    d = os.environ.get("SYNTHESIZER_GRID_DIR")
    if d is None:
        try:
            from synthesizer import get_grids_dir

            d = str(get_grids_dir())
        except Exception:  # pragma: no cover
            return None
    return d


@pytest.fixture(scope="module")
def grid():
    """BPASS grid (falls back to the synthesizer test grid)."""
    d = _grid_dir()
    if d is None:
        pytest.skip("No SYNTHESIZER_GRID_DIR")
    for name in (GRID_NAME, "test_grid.hdf5"):
        if os.path.exists(os.path.join(d, name)):
            return Grid(name, grid_dir=d)
    pytest.skip("No suitable grid found")


# ---------------------------------------------------------------------------
# Random SFH generators (incl. edge cases)
# ---------------------------------------------------------------------------
def _maxa(rng, lo=1.0):
    return 10 ** rng.uniform(np.log10(lo), np.log10(AGE_MAX)) * Myr


def _gen(name, rng, i):
    maxa = _maxa(rng)
    if name == "LogNormal":
        edge = [
            dict(tau=0.05, peak_age=0.999 * maxa, max_age=maxa),  # sharp, at max_age
            dict(tau=0.05, peak_age=0.5 * Myr, max_age=1 * Myr),  # very young
            dict(tau=2.5, peak_age=0.001 * maxa, max_age=maxa),
        ]
        if i < len(edge):
            return SFH.LogNormal(**edge[i])
        return SFH.LogNormal(
            tau=rng.uniform(0.05, 2.5), peak_age=rng.uniform(0.001, 0.999) * maxa, max_age=maxa
        )
    if name == "DelayedExponential":
        edge = [dict(tau=10 * Myr, max_age=13000 * Myr), dict(tau=1 * Myr, max_age=2 * Myr)]
        if i < len(edge):
            return SFH.DelayedExponential(**edge[i])
        return SFH.DelayedExponential(tau=10 ** rng.uniform(0, 4.3) * Myr, max_age=maxa)
    if name == "DecliningExponential":
        return SFH.DecliningExponential(tau=10 ** rng.uniform(0, 4.3) * Myr, max_age=maxa)
    if name == "Exponential":
        tau = 10 ** rng.uniform(0, 4) * Myr
        if rng.uniform() < 0.5:  # rising; keep exp(max_age / |tau|) finite
            tau = -max(tau, maxa / 300)
        return SFH.Exponential(tau=tau, max_age=maxa, min_age=rng.uniform(0, 0.5) * maxa)
    if name == "TruncatedExponential":
        return SFH.TruncatedExponential(
            tau=10 ** rng.uniform(0, 4) * Myr, max_age=maxa, min_age=rng.uniform(0, 0.9) * maxa
        )
    if name == "Constant":
        edge = [dict(max_age=1 * Myr), dict(max_age=1.05 * Myr, min_age=1.0 * Myr)]
        if i < len(edge):
            return SFH.Constant(**edge[i])
        return SFH.Constant(max_age=maxa, min_age=rng.uniform(0, 0.9) * maxa * rng.choice([0, 1]))
    if name == "Gaussian":
        return SFH.Gaussian(
            peak_age=rng.uniform(0, 1) * maxa, sigma=10 ** rng.uniform(-2, 0) * maxa, max_age=maxa
        )
    if name == "DoublePowerLaw":
        # Integrable at age 0 only for min(alpha, beta) < 1 (otherwise falls back)
        a, b = 10 ** rng.uniform(-1, np.log10(0.99)), 10 ** rng.uniform(-1, 3)
        if rng.uniform() < 0.5:
            a, b = b, a
        return SFH.DoublePowerLaw(peak_age=rng.uniform(0.01, 0.99) * AGE_MAX * Myr, alpha=a, beta=b)
    if name == "Continuity":
        n = int(rng.integers(3, 9))
        top = max(maxa.to("yr").value, 2e7)
        e = np.concatenate([[1.0], np.logspace(6.5, np.log10(top), n)])
        agebins = unyt_array(np.array([e[:-1], e[1:]]).T, "yr")
        return SFH.Continuity.init_from_prior(agebins=agebins)
    if name == "ContinuityFlex":
        return SFH.ContinuityFlex.init_from_prior(
            N_flex_bins=int(rng.integers(2, 5)),
            fixed_young_bin=[0.0, 7.5],
            fixed_old_bin=[9.7, 10.136],
        )
    if name == "Dirichlet":
        e = np.concatenate([[1.0], np.logspace(7, 10.1, 5)])
        agebins = unyt_array(np.array([e[:-1], e[1:]]).T, "yr")
        return SFH.Dirichlet.init_from_prior(agebins=agebins)
    if name == "ContinuityPSB":
        return SFH.ContinuityPSB.init_from_prior()
    if name == "DenseBasis":
        return SFH.DenseBasis.init_from_prior(
            N_tx=3, log_mass=10, log_sfr=rng.uniform(-2, 2), redshift=rng.uniform(0.1, 10)
        )
    if name == "CombinedSFH":
        return SFH.CombinedSFH(
            [
                SFH.LogNormal(tau=rng.uniform(0.1, 1), peak_age=0.5 * maxa, max_age=maxa),
                SFH.Constant(max_age=rng.uniform(0.01, 1) * maxa),
            ],
            weights=[1.0, 1e-9],
        )
    raise ValueError(name)


# (class name, number of random draws). DenseBasis quad is ~0.1 s per galaxy.
CASES = [
    ("LogNormal", 60),
    ("DelayedExponential", 50),
    ("DecliningExponential", 40),
    ("Exponential", 30),
    ("TruncatedExponential", 20),
    ("Constant", 40),
    ("Gaussian", 25),
    ("DoublePowerLaw", 40),
    ("Continuity", 25),
    ("ContinuityFlex", 10),
    ("Dirichlet", 10),
    ("ContinuityPSB", 10),
    ("DenseBasis", 8),
    ("CombinedSFH", 10),
]


def _breakpoints(sfh):
    """Natural breakpoints of an SFH (for the refined quad reference)."""
    pts = []
    for attr in ("min_age", "max_age", "peak_age"):
        v = getattr(sfh, attr, None)
        if v is not None and np.isfinite(v):
            pts.append(float(v))
    if hasattr(sfh, "tau") and hasattr(sfh, "max_age") and not isinstance(sfh, SFH.LogNormal):
        pts += list(sfh.max_age - abs(sfh.tau) * np.array([0.01, 0.1, 1, 3, 10, 30]))
        pts += list(sfh.min_age + abs(sfh.tau) * np.array([0.01, 0.1, 1, 3, 10, 30]))
    if isinstance(sfh, SFH.LogNormal):
        pts += list(sfh.max_age - np.exp(sfh.t_0 + sfh.tau * np.linspace(-6, 6, 13)))
    if isinstance(sfh, SFH.Gaussian):
        pts += list(sfh.peak_age + sfh.sigma * np.linspace(-6, 6, 13))
    if hasattr(sfh, "bin_edges"):
        pts += list(np.asarray(sfh.bin_edges, dtype=float))
    if hasattr(sfh, "sfhs"):
        for m in sfh.sfhs:
            pts += _breakpoints(m)
    return pts


def _refined_reference(sfh, edges):
    pts = np.array(_breakpoints(sfh))
    out = np.zeros(edges.size - 1)
    for i in range(out.size):
        a, b = edges[i], edges[i + 1]
        inner = np.unique(pts[(pts > a) & (pts < b)])
        knots = np.concatenate([[a], inner, [b]])
        if hasattr(sfh, "finegrid"):  # interpolated SFHs: split at knots
            fg = np.asarray(sfh.finegrid)
            knots = np.unique(np.concatenate([knots, fg[(fg > a) & (fg < b)]]))
        out[i] = sum(
            integrate.quad(sfh.get_sfr, x0, x1, limit=200, epsabs=0, epsrel=1e-11)[0]
            for x0, x1 in zip(knots[:-1], knots[1:])
        )
    return out


def _l1(a, b):
    return np.abs(a - b).sum() / np.abs(b).sum()


# ---------------------------------------------------------------------------
# Metallicity weights
# ---------------------------------------------------------------------------
def test_delta_constant_bitwise(grid):
    """DeltaConstant / float weights are identical to Synthesizer's CIC."""
    from synthesizer.particle import Stars as ParticleStars

    zs = list(grid.metallicities) + list(10 ** np.random.default_rng(0).uniform(-6, -1, 40))
    zs += [0.5 * grid.metallicities[0], 2 * grid.metallicities[-1]]
    for z in zs:
        p = ParticleStars(
            initial_masses=np.array([1.0]) * Msun,
            ages=np.array([0.0]) * yr,
            metallicities=np.array([z]),
        )
        ref = p.get_metal_dist(grid.metallicities)
        got = F.zdist_weights(ZDist.DeltaConstant(metallicity=z), grid.metallicities)
        np.testing.assert_array_equal(got, ref)
        np.testing.assert_array_equal(F.zdist_weights(float(z), grid.metallicities), ref)


def test_normal_zdist_matches_quad(grid):
    """ZDist.Normal analytic weights match Synthesizer's quad loop."""
    rng = np.random.default_rng(1)
    Z = grid.metallicities
    e = np.concatenate([[0.0], 0.5 * (Z[1:] + Z[:-1]), [Z[-1]]])
    for _ in range(20):
        zd = ZDist.Normal(mean=10 ** rng.uniform(-4, -1.4), sigma=10 ** rng.uniform(-4, -2))
        ref = np.array(
            [integrate.quad(zd.get_dist_weight, e[i], e[i + 1])[0] for i in range(Z.size)]
        )
        got = F.zdist_weights(zd, Z)
        np.testing.assert_allclose(got, ref, rtol=1e-7, atol=1e-9 * ref.sum())


# ---------------------------------------------------------------------------
# SFH accuracy
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,n", CASES, ids=[c[0] for c in CASES])
def test_sfh_accuracy(grid, name, n):
    """Fast SFZH agrees with Synthesizer (or a refined quad where it is off)."""
    if not hasattr(SFH, name):
        pytest.skip(f"SFH.{name} not available")
    if name == "DenseBasis":
        pytest.importorskip("dense_basis")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(sum(map(ord, name)))
    edges = F.age_bin_edges(grid.log10ages)
    errs, ref_errs, synth_bad, n_fallback = [], [], 0, 0
    zd = ZDist.DeltaConstant(log10metallicity=-2.3)
    for i in range(n):
        sfh = _gen(name, rng, i)
        new = F.make_parametric_stars(
            sfh, zd, grid.log10ages, grid.metallicities, initial_mass=1e9 * Msun
        )
        old = Stars(
            log10ages=grid.log10ages,
            metallicities=grid.metallicities,
            sf_hist=sfh,
            metal_dist=zd,
            initial_mass=1e9 * Msun,
        )
        assert type(new) is Stars
        if not F.is_supported_sfh(sfh):
            n_fallback += 1
        try:
            F.sfh_bin_masses(sfh, grid.log10ages, fallback=False)
        except TypeError:
            n_fallback += 1
            np.testing.assert_array_equal(new.sfzh, old.sfzh)
            continue
        err = _l1(new.sfzh, old.sfzh)
        errs.append(err)
        if err >= TOL_SYNTH or name in ("Constant", "DecliningExponential"):
            ref = _refined_reference(sfh, edges)
            ref_sfzh = ref[:, None] * F.zdist_weights(zd, grid.metallicities)[None, :]
            ref_sfzh *= 1e9 / ref_sfzh.sum()
            synth_err = _l1(old.sfzh, ref_sfzh)
            new_err = _l1(new.sfzh, ref_sfzh)
            ref_errs.append(new_err)
            if err >= TOL_SYNTH:
                assert synth_err > 5e-4, (name, sfh.parameters, err, synth_err)
                assert new_err < TOL_REF, (name, sfh.parameters, new_err)
                synth_bad += 1
        else:
            assert err < TOL_SYNTH
    errs = np.array(errs)
    SUMMARY[name] = (len(errs), n_fallback, np.median(errs), errs.max(), synth_bad, ref_errs)
    print(
        f"\n{name}: n={len(errs)} fallback={n_fallback} L1 vs synthesizer quad: "
        f"median={np.median(errs):.1e} p90={np.percentile(errs, 90):.1e} "
        f"max={errs.max():.1e}; synthesizer-quad failures={synth_bad}"
        + (f"; max err vs refined ref={max(ref_errs):.1e}" if ref_errs else "")
    )


def test_photometry(grid):
    """Photometry of fast vs quad Stars agrees to < 1e-3 mag."""
    from synthesizer.emission_models import IncidentEmission, PacmanEmission
    from synthesizer.emission_models.attenuation import Calzetti2000
    from synthesizer.instruments import FilterCollection

    filters = FilterCollection(
        tophat_dict={
            "UV": {"lam_eff": 1500 * angstrom, "lam_fwhm": 300 * angstrom},
            "U": {"lam_eff": 3650 * angstrom, "lam_fwhm": 660 * angstrom},
            "V": {"lam_eff": 5510 * angstrom, "lam_fwhm": 880 * angstrom},
            "K": {"lam_eff": 22000 * angstrom, "lam_fwhm": 4000 * angstrom},
        },
        new_lam=grid.lam,
    )
    models = {
        "incident": IncidentEmission(grid),
        "attenuated": PacmanEmission(grid, tau_v="tau_v", dust_curve=Calzetti2000(), fesc=0.0),
    }
    rng = np.random.default_rng(7)
    worst = 0.0
    for name in (
        "LogNormal",
        "DelayedExponential",
        "DecliningExponential",
        "Constant",
        "Continuity",
    ):
        for i in range(4):
            sfh = _gen(name, rng, i + 10)
            zd = ZDist.DeltaConstant(log10metallicity=rng.uniform(-4, -1.4))
            kw = dict(initial_mass=1e9 * Msun, tau_v=rng.uniform(0, 1), fesc_ly_alpha=1.0)
            new = F.make_parametric_stars(sfh, zd, grid.log10ages, grid.metallicities, **kw)
            old = Stars(
                log10ages=grid.log10ages,
                metallicities=grid.metallicities,
                sf_hist=sfh,
                metal_dist=zd,
                **kw,
            )
            for label, model in models.items():
                mags = []
                for st in (new, old):
                    gal = Galaxy(stars=st, redshift=0.0)
                    gal.stars.get_spectra(model)
                    lnu = gal.stars.spectra[label].get_photo_lnu(filters).photo_lnu.value
                    mags.append(-2.5 * np.log10(lnu))
                d = np.nanmax(np.abs(mags[0] - mags[1]))
                worst = max(worst, d)
                assert d < 1e-3, (name, label, sfh.parameters, d)
    print(f"\nmax photometric difference: {worst:.2e} mag")


# ---------------------------------------------------------------------------
# Attribute parity, normalisation, fallbacks
# ---------------------------------------------------------------------------
def test_attribute_parity(grid):
    """Downstream attributes and methods agree with the quad-built Stars."""
    sfh = SFH.LogNormal(tau=0.4, peak_age=300 * Myr, max_age=2000 * Myr)
    zd = ZDist.DeltaConstant(log10metallicity=-2.1)
    kw = dict(initial_mass=3e9 * Msun, tau_v=0.3, fesc=0.1)
    new = F.make_parametric_stars(sfh, zd, grid.log10ages, grid.metallicities, **kw)
    old = Stars(
        log10ages=grid.log10ages,
        metallicities=grid.metallicities,
        sf_hist=sfh,
        metal_dist=zd,
        **kw,
    )
    assert new.sf_hist_func is sfh and new.metal_dist_func is zd
    assert new._instant_sf is None and old._instant_sf is None
    assert new._instant_metallicity == old._instant_metallicity
    assert new.tau_v == old.tau_v and new.fesc == old.fesc
    assert new.metallicity_grid_type == old.metallicity_grid_type
    assert set(vars(new)) == set(vars(old))
    np.testing.assert_allclose(new.initial_mass, old.initial_mass, rtol=1e-12)
    np.testing.assert_allclose(new.sfzh.sum(), 3e9, rtol=1e-12)
    np.testing.assert_allclose(new.sfzh_normalisation, old.sfzh_normalisation, rtol=1e-6)
    np.testing.assert_allclose(new.sf_hist, old.sf_hist, rtol=1e-5, atol=1e-8 * 3e9)
    np.testing.assert_allclose(new.metal_dist, old.metal_dist, rtol=1e-6)
    np.testing.assert_allclose(
        new.calculate_average_sfr((0, 1e8)), old.calculate_average_sfr((0, 1e8)), rtol=1e-5
    )
    np.testing.assert_allclose(new.calculate_mean_age(), old.calculate_mean_age(), rtol=1e-6)
    np.testing.assert_allclose(
        new.calculate_mean_metallicity(), old.calculate_mean_metallicity(), rtol=1e-10
    )
    np.testing.assert_allclose(
        new.calculate_surviving_mass(grid), old.calculate_surviving_mass(grid), rtol=1e-6
    )
    early_new = new.get_at_earlier_time(100 * Myr)
    early_old = old.get_at_earlier_time(100 * Myr)
    np.testing.assert_allclose(early_new.sfzh, early_old.sfzh, rtol=1e-5, atol=1e-8 * 3e9)

    # Surviving mass normalisation path
    new = F.make_parametric_stars(
        sfh, zd, grid.log10ages, grid.metallicities, surviving_mass=1e9 * Msun, grid=grid
    )
    old = Stars(
        log10ages=grid.log10ages,
        metallicities=grid.metallicities,
        sf_hist=sfh,
        metal_dist=zd,
        surviving_mass=1e9 * Msun,
        grid=grid,
    )
    np.testing.assert_allclose(new.initial_mass, old.initial_mass, rtol=1e-6)
    np.testing.assert_allclose((new.sfzh * grid.stellar_fraction).sum(), 1e9, rtol=1e-10)

    # No mass given: initial_mass is the raw integral
    new = F.make_parametric_stars(sfh, zd, grid.log10ages, grid.metallicities)
    old = Stars(
        log10ages=grid.log10ages, metallicities=grid.metallicities, sf_hist=sfh, metal_dist=zd
    )
    np.testing.assert_allclose(new.initial_mass, old.initial_mass, rtol=1e-6)


def test_normal_zdist_stars(grid):
    """End-to-end with a Normal metallicity distribution."""
    sfh = SFH.DelayedExponential(tau=500 * Myr, max_age=3000 * Myr)
    zd = ZDist.Normal(mean=0.01, sigma=0.005)
    new = F.make_parametric_stars(
        sfh, zd, grid.log10ages, grid.metallicities, initial_mass=1e9 * Msun
    )
    old = Stars(
        log10ages=grid.log10ages,
        metallicities=grid.metallicities,
        sf_hist=sfh,
        metal_dist=zd,
        initial_mass=1e9 * Msun,
    )
    assert _l1(new.sfzh, old.sfzh) < 1e-6


def test_age_offset(grid):
    """age_offset reproduces Synthesizer's shifted integration."""
    edges = F.age_bin_edges(grid.log10ages)
    for sfh in (
        SFH.LogNormal(tau=0.3, peak_age=300 * Myr, max_age=2000 * Myr),
        SFH.Constant(max_age=500 * Myr, min_age=50 * Myr),
    ):
        for off in (1e6, 3e7, 3e8):
            ref = _refined_reference(sfh, edges + off)
            got = F.sfh_bin_masses(sfh, grid.log10ages, age_offset=off * yr)
            assert _l1(got, ref) < 1e-6
    sfh = SFH.LogNormal(tau=0.3, peak_age=300 * Myr, max_age=2000 * Myr)
    zd = ZDist.DeltaConstant(log10metallicity=-2.0)
    fast = F.make_parametric_stars(
        sfh, zd, grid.log10ages, grid.metallicities, initial_mass=1e9 * Msun, subclass=True
    )
    assert isinstance(fast, F.FastParametricStars) and isinstance(fast, Stars)
    old = Stars(
        log10ages=grid.log10ages,
        metallicities=grid.metallicities,
        sf_hist=sfh,
        metal_dist=zd,
        initial_mass=1e9 * Msun,
    )
    assert _l1(fast.sfzh, old.sfzh) < 1e-6
    assert (
        _l1(fast.get_at_earlier_time(200 * Myr).sfzh, old.get_at_earlier_time(200 * Myr).sfzh)
        < 1e-6
    )
    np.testing.assert_allclose(
        fast.calculate_initial_mass_at_age(200 * Myr),
        old.calculate_initial_mass_at_age(200 * Myr),
        rtol=1e-6,
    )


class _MyLogNormal(SFH.LogNormal):
    """User subclass: must fall back (it may override _sfr)."""


def test_fallbacks(grid):
    """Unsupported inputs reproduce Synthesizer's result exactly."""
    warnings.filterwarnings("ignore")
    zd = ZDist.DeltaConstant(log10metallicity=-2.0)
    cases = [
        (_MyLogNormal(tau=0.3, peak_age=100 * Myr, max_age=1000 * Myr), zd),
        (SFH.DoublePowerLaw(peak_age=3000 * Myr, alpha=2.0, beta=30.0), zd),  # diverges at 0
        (30 * Myr, zd),  # instantaneous burst
        (30 * Myr, 0.01),  # instantaneous burst + float Z
        (SFH.Constant(max_age=100 * Myr), 0.004),  # float Z (fast path)
    ]
    for sfh, z in cases:
        new = F.make_parametric_stars(
            sfh, z, grid.log10ages, grid.metallicities, initial_mass=1e9 * Msun
        )
        old = Stars(
            log10ages=grid.log10ages,
            metallicities=grid.metallicities,
            sf_hist=sfh,
            metal_dist=z,
            initial_mass=1e9 * Msun,
        )
        if F.is_supported_sfh(sfh) and not isinstance(sfh, SFH.DoublePowerLaw):
            assert _l1(new.sfzh, old.sfzh) < 1e-6
        else:
            np.testing.assert_array_equal(new.sfzh, old.sfzh)
    assert not F.is_supported_sfh(_MyLogNormal(tau=0.3, peak_age=100 * Myr, max_age=1000 * Myr))
    assert F.fast_sfzh(30 * Myr, zd, grid.log10ages, grid.metallicities) is None
    # sfh_bin_masses always returns an answer (quad loop fallback)
    s = _MyLogNormal(tau=0.3, peak_age=100 * Myr, max_age=1000 * Myr)
    np.testing.assert_allclose(
        F.sfh_bin_masses(s, grid.log10ages),
        F._quad_bin_masses(s, F.age_bin_edges(grid.log10ages), 0.0),
    )
    with pytest.raises(TypeError):
        F.sfh_bin_masses(s, grid.log10ages, fallback=False)
