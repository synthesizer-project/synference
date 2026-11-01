"""Tests for the tabulated IGM transmission (synference.fast_igm)."""

import os
import pickle
import sys

import numpy as np
import pytest
from astropy.cosmology import Planck18
from synthesizer.emission_models import PacmanEmission
from synthesizer.emission_models.attenuation import Calzetti2000, Inoue14, Madau96
from synthesizer.emissions import Sed
from synthesizer.instruments import FilterCollection, Instrument
from synthesizer.parametric import SFH, Galaxy, Stars, ZDist
from unyt import Angstrom, Hz, Msun, Myr, erg, s

try:
    from synference.fast_igm import TabulatedIGM
except ImportError:  # pragma: no cover - e.g. optional deps of the package missing
    import importlib.util

    _path = os.path.join(os.path.dirname(__file__), "..", "src", "synference", "fast_igm.py")
    _spec = importlib.util.spec_from_file_location("synference_fast_igm", _path)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["synference_fast_igm"] = _mod  # so instances can be pickled
    _spec.loader.exec_module(_mod)
    TabulatedIGM = _mod.TabulatedIGM


# A modest log-spaced rest-frame grid (fast to tabulate)
LAM = np.logspace(np.log10(80.0), np.log10(5e5), 1500)
Z_MAX = 15.0
DZ = 0.05


@pytest.fixture(scope="module")
def tab_igm():
    """Tabulated Inoue14 prepared for LAM."""
    return TabulatedIGM(Inoue14, z_max=Z_MAX, dz=DZ, lam=LAM * Angstrom)


@pytest.fixture(scope="module")
def sf_sed():
    """A young, star-forming like SED: blue power law with a Lyman break."""
    lnu = 1e28 * (LAM / 1500.0) ** 0.5  # beta ~ -2.5 in f_lambda
    lnu[LAM < 912.0] *= 0.05  # partial Lyman-continuum escape
    lnu *= 1 + 5 * np.exp(-0.5 * ((LAM - 1215.67) / 2.0) ** 2)  # Lya
    return Sed(LAM * Angstrom, lnu * erg / s / Hz)


@pytest.fixture(scope="module")
def tophat_filters():
    """Broad and medium top-hat bands spanning u to 5 micron."""
    tophat = {}
    for cen, frac in [
        (3550, 0.18),
        (4800, 0.30),
        (6200, 0.25),
        (7700, 0.20),
        (8900, 0.12),
        (9800, 0.08),
        (11500, 0.22),
        (14000, 0.10),
        (15000, 0.21),
        (16200, 0.10),
        (20000, 0.23),
        (27700, 0.25),
        (35600, 0.22),
        (41000, 0.05),
        (44400, 0.24),
    ]:
        w = cen * frac
        tophat[f"th{cen}_{int(100 * frac)}"] = {
            "lam_min": (cen - w / 2) * Angstrom,
            "lam_max": (cen + w / 2) * Angstrom,
        }
    return FilterCollection(tophat_dict=tophat, new_lam=np.logspace(3, 5, 6000) * Angstrom)


@pytest.fixture(scope="module")
def small_grid():
    """The synthesizer test grid (downloaded by conftest's fixture if needed)."""
    from synthesizer import get_grids_dir
    from synthesizer.grid import Grid

    grid_dir = get_grids_dir()
    if not os.path.exists(os.path.join(grid_dir, "test_grid.hdf5")):
        pytest.skip("synthesizer test grid not available")
    new_lam = np.logspace(np.log10(80.0), np.log10(5e5), 2000) * Angstrom
    return Grid(grid_name="test_grid", grid_dir=grid_dir, new_lam=new_lam)


def _galaxies(grid, zs):
    gals = []
    for i, z in enumerate(zs):
        stars = Stars(
            log10ages=grid.log10ages,
            metallicities=grid.metallicities,
            sf_hist=SFH.Constant(max_age=(50 + 30 * i) * Myr),
            metal_dist=ZDist.DeltaConstant(log10metallicity=-2.5),
            initial_mass=1e9 * Msun,
        )
        gals.append(Galaxy(stars=stars, redshift=z))
    return gals


def _model(grid):
    return PacmanEmission(grid=grid, tau_v=0.2, dust_curve=Calzetti2000(), fesc=0.1)


# ----------------------------------------------------------------------------
# Basic behaviour
# ----------------------------------------------------------------------------
def test_not_callable_and_interface(tab_igm):
    """Instances must not be callable (get_fnu would call igm())."""
    assert not callable(tab_igm)
    assert hasattr(tab_igm, "get_transmission")
    assert tab_igm.name == "TabulatedInoue14"
    # Class or instance bases are both accepted
    assert isinstance(TabulatedIGM(Inoue14(scale_tau=1.0), dz=0.5).base, Inoue14)


def test_red_side_is_exactly_one():
    """Inoue14 and Madau96 are exactly transparent redward of Lya."""
    red = LAM[LAM >= 1216.0]
    for base in (Inoue14(), Madau96()):
        for z in (0.1, 3.0, 7.5, 14.0):
            t = base.get_transmission(z, red * (1 + z) * Angstrom)
            assert np.all(t == 1.0)


@pytest.mark.parametrize("base", [Inoue14, Madau96])
def test_transmission_accuracy(base):
    """Per-wavelength transmission vs the exact model at random redshifts."""
    lam = LAM[::3]
    tab = TabulatedIGM(base, z_max=Z_MAX, dz=DZ, lam=lam)
    exact = base()
    rng = np.random.default_rng(42)
    max_err = 0.0
    for z in np.concatenate([rng.uniform(0.01, Z_MAX, 25), [1.2, 4.7, 6.0]]):
        lam_obs = lam * (1 + z) * Angstrom
        t_tab = tab.get_transmission(z, lam_obs)
        t_ex = exact.get_transmission(z, lam_obs)
        assert t_tab.shape == t_ex.shape
        assert np.all((t_tab >= 0) & (t_tab <= 1))
        np.testing.assert_array_equal(t_tab[lam >= 1216.0], 1.0)
        max_err = max(max_err, np.abs(t_tab - t_ex).max())
    assert tab.n_exact_calls == 0
    assert max_err < 2.5e-3


def test_photometric_accuracy(tab_igm, sf_sed, tophat_filters):
    """Band fluxes vs exact Inoue14 to < 1e-3 mag (bands not dropped out)."""
    exact = Inoue14()
    rng = np.random.default_rng(1)
    worst = 0.0
    for z in np.concatenate([rng.uniform(0.05, Z_MAX, 40), [1.2, 4.7]]):
        sed = Sed(sf_sed.lam, sf_sed.lnu)
        sed.get_fnu(Planck18, z, igm=None)
        fnu0 = np.asarray(sed._fnu, dtype=np.float64)
        obslam = sed.obslam
        arr = np.vstack(
            [
                fnu0,
                fnu0 * exact.get_transmission(z, obslam),
                fnu0 * tab_igm.get_transmission(z, obslam),
            ]
        )
        f = tophat_filters.apply_filters(
            np.ascontiguousarray(arr), nu=np.asarray(sed._obsnu, dtype=np.float64)
        )
        f0, fe, ft = f[:, 0], f[:, 1], f[:, 2]
        ok = fe > 1e-2 * f0  # band not (almost) completely dropped out
        dmag = np.abs(2.5 * np.log10(ft[ok] / fe[ok]))
        if dmag.size:
            worst = max(worst, dmag.max())
        # Absolute flux error relative to the unattenuated flux everywhere
        assert np.all(np.abs(ft - fe) <= 1e-3 * f0 + 1e-30)
    assert worst < 1e-3


# ----------------------------------------------------------------------------
# Fallbacks, lazy building, caching and pickling
# ----------------------------------------------------------------------------
def test_out_of_table_fallback(tab_igm):
    """Redshifts outside the table use the exact base model."""
    exact = Inoue14()
    z = Z_MAX + 2.0
    lam_obs = LAM * (1 + z) * Angstrom
    n0 = tab_igm.n_exact_calls
    np.testing.assert_array_equal(
        tab_igm.get_transmission(z, lam_obs), exact.get_transmission(z, lam_obs)
    )
    assert tab_igm.n_exact_calls == n0 + 1


def test_lazy_build_and_arbitrary_arrays():
    """Unseen arrays are exact on first sight and tabulated on the second."""
    tab = TabulatedIGM(Inoue14, z_max=8.0, dz=0.1)
    exact = Inoue14()
    lam = np.linspace(700.0, 3000.0, 200)
    t1 = tab.get_transmission(3.0, lam * 4.0 * Angstrom)
    assert tab.n_exact_calls == 1 and len(tab.tables) == 0
    np.testing.assert_array_equal(t1, exact.get_transmission(3.0, lam * 4.0 * Angstrom))
    # Second sight (different redshift, same rest-frame array) builds
    t2 = tab.get_transmission(5.0, lam * 6.0 * Angstrom)
    assert len(tab.tables) == 1 and tab.n_table_calls == 1
    np.testing.assert_allclose(t2, exact.get_transmission(5.0, lam * 6.0 * Angstrom), atol=5e-3)
    # Unitless input (Angstrom) and other length units work too
    t3 = tab.get_transmission(5.0, lam * 6.0)
    t4 = tab.get_transmission(5.0, (lam * 6.0 * Angstrom).to("um"))
    np.testing.assert_allclose(t3, t2)
    np.testing.assert_allclose(t4, t2)
    # Scalar-like / 2D shapes are preserved
    assert tab.get_transmission(5.0, np.array([5000.0]) * Angstrom).shape == (1,)


@pytest.mark.parametrize("n_b", [200, 150], ids=["same_shape", "different_shape"])
def test_alternating_unseen_arrays_build(n_b):
    """Two pending arrays seen alternately both reach the build path.

    Regression test: removing a pending entry with ``list.remove`` compared
    the stored ndarrays with ``==`` and raised when another array was pending
    ahead of it.
    """
    tab = TabulatedIGM(Inoue14, z_max=8.0, dz=0.1, min_hits=2)
    exact = Inoue14()
    lam_a = np.linspace(700.0, 3000.0, 200)
    lam_b = np.linspace(650.0, 2500.0, n_b)
    for z, lam in [(3.0, lam_a), (3.0, lam_b), (5.0, lam_b), (5.0, lam_a)]:
        t = tab.get_transmission(z, lam * (1 + z) * Angstrom)
        np.testing.assert_allclose(
            t, exact.get_transmission(z, lam * (1 + z) * Angstrom), atol=5e-3
        )
    assert len(tab.tables) == 2
    assert tab.n_table_calls == 2 and tab.n_exact_calls == 2


def test_pickle_round_trip(tab_igm):
    """Pickled copies give identical results and don't rebuild."""
    z = 7.3
    lam_obs = LAM * (1 + z) * Angstrom
    ref = tab_igm.get_transmission(z, lam_obs)
    blob = pickle.dumps(tab_igm)
    clone = pickle.loads(blob)
    n_exact = clone.n_exact_calls
    np.testing.assert_array_equal(clone.get_transmission(z, lam_obs), ref)
    assert clone.n_exact_calls == n_exact
    assert not callable(clone)
    # Without tables the pickle is tiny and the clone rebuilds lazily
    light = TabulatedIGM(Inoue14, z_max=Z_MAX, dz=DZ, pickle_tables=False)
    light._tables, light._order = tab_igm._tables, tab_igm._order
    small = pickle.dumps(light)
    assert len(small) < len(blob) / 10
    assert pickle.loads(small).tables == []


def test_disk_cache(tmp_path):
    """Tables written to cache_dir are reused by new instances."""
    lam = LAM[::4]
    a = TabulatedIGM(Inoue14, z_max=6.0, dz=0.1, cache_dir=str(tmp_path), lam=lam)
    files = list(tmp_path.glob("tabigm_*.npz"))
    assert len(files) == 1
    b = TabulatedIGM(Inoue14, z_max=6.0, dz=0.1, cache_dir=str(tmp_path))

    def no_build(*args, **kwargs):
        raise AssertionError("should have loaded the table from cache_dir")

    b._build = no_build
    b.prepare(lam)
    lam_obs = lam * 4.5 * Angstrom
    np.testing.assert_array_equal(
        a.get_transmission(3.5, lam_obs), b.get_transmission(3.5, lam_obs)
    )


def test_bad_red_cut_raises():
    """A cut redward of which T != 1 is detected at build time."""
    with pytest.raises(ValueError):
        TabulatedIGM(Inoue14, z_max=5.0, dz=0.5, lam_rest_max=1000.0, lam=LAM)


# ----------------------------------------------------------------------------
# Integration with Synthesizer
# ----------------------------------------------------------------------------
def test_sed_get_fnu_and_photo(tab_igm, sf_sed, tophat_filters):
    """Works as igm= in Sed.get_fnu and the subsequent get_photo_fnu."""
    for z in (2.5, 6.8, 11.0):
        s_ex = Sed(sf_sed.lam, sf_sed.lnu)
        s_tab = Sed(sf_sed.lam, sf_sed.lnu)
        f_ex = s_ex.get_fnu(Planck18, z, igm=Inoue14)
        n_tab = tab_igm.n_table_calls
        f_tab = s_tab.get_fnu(Planck18, z, igm=tab_igm)
        assert tab_igm.n_table_calls == n_tab + 1
        assert f_tab.units == f_ex.units
        scale = f_ex.max()
        assert np.abs(f_tab - f_ex).max() <= 3e-3 * scale
        p_ex = s_ex.get_photo_fnu(tophat_filters)
        p_tab = s_tab.get_photo_fnu(tophat_filters)
        for f in tophat_filters.filter_codes:
            assert abs(p_tab[f] - p_ex[f]) <= 1e-3 * abs(p_ex[f]) + 1e-6 * scale


def test_galaxy_observed_spectra_and_lines(small_grid):
    """Works as igm= in Galaxy.get_observed_spectra / get_observed_lines."""
    tab = TabulatedIGM(Inoue14, z_max=Z_MAX, dz=0.1, min_hits=1)
    model = _model(small_grid)
    line_ids = [lid for lid in small_grid.available_lines if "1215" not in lid][:4]
    for igm in (Inoue14, tab):
        gal = _galaxies(small_grid, [6.5])[0]
        gal.stars.get_spectra(model)
        gal.get_observed_spectra(Planck18, igm=igm)
        gal.stars.get_lines(line_ids, model)
        gal.get_observed_lines(Planck18, igm=igm)
        if igm is Inoue14:
            ref_fnu = gal.stars.spectra["emergent"].fnu.copy()
            ref_flux = gal.stars.lines["emergent"].flux.copy()
        else:
            fnu = gal.stars.spectra["emergent"].fnu
            flux = gal.stars.lines["emergent"].flux
    assert tab.n_table_calls > 0
    np.testing.assert_allclose(fnu, ref_fnu, rtol=5e-3, atol=1e-3 * ref_fnu.max().value)
    np.testing.assert_allclose(flux, ref_flux, rtol=5e-3, atol=1e-6 * ref_flux.max().value)


def test_pipeline(small_grid, tophat_filters):
    """Works as igm= in a small Pipeline (nthreads=1) incl. photometry & lines."""
    from synthesizer.pipeline import Pipeline

    tab = TabulatedIGM(Inoue14, z_max=Z_MAX, dz=0.1, lam=small_grid.lam)
    inst = Instrument("TH", filters=tophat_filters)
    line_ids = [lid for lid in small_grid.available_lines if "1215" not in lid][:3]
    zs = [3.0, 6.2, 9.4]
    out = {}
    for name, igm in (("exact", Inoue14), ("tab", tab)):
        gals = _galaxies(small_grid, zs)
        pipe = Pipeline(emission_model=_model(small_grid), nthreads=1, verbose=0)
        pipe.get_observed_spectra(Planck18, igm=igm)
        pipe.get_photometry_fluxes(inst, igm=igm)
        pipe.get_observed_lines(Planck18, igm=igm, line_ids=line_ids)
        pipe.add_galaxies(gals)
        pipe.run()
        out[name] = gals
    # Only the first sighting of the line wavelength array is computed exactly
    assert tab.n_table_calls > 0 and tab.n_exact_calls <= 1
    for g_ex, g_tab in zip(out["exact"], out["tab"]):
        ph_ex = g_ex.stars.photo_fnu["emergent"]
        ph_tab = g_tab.stars.photo_fnu["emergent"]
        for f in tophat_filters.filter_codes:
            assert abs(ph_tab[f] - ph_ex[f]) <= 1e-3 * abs(ph_ex[f]) + 1e-30 * ph_ex[f].units
        np.testing.assert_allclose(
            g_tab.stars.lines["emergent"].flux, g_ex.stars.lines["emergent"].flux, rtol=5e-3
        )
