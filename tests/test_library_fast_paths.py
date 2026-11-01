"""Tests for the fast/parallel library generation paths and audit bug fixes."""

import h5py
import numpy as np
import pytest
from astropy.cosmology import Planck18
from synthesizer.emission_models import PacmanEmission, TotalEmission
from synthesizer.emission_models.attenuation import Calzetti2000, PowerLaw
from synthesizer.emission_models.generators.dust.blackbody import Blackbody
from synthesizer.emission_models.generators.dust.casey12 import Casey12
from synthesizer.emission_models.generators.dust.greybody import Greybody
from synthesizer.parametric import SFH, Galaxy, Stars, ZDist
from unyt import K, Msun, Myr, um

from synference import GalaxyBasis, GalaxySimulator, draw_from_hypercube
from synference.fast_igm import TabulatedIGM
from synference.library import (
    _dust_curve_param_clashes,
    _line_dependency_labels,
    calculate_line_luminosity,
    create_galaxy,
)
from synference.utils import (
    _dust_emission_init_params,
    load_dust_emission_params,
    save_emission_model,
)

# Synthesizer serialises the child labels of each emission model from a set,
# so their order depends on the (per process) string hash seed.
_UNORDERED_SUFFIX = "/Children"
_RUN_ATTRS = ("date_created", "pipeline_time", "CreationDT", "timestamp")


def _make_inputs(n, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.5, 12.0, n)
    age = (Planck18.age(z) - Planck18.age(20)).to("Myr").value
    sfhs = [
        SFH.LogNormal(
            tau=rng.uniform(0.1, 1.5),
            peak_age=rng.uniform(0.05, 0.95) * age[i] * Myr,
            max_age=age[i] * Myr,
        )
        for i in range(n)
    ]
    zds = [ZDist.DeltaConstant(log10metallicity=rng.uniform(-3.5, -1.5)) for i in range(n)]
    return z, sfhs, zds, rng.uniform(7, 11, n), rng.uniform(0, 2, n)


def _make_basis(grid, model, instrument, n, name="fast", seed=0, **kwargs):
    z, sfhs, zds, logm, tau_v = _make_inputs(n, seed)
    basis = GalaxyBasis(
        model_name=name,
        redshifts=z,
        grid=grid,
        emission_model=model,
        sfhs=sfhs,
        metal_dists=zds,
        log_stellar_masses=logm,
        galaxy_params={"tau_v": tau_v},
        instrument=instrument,
        build_library=False,
        redshift_dependent_sfh=True,
        **kwargs,
    )
    return basis, logm


def _dust_model(grid, **kwargs):
    return PacmanEmission(
        grid=grid,
        tau_v="tau_v",
        dust_curve=Calzetti2000(),
        dust_emission=Greybody(temperature=40 * K, emissivity=1.5),
        fesc=0.0,
        fesc_ly_alpha=0.0,
        **kwargs,
    )


def _read_all(path):
    out = {}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            out[name] = (obj[()], dict(obj.attrs), obj.dtype, obj.shape)
        else:
            out[name] = (None, dict(obj.attrs), None, None)

    with h5py.File(path, "r") as f:
        f.visititems(visit)
        root = dict(f.attrs)
    return root, out


def _assert_same_file(path_a, path_b, phot_atol_mag=None):
    root_a, a = _read_all(path_a)
    root_b, b = _read_all(path_b)
    assert set(root_a) == set(root_b)
    for key in root_a:
        if key not in _RUN_ATTRS:
            assert np.array_equal(np.asarray(root_a[key]), np.asarray(root_b[key])), key
    assert set(a) == set(b)
    for name in a:
        va, attrs_a, dt_a, sh_a = a[name]
        vb, attrs_b, dt_b, sh_b = b[name]
        assert set(attrs_a) == set(attrs_b), name
        assert dt_a == dt_b and sh_a == sh_b, name
        if va is None:
            continue
        if name.endswith(_UNORDERED_SUFFIX):
            assert sorted(va.tolist()) == sorted(vb.tolist()), name
            continue
        if phot_atol_mag is not None and va.dtype.kind == "f" and "Photometry" in name:
            pos = (va > 0) & (vb > 0)
            assert np.array_equal(va > 0, vb > 0), name
            dmag = np.abs(2.5 * np.log10(va[pos] / vb[pos]))
            assert dmag.size == 0 or dmag.max() < phot_atol_mag, (name, dmag.max())
        elif phot_atol_mag is None:
            assert np.array_equal(va, vb, equal_nan=va.dtype.kind == "f"), name


# ---------------------------------------------------------------------------
# Parallel vs serial
# ---------------------------------------------------------------------------
def test_parallel_matches_serial(test_grid, mock_instrument, tmp_path):
    """n_proc=2 (worker processes + merge) gives the same files as n_proc=1."""
    n = 60  # >= 2 * 25 galaxies so two workers are used
    paths = {}
    for n_proc in (1, 2):
        model = _dust_model(test_grid)
        basis, logm = _make_basis(test_grid, model, mock_instrument, n)
        out = tmp_path / f"np{n_proc}"
        basis.create_mock_library(
            "lib",
            log_stellar_masses=logm,
            emission_model_key="total",
            out_dir=str(out),
            n_proc=n_proc,
            overwrite=True,
        )
        paths[n_proc] = out
        # no temporary chunk files left behind
        assert sorted(p.name for p in out.iterdir()) == ["fast.hdf5", "lib.hdf5"]

    _assert_same_file(paths[1] / "fast.hdf5", paths[2] / "fast.hdf5")
    _assert_same_file(paths[1] / "lib.hdf5", paths[2] / "lib.hdf5")


def test_parallel_prebuilt_galaxies_and_batches(test_grid, mock_instrument, tmp_path):
    """Prebuilt galaxy lists and several batches also work in parallel."""
    model = _dust_model(test_grid)
    basis, _ = _make_basis(test_grid, model, mock_instrument, 60)
    galaxies = basis._create_matched_galaxies(log_base_masses=9)
    assert len(galaxies) == 60
    basis.process_galaxies(
        galaxies,
        "pre.hdf5",
        out_dir=str(tmp_path),
        n_proc=2,
        verbose=0,
        emission_model_keys="total",
        batch_size=30,
    )
    # two batches -> two files, each with 30 galaxies and the metadata attrs
    for i in (1, 2):
        with h5py.File(tmp_path / f"pre_{i}.hdf5", "r") as f:
            assert f["Galaxies/redshift"].shape == (30,)
            assert "varying_param_names" in f.attrs
            assert "Wavelengths" in f


def test_parallel_loky_and_errors(test_grid, mock_instrument, tmp_path):
    """The non-fork (loky) executor gives the same result; worker errors propagate."""
    files = {}
    for method in ("fork", "loky"):
        model = _dust_model(test_grid)
        basis, _ = _make_basis(test_grid, model, mock_instrument, 50)
        recipe = basis._create_matched_galaxies(log_base_masses=9, lazy=True)
        assert basis.galaxies == [] and len(recipe) == 50
        basis.process_galaxies(
            recipe,
            f"{method}.hdf5",
            out_dir=str(tmp_path),
            n_proc=2,
            verbose=0,
            emission_model_keys="total",
            start_method=method,
        )
        files[method] = tmp_path / f"{method}.hdf5"
    _assert_same_file(files["fork"], files["loky"])

    def broken(gal):
        raise RuntimeError("boom")

    model = _dust_model(test_grid)
    basis, _ = _make_basis(test_grid, model, mock_instrument, 50)
    recipe = basis._create_matched_galaxies(log_base_masses=9, lazy=True)
    with pytest.raises(RuntimeError, match="boom"):
        basis.process_galaxies(
            recipe,
            "err.hdf5",
            out_dir=str(tmp_path / "err"),
            n_proc=2,
            verbose=0,
            emission_model_keys="total",
            broken=broken,
        )
    assert list((tmp_path / "err").iterdir()) == []  # chunk files cleaned up


# ---------------------------------------------------------------------------
# Fast path opt-out
# ---------------------------------------------------------------------------
def test_create_galaxy_fast_sfzh_opt_out(test_grid, simple_sfh, simple_zdist):
    """fast_sfzh=False reproduces Stars(...) exactly; the fast path is close."""
    ref = Stars(
        log10ages=test_grid.log10ages,
        metallicities=test_grid.metallicities,
        sf_hist=simple_sfh,
        metal_dist=simple_zdist,
        initial_mass=1e9 * Msun,
        tau_v=0.3,
    )
    slow = create_galaxy(
        simple_sfh, 7.0, simple_zdist, test_grid, log_stellar_masses=9, fast_sfzh=False, tau_v=0.3
    )
    fast = create_galaxy(
        simple_sfh, 7.0, simple_zdist, test_grid, log_stellar_masses=9, fast_sfzh=True, tau_v=0.3
    )
    assert np.array_equal(slow.stars.sfzh, ref.sfzh)
    np.testing.assert_allclose(fast.stars.sfzh, ref.sfzh, rtol=0, atol=1e-6 * ref.sfzh.max())
    assert fast.stars.tau_v == 0.3


def test_fast_paths_vs_opt_out(test_grid, mock_instrument, tmp_path):
    """Default fast paths agree with the original paths to < 1e-3 mag."""
    paths = {}
    for label, kwargs in {
        "slow": dict(fast_sfzh=False, tabulate_igm=False),
        "fast": dict(fast_sfzh=True, tabulate_igm=True),
    }.items():
        model = _dust_model(test_grid)
        basis, logm = _make_basis(test_grid, model, mock_instrument, 8, **kwargs)
        basis.create_mock_library(
            "lib",
            log_stellar_masses=logm,
            emission_model_key="total",
            out_dir=str(tmp_path / label),
            n_proc=1,
            overwrite=True,
        )
        paths[label] = tmp_path / label
    _assert_same_file(paths["slow"] / "fast.hdf5", paths["fast"] / "fast.hdf5", phot_atol_mag=1e-3)


def test_resolve_igm(test_grid, mock_instrument):
    """tabulate_igm controls whether a (cached) TabulatedIGM is used."""
    from synthesizer.emission_models.attenuation import Inoue14

    model = _dust_model(test_grid)
    basis, _ = _make_basis(test_grid, model, mock_instrument, 4)
    assert basis._resolve_igm(Inoue14, [1.0, 2.0]) is Inoue14  # auto, few galaxies
    assert basis._resolve_igm(None, [1.0], tabulate_igm=True) is None
    tab = basis._resolve_igm(Inoue14, [1.0, 7.3], tabulate_igm=True)
    assert isinstance(tab, TabulatedIGM) and tab.z_max >= 7.3
    assert basis._resolve_igm(Inoue14, [3.0, 7.3], tabulate_igm=True) is tab  # cached
    assert basis._resolve_igm(Inoue14, [1.0], tabulate_igm=False) is Inoue14


def test_default_saved_spectra(test_grid, mock_instrument):
    """emission_model_keys=None saves only the root model unless configured."""
    model = _dust_model(test_grid)
    basis, _ = _make_basis(test_grid, model, mock_instrument, 4)
    assert basis._configure_saved_spectra(None) == ["total"]
    assert model.saved_labels == ["total"]
    # A user-configured selection is respected
    model.save_spectra("total", "attenuated")
    basis._configure_saved_spectra(None)
    assert sorted(model.saved_labels) == ["attenuated", "total"]


# ---------------------------------------------------------------------------
# Audit fixes
# ---------------------------------------------------------------------------
def test_s1_float64_hypercube_and_float32_inputs(test_grid, simple_sfh):
    """Hypercube draws are float64; float32 inputs no longer break Synthesizer."""
    draws = draw_from_hypercube({"a": (0, 1), "b": (-3, -1)}, N=10, rng=1)
    assert all(v.dtype == np.float64 for v in draws.values())

    zd = ZDist.DeltaConstant(log10metallicity=np.float32(-2.0))
    for fast in (True, False):
        gal = create_galaxy(
            simple_sfh,
            6.0,
            zd,
            test_grid,
            log_stellar_masses=np.float32(9.0),
            fast_sfzh=fast,
            tau_v=np.float32(0.4),
        )
        assert gal.stars.sfzh.dtype == np.float64
        assert isinstance(gal.stars.tau_v, float)
    # the user's ZDist is not modified
    assert isinstance(zd.metallicity_, np.float32)


def test_s2_lines_with_greybody(test_grid, mock_instrument, tmp_path):
    """Emission lines can be saved with an energy-balance dust emission model."""
    model = _dust_model(test_grid)
    assert set(_line_dependency_labels(model)) == {"dust_emission", "reprocessed", "attenuated"}
    basis, logm = _make_basis(test_grid, model, mock_instrument, 3)
    basis.create_mock_library(
        "lib",
        log_stellar_masses=logm,
        emission_model_key="total",
        out_dir=str(tmp_path),
        n_proc=1,
        overwrite=True,
        em_lines_to_save=["H 1 6562.80A"],
    )
    with h5py.File(tmp_path / "fast.hdf5", "r") as f:
        lum = f["Galaxies/Stars/Lines/Luminosity/total"]
        assert lum.shape[0] == 3 and np.all(lum[()] > 0)


def test_s5_line_luminosity_erg_s(test_parametric_galaxy, mock_emission_model):
    """calculate_line_luminosity returns erg/s."""
    lum = calculate_line_luminosity(test_parametric_galaxy, mock_emission_model, "Ha", "emergent")
    assert str(lum.units) == "erg/s"
    assert lum.value > 1e38


@pytest.mark.parametrize(
    "gen",
    [
        Greybody(temperature=45 * K, emissivity=1.7, lam_0=80 * um, do_cmb_heating=True),
        Blackbody(temperature=55 * K),
        Casey12(temperature=42 * K, emissivity=1.6, alpha=2.2),
    ],
    ids=["Greybody", "Blackbody", "Casey12"],
)
def test_s6_dust_emission_round_trip(test_grid, gen, tmp_path):
    """Dust emission parameters survive save -> HDF5 attrs -> reload."""
    model = PacmanEmission(
        grid=test_grid,
        tau_v=0.2,
        dust_curve=Calzetti2000(),
        dust_emission=gen,
        fesc=0.0,
        fesc_ly_alpha=0.0,
    )
    saved = save_emission_model(model)
    # Calzetti2000's cent_lam/gamma keep their units (needed by from_library)
    units = dict(zip(saved["dust_attenuation_keys"], saved["dust_attenuation_units"]))
    assert units["cent_lam"] != "" and units["gamma"] != ""
    assert "lam_c" not in saved["dust_emission_keys"]
    assert "do_cmb_heating" in saved["dust_emission_keys"]
    with h5py.File(tmp_path / "m.hdf5", "w") as f:
        for key in ("dust_emission_keys", "dust_emission_values", "dust_emission_units"):
            f.attrs[key] = saved[key]
        keys, values, units = (
            f.attrs["dust_emission_keys"],
            f.attrs["dust_emission_values"],
            f.attrs["dust_emission_units"],
        )
    params = load_dust_emission_params(type(gen), keys, values, units)
    new = type(gen)(**params)
    assert _dust_emission_init_params(new).keys() == _dust_emission_init_params(gen).keys()
    for key, value in _dust_emission_init_params(gen).items():
        other = getattr(new, key)
        if hasattr(value, "units"):
            assert np.isclose(other.to(value.units).value, value.value), key
        else:
            assert other == value and type(other) is type(value), key


def test_s6_unstorable_dust_emission_raises():
    """Generators with parameters that cannot be stored give a clear error."""

    class FakeGrid:
        pass

    class FakeGen:
        def __init__(self, grid, qpah=1.0):
            self.grid = grid
            self.qpah = qpah

    with pytest.raises(ValueError, match="cannot be saved"):
        _dust_emission_init_params(FakeGen(FakeGrid()))


def test_s9_invalid_emission_model_key(test_grid, mock_instrument, tmp_path):
    """A key that is not a label of the model gives a clear error up front."""
    model = TotalEmission(test_grid, Calzetti2000(), tau_v=0.1, fesc=0.0)
    assert model.label != "total"  # Synthesizer ignores label= here
    basis, logm = _make_basis(test_grid, model, mock_instrument, 3)
    with pytest.raises(ValueError, match="emission_model.label"):
        basis.create_mock_library(
            "lib",
            log_stellar_masses=logm,
            emission_model_key="total",
            out_dir=str(tmp_path),
            n_proc=1,
        )
    with pytest.raises(ValueError, match="root label"):
        GalaxySimulator(
            SFH.LogNormal,
            ZDist.DeltaConstant,
            test_grid,
            mock_instrument,
            model,
            emission_model_key="total",
        )


def test_dust_curve_param_clash(test_grid):
    """Galaxy params named like a fixed dust curve parameter are reported."""
    model = PacmanEmission(
        grid=test_grid, tau_v="tau_v", dust_curve=PowerLaw(slope=-1.0), dust_emission=None
    )
    assert _dust_curve_param_clashes(model, {"tau_v", "slope"}) == {"PowerLaw": ["slope"]}
    aliased = PacmanEmission(
        grid=test_grid, tau_v="tau_v", dust_curve=PowerLaw(slope="slope"), dust_emission=None
    )
    assert _dust_curve_param_clashes(aliased, {"slope"}) == {}


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
def test_simulator_fast_vs_original(test_grid, mock_instrument):
    """GalaxySimulator fast paths agree with the original ones (< 1e-3 mag)."""
    out = {}
    for label, kw in {
        "slow": dict(fast_sfzh=False, tabulate_igm=False),
        "fast": dict(fast_sfzh=True, tabulate_igm=True),
    }.items():
        model = _dust_model(test_grid)
        sim = GalaxySimulator(
            sfh_model=SFH.LogNormal,
            zdist_model=ZDist.DeltaConstant,
            grid=test_grid,
            instrument=mock_instrument,
            emission_model=model,
            emission_model_key="total",
            emitter_params={"stellar": ["tau_v"]},
            param_order=[
                "redshift",
                "log_mass",
                "tau",
                "peak_age",
                "max_age",
                "log10metallicity",
                "tau_v",
            ],
            param_units={"peak_age": Myr, "max_age": Myr},
            out_flux_unit="AB",
            **kw,
        )
        rng = np.random.default_rng(3)
        res = []
        for _ in range(5):
            z = rng.uniform(1, 10)
            age = (Planck18.age(z) - Planck18.age(20)).to("Myr").value
            p = [
                z,
                rng.uniform(8, 10),
                rng.uniform(0.2, 1),
                rng.uniform(0.1, 0.9) * age,
                age,
                rng.uniform(-3, -1.5),
                np.float32(rng.uniform(0, 1)),
            ]
            res.append(sim.simulate(p))
        out[label] = np.array(res)
    assert np.all(np.isfinite(out["slow"]))
    np.testing.assert_allclose(out["fast"], out["slow"], rtol=0, atol=1e-3)


def test_parametric_galaxy_unchanged_by_fast_stars(test_grid, simple_sfh, simple_zdist):
    """Spectra from fast-SFZH stars match the original Stars."""
    model = _dust_model(test_grid)
    model.save_spectra("total")
    gals = [
        create_galaxy(simple_sfh, 5.0, simple_zdist, test_grid, 9, fast_sfzh=f, tau_v=0.5)
        for f in (False, True)
    ]
    for gal in gals:
        assert isinstance(gal, Galaxy)
        gal.stars.get_spectra(model)
    np.testing.assert_allclose(
        gals[1].stars.spectra["total"].lnu, gals[0].stars.spectra["total"].lnu, rtol=1e-5
    )
