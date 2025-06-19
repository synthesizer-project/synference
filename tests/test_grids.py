"""This module contains fixtures and tests for the grid generation classes."""

import os
import subprocess
from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest
from astropy.cosmology import Planck18
from scipy.stats import uniform
from synthesizer.emission_models import IncidentEmission, TotalEmission
from synthesizer.emission_models.attenuation import Calzetti2000
from synthesizer.grid import Grid
from synthesizer.instruments import FilterCollection, Instrument
from synthesizer.parametric import SFH, Galaxy, Stars, ZDist
from unyt import Jy, Msun, Myr, nJy, unyt_array

test_dir = os.path.dirname(os.path.abspath(__file__))
grid_dir = test_dir + "/test_grids/"
os.environ["SYNTHESIZER_GRID_DIR"] = grid_dir

from sbifitter import (  # noqa E402
    CombinedBasis,
    GalaxyBasis,
    SBI_Fitter,
    calculate_muv,
    draw_from_hypercube,
    generate_sfh_basis,
)


# Fixtures for common test objects
@pytest.fixture
def test_grid():
    """Fixture to create a test Grid object."""
    if not os.path.exists(f"{test_dir}/test_grids/test_grid.hdf5"):
        subprocess.run(
            [
                "synthesizer-download",
                "--test-grids",
                "--destination",
                f"{test_dir}/test_grids/",
            ],
            check=True,
        )

    return Grid(grid_name="test_grid", grid_dir=f"{test_dir}/test_grids/")


@pytest.fixture
def mock_instrument():
    """Fixture to create a mock Instrument object with JWST filters."""
    filter_codes = [
        "JWST/NIRCam.F070W",
        "JWST/NIRCam.F090W",
        "JWST/NIRCam.F115W",
        "JWST/NIRCam.F200W",
        "JWST/NIRCam.F277W",
        "JWST/NIRCam.F356W",
        "JWST/NIRCam.F444W",
    ]
    filterset = FilterCollection(filter_codes=filter_codes)
    instrument = Instrument("JWST", filters=filterset)
    return instrument


@pytest.fixture
def mock_emission_model(test_grid):
    """Fixture to create a mock TotalEmission model for testing."""
    return TotalEmission(
        grid=test_grid,
        fesc=0.1,
        fesc_ly_alpha=0.1,
        dust_curve=Calzetti2000(),
        dust_emission_model=None,
    )


@pytest.fixture
def simple_sfh():
    """Fixture to create a simple SFH model."""
    return SFH.LogNormal(tau=0.5, peak_age=100 * Myr, max_age=300 * Myr)


@pytest.fixture
def simple_zdist():
    """Fixture to create a simple metallicity distribution."""
    return ZDist.DeltaConstant(log10metallicity=-1.0)


@pytest.fixture
def grid_basis_params(
    test_grid, mock_emission_model, mock_instrument, simple_sfh, simple_zdist
):
    """Fixture to create parameters for GalaxyBasis with a grid."""
    return {
        "model_name": "test_basis",
        "redshifts": np.array([6.0, 7.0, 8.0]),
        "grid": test_grid,
        "emission_model": mock_emission_model,
        "sfhs": [simple_sfh],
        "metal_dists": [simple_zdist],
        "cosmo": Planck18,
        "galaxy_params": {"tau_v": [0.2, 0.3, 0.4]},
        "instrument": mock_instrument,
        "redshift_dependent_sfh": False,
        "build_grid": True,
    }


@pytest.fixture
def lhc_grid(lhc_prior, n_params=1e2):
    """Fixture to create a Latin Hypercube sample grid for testing."""
    return draw_from_hypercube(N=int(n_params), param_ranges=lhc_prior, rng=42)


@pytest.fixture
def lhc_prior():
    """Fixture to create a mock prior distribution for testing."""
    return {
        "redshift": (0.01, 10),
        "masses": (5, 11),
        "tau_v": (0, 2),
        "peak_age": (0, 0.99),
        "tau": (0.1, 1.5),
        "log_zmet": (-3, -1.39),  # max of grid (e.g. 0.04)
    }


@pytest.fixture
def test_sfh():
    """Fixture to create a simple SFH model for testing."""
    return SFH.LogNormal


@pytest.fixture
def test_zmet():
    """Fixture to create a simple metallicity distribution for testing."""
    return ZDist.DeltaConstant


@pytest.fixture
def lhc_basis_params(
    test_grid,
    mock_emission_model,
    mock_instrument,
    lhc_prior,
    lhc_grid,
    test_zmet,
    test_sfh,
    n_params=1e2,
):
    """Fixture to create parameters for GalaxyBasis with LHC sampling."""
    all_param_dict = {}
    for i, key in enumerate(lhc_prior.keys()):
        all_param_dict[key] = lhc_grid[:, i]

    Z_dists = [test_zmet(log10metallicity=log_z) for log_z in all_param_dict["log_zmet"]]
    sfh_param_arrays = np.vstack((all_param_dict["tau"], all_param_dict["peak_age"])).T

    sfh_models, _ = generate_sfh_basis(
        sfh_type=test_sfh,
        sfh_param_names=["tau", "peak_age_norm"],
        sfh_param_arrays=sfh_param_arrays,
        redshifts=np.array(all_param_dict["redshift"]),
        max_redshift=20,
        cosmo=Planck18,
        sfh_param_units=[None, None],
        iterate_redshifts=False,
        calculate_min_age=False,
    )

    return {
        "model_name": "test_lhc_basis",
        "redshifts": all_param_dict["redshift"],
        "grid": test_grid,
        "emission_model": mock_emission_model,
        "metal_dists": Z_dists,
        "cosmo": Planck18,
        "instrument": mock_instrument,
        "galaxy_params": {"tau_v": all_param_dict["tau_v"]},
        # "stellar_masses": unyt_array(all_param_dict["masses"], Msun),
        "redshift_dependent_sfh": True,
        "build_grid": False,
        "sfhs": sfh_models,
    }


@pytest.fixture
def test_parametric_galaxy(test_grid, simple_sfh, simple_zdist):
    """Fixture to create a mock Galaxy object for testing."""
    return Galaxy(
        stars=Stars(
            sf_hist=simple_sfh,
            metal_dist=simple_zdist,
            metallicities=test_grid.metallicities,
            log10ages=test_grid.log10ages,
            initial_mass=1e9 * Msun,
            tau_v=0.2,
        ),
        redshift=7.0,
    )


@pytest.fixture
def test_parametric_galaxies(test_parametric_galaxy, Ngalaxies=10):
    """Fixture to create a list of mock Galaxy objects for testing."""
    return [test_parametric_galaxy for _ in range(Ngalaxies)]


def check_hdf5(hfile, expected_keys, expected_attrs=None):
    """Check that the synthesizer pipeline HDF5 file contains the expected keys."""
    with h5py.File(hfile, "r") as f:
        for key in expected_keys:
            assert key in f, f"Key '{key}' not found in HDF5 file."

        if expected_attrs is not None:
            for key, attrs in expected_attrs.items():
                assert key in f.attrs, f"Attribute '{key}' not found in HDF5 file."
                assert np.all(f.attrs[key] == attrs), (
                    f"Attribute '{key}' does not match expected value."
                )


class TestGalaxyBasis:
    """Test suite for the GalaxyBasis class."""

    def test_init_grid(self, grid_basis_params):
        """Test that GalaxyBasis initializes correctly with valid parameters."""
        basis = GalaxyBasis(**grid_basis_params)

        assert basis.model_name == "test_basis"
        assert np.array_equal(basis.redshifts, grid_basis_params["redshifts"])
        assert basis.grid == grid_basis_params["grid"]
        assert basis.emission_model == grid_basis_params["emission_model"]
        assert basis.per_particle is False

    def test_init_lhc(self, lhc_basis_params):
        """Test that GalaxyBasis initializes correctly with LHC parameters."""
        basis = GalaxyBasis(**lhc_basis_params)

        assert basis.model_name == "test_lhc_basis"
        assert np.array_equal(basis.redshifts, lhc_basis_params["redshifts"])
        assert basis.grid == lhc_basis_params["grid"]
        assert basis.emission_model == lhc_basis_params["emission_model"]
        assert basis.per_particle is False

    def test_process_priors(self):
        """Test that process_priors correctly handles prior distributions."""
        basis = GalaxyBasis(
            model_name="test_basis",
            redshifts=np.array([7.0]),
            grid=test_grid,
            emission_model=mock_emission_model,
            sfhs=[simple_sfh],
            metal_dists=[simple_zdist],
        )

        prior_dict = {
            "prior": uniform,
            "size": 100,
            "loc": 0.0,
            "scale": 1.0,
            "units": None,
        }

        result = basis.process_priors(prior_dict)
        assert len(result) == 100
        assert all(0 <= x <= 1.0 for x in result)

    def test_create_galaxy(self, grid_basis_params, simple_sfh, simple_zdist):
        """Test that create_galaxy correctly creates a Galaxy object."""
        basis = GalaxyBasis(**grid_basis_params)

        # Test with single mass
        galaxy = basis.create_galaxy(
            sfh=simple_sfh,
            redshift=7.0,
            metal_dist=simple_zdist,
            base_masses=1e9 * Msun,
            stellar_mass=1e9 * Msun,
            tau_v=0.2,
        )

        assert isinstance(galaxy, Galaxy)
        assert galaxy.redshift == 7.0, (
            f"Expected redshift to be 7.0, got {galaxy.redshift}"
        )
        assert galaxy.stars.tau_v == 0.2, (
            f"Expected tau_v to be 0.2, got {galaxy.stars.tau_v}"
        )
        assert galaxy.stars.initial_mass.value == 1e9, (
            f"Expected initial mass to be 1e9 Msun, got {galaxy.stars.initial_mass}"
        )

    def test_create_galaxies_grid(self, grid_basis_params):
        """Test that create_galaxies correctly creates multiple galaxies."""
        basis = GalaxyBasis(**grid_basis_params)

        galaxies = basis._create_galaxies(base_masses=1e8 * Msun)

        assert len(galaxies) > 0
        assert basis.create_galaxy

        # Check that varying_param_names and fixed_param_names are populated
        assert hasattr(basis, "varying_param_names")
        assert hasattr(basis, "fixed_param_names")

    def test_create_galaxies_lhc(self, lhc_basis_params):
        """Test that create_galaxies correctly creates galaxies for LHC parameters."""
        basis = GalaxyBasis(**lhc_basis_params)

        galaxies = basis._create_matched_galaxies()

        assert len(galaxies) > 0

        # Check that varying_param_names and fixed_param_names are populated
        assert hasattr(basis, "varying_param_names")
        assert hasattr(basis, "fixed_param_names")

    def test_process_galaxies_grid(self, grid_basis_params):
        """Test that process_galaxies correctly processes galaxies."""
        basis = GalaxyBasis(**grid_basis_params)

        galaxies = basis._create_galaxies(base_masses=1e9 * Msun)

        params = basis.all_parameters

        basis.process_galaxies(
            galaxies=galaxies,
            out_name="test_output_grid.hdf5",
            out_dir=f"{test_dir}/test_output/",
            n_proc=1,
            verbose=0,
            save=True,
        )

        expected_keys = list(params.keys())
        expected_keys = [f"Galaxies/{i}" for i in expected_keys]

        check_hdf5(
            hfile=f"{test_dir}/test_output/test_output_grid.hdf5",
            expected_keys=expected_keys,
        )

    def test_process_galaxies_lhc(self, lhc_basis_params):
        """Test that process_galaxies correctly processes galaxies for LHC parameters."""
        basis = GalaxyBasis(**lhc_basis_params)
        galaxies = basis._create_matched_galaxies()

        params = basis.all_parameters

        basis.process_galaxies(
            galaxies=galaxies,
            out_name="test_output_lhc.hdf5",
            out_dir=f"{test_dir}/test_output/",
            n_proc=1,
            verbose=0,
            save=True,
        )

        # test output of hfile is correct after creation e.g. dependent on above tests
        expected_keys = list(params.keys())
        expected_keys = [f"Galaxies/{i}" for i in expected_keys]

        check_hdf5(
            hfile=f"{test_dir}/test_output/test_output_grid.hdf5",
            expected_keys=expected_keys,
        )

    def test_plot_galaxy(self, grid_basis_params, test_parametric_galaxies):
        """Test that plot_galaxy correctly plots a Galaxy object."""
        basis = GalaxyBasis(**grid_basis_params)

        basis.galaxies = test_parametric_galaxies

        fig = basis.plot_galaxy(idx=0, out_dir=f"{test_dir}/test_output/")

        assert fig is not None, "Plotting function did not return a figure."
        # Check that the plot file was created
        plot_file = f"{test_dir}/test_output/test_basis_0.png"
        assert os.path.exists(plot_file), f"Plot file {plot_file} was not created."

    def test_full_single_cat_creation(self, lhc_basis_params):
        """Test that full_single_cat_creation creates a single catalog."""
        basis = GalaxyBasis(**lhc_basis_params)

        combined = basis.create_mock_cat(
            stellar_masses=unyt_array([1e9] * len(lhc_basis_params["redshifts"]), Msun),
            emission_model_key="total",
            out_name="test_combined_simple",
            out_dir=f"{test_dir}/test_output/",
            n_proc=1,
            overwrite=True,
        )

        # Check that the output file exists
        out_file = f"{test_dir}/test_output/grid_test_combined_simple.hdf5"
        assert os.path.exists(out_file), f"Output file {out_file} was not created."

        # Check that the expected keys are in the output file
        expected_keys = [
            "Grid/Parameters",
            "Grid/Photometry",
            "Grid/SupplementaryParameters",
        ]

        expected_attrs = {
            "FilterCodes": basis.instrument.filters.filter_codes,
            "ParameterNames": combined.grid_parameter_names,
            "model_name": [basis.model_name],
        }
        check_hdf5(out_file, expected_keys=expected_keys, expected_attrs=expected_attrs)


class TestCombinedBasis:
    """Test suite for the CombinedBasis class."""

    @pytest.fixture
    def combined_grid_basis_params(self, grid_basis_params):
        """Fixture to create parameters for CombinedBasis."""
        basis1 = GalaxyBasis(**grid_basis_params)
        basis2 = GalaxyBasis(**grid_basis_params)

        # Modify basis2 to have different emission model
        basis2.emission_model = TotalEmission(
            grid=grid_basis_params["grid"],
            fesc=0.2,
            fesc_ly_alpha=0.2,
            dust_curve=Calzetti2000(),
            dust_emission_model=None,
        )
        basis1.model_name = "test_grid_basis1"
        basis2.model_name = "test_grid_basis2"

        return {
            "bases": [basis1, basis2],
            "total_stellar_masses": unyt_array(
                [1e9] * len(grid_basis_params["redshifts"]), Msun
            ),
            "redshifts": grid_basis_params["redshifts"],
            "base_emission_model_keys": ["total", "total"],
            "combination_weights": np.array(
                [np.array([i, 1 - i]) for i in np.arange(0, 1.1, 0.25)]
            ),
            "out_name": "test_combined",
            "out_dir": f"{test_dir}/test_output/",
            "base_masses": 1e9 * Msun,
            "draw_parameter_combinations": True,
        }

    @pytest.fixture
    def combined_lhc_basis_params(self, lhc_basis_params):
        """Fixture to create parameters for CombinedBasis with LHC."""
        basis1 = GalaxyBasis(**lhc_basis_params)
        basis2 = GalaxyBasis(**lhc_basis_params)

        # Modify basis2 to have different emission model
        basis2.emission_model = TotalEmission(
            grid=lhc_basis_params["grid"],
            fesc=0.2,
            fesc_ly_alpha=0.2,
            dust_curve=Calzetti2000(),
            dust_emission_model=None,
        )
        basis1.model_name = "test_lhc_basis1"
        basis2.model_name = "test_lhc_basis2"

        return {
            "bases": [basis1, basis2],
            "total_stellar_masses": unyt_array(
                [1e9] * len(lhc_basis_params["redshifts"]), Msun
            ),
            "redshifts": lhc_basis_params["redshifts"],
            "base_emission_model_keys": ["total", "total"],
            "combination_weights": np.array(
                [
                    np.array([i, 1 - i])
                    for i in np.random.uniform(
                        0, 1, size=len(lhc_basis_params["redshifts"])
                    )
                ]
            ),
            "out_name": "test_combined_lhc",
            "out_dir": f"{test_dir}/test_output/",
            "base_masses": 1e9 * Msun,
            "draw_parameter_combinations": False,
        }

    def test_init_combined(self, combined_grid_basis_params):
        """Test that CombinedBasis initializes correctly with valid parameters."""
        combined = CombinedBasis(**combined_grid_basis_params)

        assert combined.out_name == "test_combined"
        assert len(combined.bases) == 2
        assert np.array_equal(combined.redshifts, combined_grid_basis_params["redshifts"])
        assert combined.base_emission_model_keys == ["total", "total"]
        assert combined.out_dir == f"{test_dir}/test_output/"

    def test_process_bases(self, combined_grid_basis_params):
        """Test that process_bases correctly processes the bases."""
        # Call process_bases to ensure it runs without errors

        combined = CombinedBasis(**combined_grid_basis_params)

        combined.process_bases(n_proc=1, overwrite=True)

        for base in combined.bases:
            out_dir = combined.out_dir
            out_name = f"{base.model_name}.hdf5"
            expected_keys = list(base.all_parameters.keys())
            expected_keys = [f"Galaxies/{i}" for i in expected_keys]
            check_hdf5(f"{out_dir}/{out_name}", expected_keys=expected_keys)

    def test_create_grid(self, combined_grid_basis_params):
        """Test that create_grid correctly creates a combined grid."""
        combined = CombinedBasis(**combined_grid_basis_params)

        # Passing in extra analysis function to pipeline to calculate mUV.
        # Any funciton could be passed in.
        combined.process_bases(overwrite=True, mUV=(calculate_muv, Planck18), n_proc=1)

        combined.create_grid()

        # Check that the output file exists
        out_file = f"{combined.out_dir}/{combined.out_name}.hdf5"
        assert os.path.exists(out_file), f"Output file {out_file} was not created."

        # Check that the expected keys are in the output file
        expected_keys = [
            "Grid/Parameters",
            "Grid/Photometry",
            "Grid/SupplementaryParameters",
        ]

        expected_attrs = {
            "FilterCodes": combined.bases[0].instrument.filters.filter_codes,
            "ParameterNames": combined.grid_parameter_names,
            "model_name": [base.model_name for base in combined.bases],
        }

        check_hdf5(
            out_file,
            expected_keys=expected_keys,
            expected_attrs=expected_attrs,
        )

    def test_create_full_grid(self, combined_lhc_basis_params):
        """Test that create_grid correctly creates a combined grid for LHC parameters."""
        combined = CombinedBasis(**combined_lhc_basis_params)

        combined.process_bases(overwrite=True, mUV=(calculate_muv, Planck18), n_proc=1)

        combined.create_grid()

        # Check that the output file exists
        out_file = f"{combined.out_dir}/{combined.out_name}.hdf5"
        assert os.path.exists(out_file), f"Output file {out_file} was not created."

        # Check that the expected keys are in the output file
        expected_keys = [
            "Grid/Parameters",
            "Grid/Photometry",
            "Grid/SupplementaryParameters",
        ]

        expected_attrs = {
            "FilterCodes": combined.bases[0].instrument.filters.filter_codes,
            "ParameterNames": combined.grid_parameter_names,
            "model_name": [base.model_name for base in combined.bases],
        }

        check_hdf5(
            out_file,
            expected_keys=expected_keys,
            expected_attrs=expected_attrs,
        )


@pytest.fixture
def test_sbi_grid():
    """Fixture to create a test SBI grid for testing SBIFitter."""
    return f"{test_dir}/test_grids/sbi_test_grid.hdf5"


class TestSBIFitter:
    """Test suite for the SBI_Fitter class."""

    def test_init_sbifitter_from_grid(self, test_sbi_grid):
        """Test that SBIFitter initializes correctly with a valid grid."""
        fitter = SBI_Fitter.init_from_hdf5(model_name="test_sbi", hdf5_path=test_sbi_grid)

        assert fitter.grid_path == test_sbi_grid, (
            "SBIFitter did not initialize with the correct grid file."
        )

    def test_sbifitter_feature_array_creation(self, test_sbi_grid):
        """Test that SBIFitter can create a basic feature array from the grid."""
        fitter = SBI_Fitter.init_from_hdf5(model_name="test_sbi", hdf5_path=test_sbi_grid)

        fitter.create_feature_array_from_raw_photometry()

        assert fitter.has_features, (
            "SBIFitter did not create a feature array from the raw photometry."
        )
        assert (
            len(fitter.simple_fitted_parameter_names) > 0
        ), """SBIFitter simple fitted parameter names are empty
             after feature array creation."""
        assert np.shape(fitter.feature_array)[0] == len(fitter.fitted_parameter_array), (
            "SBIFitter feature array shape does not match the expected shape."
        )

        # Test no normalization
        fitter.create_feature_array_from_raw_photometry(normalize_method=None)

        # Test no extra features
        fitter.create_feature_array_from_raw_photometry(extra_features=[])

        # Test with extra features
        fitter.create_feature_array_from_raw_photometry(
            extra_features=["redshift", "log_mass"]
        )
        assert (
            "redshift" in fitter.feature_names
        ), f"""Redshift not found in simple fitted parameter names after adding as
            extra feature, got {fitter.simple_fitted_parameter_names}"""
        assert (
            "log_mass" in fitter.feature_names
        ), f"""Mass not found in simple fitted parameter names after adding as
            extra feature, got {fitter.simple_fitted_parameter_names}"""

        # Test adding a color feature
        fitter.create_feature_array_from_raw_photometry(extra_features=["F200W - F070W"])

        # Test a different unit
        fitter.create_feature_array_from_raw_photometry(normed_flux_units=nJy)

        # Test including errors
        depths = np.array([29] * len(fitter.raw_photometry_names))
        depths = (10 ** (-0.4 * (depths - 8.9))) / 5 * Jy
        fitter.create_feature_array_from_raw_photometry(
            scatter_fluxes=True, depths=depths
        )

        fitter.create_feature_array_from_raw_photometry(
            scatter_fluxes=3,
            depths=depths,
            include_errors_in_feature_array=True,
        )

        # Test including missing bands randomly
        fitter.create_feature_array_from_raw_photometry(
            simulate_missing_fluxes=True, missing_flux_fraction=0.3
        )

        # Test including a specific variety of missing bands
        missing_flux_options = np.random.randint(
            0, 1, (5, len(fitter.raw_photometry_names))
        ).astype(bool)
        fitter.create_feature_array_from_raw_photometry(
            simulate_missing_fluxes=True,
            missing_flux_options=missing_flux_options,
        )

        # Test removing a parameter
        fitter.create_feature_array_from_raw_photometry(parameters_to_remove=["mass"])

        # Test removing a feature
        fitter.create_feature_array_from_raw_photometry(
            photometry_to_remove=[fitter.raw_photometry_names[0]]
        )

        # Test removing a feature that does not exist
        with pytest.raises(
            AssertionError,
            match="photometry_to_remove must be a list of filter names to remove.",
        ):
            fitter.create_feature_array_from_raw_photometry(photometry_to_remove=123)

        # Test a different limit
        fitter.create_feature_array_from_raw_photometry(norm_mag_limit=100)


# Integration tests for full pipeline
def test_full_pipeline_integration(tmp_path):
    """Integration test for the full pipeline.

    This test creates a minimal version of the full pipeline to verify
    that all components work together correctly.
    """
    # Skip this test if we're not running integration tests
    pytest.skip("Integration test - skipping for normal test runs")

    # Create temporary directory for output
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    # Create minimal grid
    test_grid = MagicMock(spec=Grid)
    test_grid.log10ages = np.linspace(6, 10, 5)
    test_grid.metallicity = np.array([0.0001, 0.02])
    test_grid.wavelength = np.logspace(3, 5, 20)

    # Create minimal instrument
    filterset = MagicMock(spec=FilterCollection)
    filterset.filter_codes = ["JWST/NIRCam.F070W", "JWST/NIRCam.F200W"]
    instrument = MagicMock(spec=Instrument)
    instrument.filters = filterset
    instrument.label = "JWST"

    # Create minimal SFH and metallicity distribution
    sfh = SFH.LogNormal(tau=0.5, peak_age=100 * Myr, max_age=300 * Myr)
    zdist = ZDist.DeltaConstant(log10metallicity=-1.0)

    # Create minimal emission models
    emission_model1 = MagicMock(spec=TotalEmission)
    emission_model1.set_per_particle = MagicMock()

    emission_model2 = MagicMock(spec=IncidentEmission)
    emission_model2.set_per_particle = MagicMock()

    # Create bases
    basis1 = GalaxyBasis(
        model_name="PopII",
        redshifts=np.array([6.0]),
        grid=test_grid,
        emission_model=emission_model1,
        sfhs=[sfh],
        metal_dists=[zdist],
        cosmo=Planck18,
        instrument=instrument,
        galaxy_params={"tau_v": 0.5},
        redshift_dependent_sfh=False,
    )

    basis2 = GalaxyBasis(
        model_name="PopIII",
        redshifts=np.array([6.0]),
        grid=test_grid,
        emission_model=emission_model2,
        sfhs=[sfh],
        metal_dists=[zdist],
        cosmo=Planck18,
        instrument=instrument,
        galaxy_params={},
        redshift_dependent_sfh=False,
    )

    # Mock create_galaxies and process_galaxies
    basis1.create_galaxies = MagicMock(return_value=[MagicMock(spec=Galaxy)])
    basis1.process_galaxies = MagicMock()
    basis2.create_galaxies = MagicMock(return_value=[MagicMock(spec=Galaxy)])
    basis2.process_galaxies = MagicMock()

    # Create combined basis
    combined = CombinedBasis(
        bases=[basis1, basis2],
        total_stellar_masses=unyt_array([1e9], Msun),
        redshifts=np.array([6.0]),
        base_emission_model_keys=["total", "incident"],
        combination_weights=np.array([[0.7, 0.3]]),
        out_name="test_combined",
        out_dir=str(out_dir),
        base_masses=1e9 * Msun,
    )

    # Mock load_bases
    mock_outputs = {
        "PopII": {
            "properties": {
                "redshift": np.array([6.0]),
                "mass": unyt_array([1e9], "Msun"),
                "PopII/tau": np.array([0.5]),
            },
            "observed_spectra": np.ones((20, 1)),
            "wavelengths": np.logspace(3, 5, 20),
            "observed_photometry": {
                "F070W": np.array([1.0]),
                "F200W": np.array([2.0]),
            },
        },
        "PopIII": {
            "properties": {
                "redshift": np.array([6.0]),
                "mass": unyt_array([1e9], "Msun"),
                "PopIII/metallicity": np.array([-1.0]),
            },
            "observed_spectra": np.ones((20, 1)),
            "wavelengths": np.logspace(3, 5, 20),
            "observed_photometry": {
                "F070W": np.array([3.0]),
                "F200W": np.array([4.0]),
            },
        },
    }
    combined.load_bases = MagicMock(return_value=mock_outputs)

    # Mock save_grid
    combined.save_grid = MagicMock()

    # Process bases and create grid
    combined.process_bases()
    combined.create_grid()

    # Check that all methods were called
    basis1.create_galaxies.assert_called_once()
    basis1.process_galaxies.assert_called_once()
    basis2.create_galaxies.assert_called_once()
    basis2.process_galaxies.assert_called_once()
    combined.load_bases.assert_called_once()
    combined.save_grid.assert_called_once()


# Run the tests

if __name__ == "__main__":
    pytest.main([__file__])
    # To run the tests, use the command:
    # pytest -v ltu-ili_testing/tests/basis_tests.py
