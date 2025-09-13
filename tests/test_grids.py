"""This module contains fixtures and tests for the grid generation classes."""

import os

import h5py
import numpy as np
import pytest
from astropy.cosmology import Planck18
from scipy.stats import uniform
from synthesizer.emission_models import TotalEmission
from synthesizer.emission_models.attenuation import Calzetti2000
from synthesizer.instruments import FilterCollection
from synthesizer.parametric import Galaxy
from unyt import Angstrom, Jy, Myr, nJy, unyt_array
from unyt.dimensions import mass, time

from synference import (  # noqa E402
    CombinedBasis,
    GalaxyBasis,
    SBI_Fitter,
    calculate_muv,
    draw_from_hypercube,
    generate_sfh_basis,
)


def check_hdf5(hfile, expected_keys, expected_attrs=None, check_size=False):
    """Check that the synthesizer pipeline HDF5 file contains the expected keys."""
    with h5py.File(hfile, "r") as f:
        for key in expected_keys:
            assert key in f, f"Key '{key}' not found in HDF5 file."
            assert isinstance(f[key], h5py.Dataset), f"Key '{key}' is not a dataset in HDF5 file."
            if check_size:
                # Check all numpy arrays are 2D
                assert f[key].ndim == 2, f"Key '{key}' is not a 2D array in HDF5 file."

        if expected_attrs is not None:
            for key, attrs in expected_attrs.items():
                assert key in f.attrs, f"Attribute '{key}' not found in HDF5 file."
                assert np.all(f.attrs[key] == attrs), (
                    f"Attribute '{key}' does not match expected value."
                )


@pytest.fixture
def combined_grid_basis_params(grid_basis_params, test_dir):
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
        "log_stellar_masses": [9] * len(grid_basis_params["redshifts"]),
        "redshifts": grid_basis_params["redshifts"],
        "base_emission_model_keys": ["emergent", "emergent"],
        "combination_weights": np.array([np.array([i, 1 - i]) for i in np.arange(0, 1.1, 0.25)]),
        "out_name": "test_combined",
        "out_dir": f"{test_dir}/test_output/",
        "log_base_masses": 9,
        "draw_parameter_combinations": True,
    }


@pytest.fixture
def combined_lhc_basis_params(lhc_basis_params, test_dir):
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
        "log_stellar_masses": [9] * len(lhc_basis_params["redshifts"]),
        "redshifts": lhc_basis_params["redshifts"],
        "base_emission_model_keys": ["emergent", "emergent"],
        "combination_weights": np.array(
            [
                np.array([i, 1 - i])
                for i in np.random.uniform(0, 1, size=len(lhc_basis_params["redshifts"]))
            ]
        ),
        "out_name": "test_combined_lhc",
        "out_dir": f"{test_dir}/test_output/",
        "log_base_masses": 9,
        "draw_parameter_combinations": False,
    }


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

    def test_process_priors(self, test_grid, mock_emission_model):
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
            log_stellar_masses=9,
            tau_v=0.2,
        )

        assert isinstance(galaxy, Galaxy)
        assert galaxy.redshift == 7.0, f"Expected redshift to be 7.0, got {galaxy.redshift}"
        assert galaxy.stars.tau_v == 0.2, f"Expected tau_v to be 0.2, got {galaxy.stars.tau_v}"
        assert galaxy.stars.initial_mass.value == 1e9, (
            f"Expected initial mass to be 1e9 Msun, got {galaxy.stars.initial_mass}"
        )

    def test_create_galaxies_grid(self, grid_basis_params):
        """Test that create_galaxies correctly creates multiple galaxies."""
        basis = GalaxyBasis(**grid_basis_params)

        galaxies = basis._create_galaxies(log_base_masses=9)

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

    def test_process_galaxies_grid(self, grid_basis_params, test_dir):
        """Test that process_galaxies correctly processes galaxies."""
        basis = GalaxyBasis(**grid_basis_params)

        galaxies = basis._create_galaxies(log_base_masses=9)

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

    def test_process_galaxies_lhc(self, lhc_basis_params, test_dir):
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

    def test_plot_galaxy(self, grid_basis_params, test_parametric_galaxies, test_dir):
        """Test that plot_galaxy correctly plots a Galaxy object."""
        basis = GalaxyBasis(**grid_basis_params)

        basis.galaxies = test_parametric_galaxies

        fig = basis.plot_galaxy(
            idx=0, out_dir=f"{test_dir}/test_output/", emission_model_keys=["emergent"]
        )

        assert fig is not None, "Plotting function did not return a figure."
        # Check that the plot file was created
        plot_file = f"{test_dir}/test_output/test_basis_0.png"
        assert os.path.exists(plot_file), f"Plot file {plot_file} was not created."

    def test_full_single_cat_creation(self, lhc_basis_params, test_dir):
        """Test that full_single_cat_creation creates a single catalog."""
        basis = GalaxyBasis(**lhc_basis_params)

        combined = basis.create_mock_cat(
            log_stellar_masses=[9] * len(lhc_basis_params["redshifts"]),
            emission_model_key="emergent",
            out_name="test_combined_simple",
            out_dir=f"{test_dir}/test_output/",
            n_proc=1,
            overwrite=True,
        )

        # Check that the output file exists
        out_file = f"{test_dir}/test_output/test_combined_simple.hdf5"
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

    def test_init_combined(self, combined_grid_basis_params, test_dir):
        """Test that CombinedBasis initializes correctly with valid parameters."""
        combined = CombinedBasis(**combined_grid_basis_params)

        assert combined.out_name == "test_combined"
        assert len(combined.bases) == 2
        assert np.array_equal(combined.redshifts, combined_grid_basis_params["redshifts"])
        assert combined.base_emission_model_keys == ["emergent", "emergent"]
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
            check_size=True,
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
            check_size=True,
        )


class TestSBIFitter:
    """Test suite for the SBI_Fitter class."""

    def test_init_sbifitter_from_grid(self, test_sbi_grid):
        """Test that synference initializes correctly with a valid grid."""
        fitter = SBI_Fitter.init_from_hdf5(model_name="test_sbi", hdf5_path=test_sbi_grid)

        assert fitter.grid_path == test_sbi_grid, (
            "synference did not initialize with the correct grid file."
        )

    def test_sbifitter_feature_array_creation(self, test_sbi_grid):
        """Test that synference can create a basic feature array from the grid."""
        fitter = SBI_Fitter.init_from_hdf5(model_name="test_sbi", hdf5_path=test_sbi_grid)

        fitter.create_feature_array_from_raw_photometry()

        assert fitter.has_features, (
            "synference did not create a feature array from the raw photometry."
        )
        assert (
            len(fitter.simple_fitted_parameter_names) > 0
        ), """synference simple fitted parameter names are empty
             after feature array creation."""
        assert np.shape(fitter.feature_array)[0] == len(fitter.fitted_parameter_array), (
            "synference feature array shape does not match the expected shape."
        )

        # Test no normalization
        fitter.create_feature_array_from_raw_photometry(normalize_method=None)

        # Test no extra features
        fitter.create_feature_array_from_raw_photometry(extra_features=[])

        # Test with extra features
        fitter.create_feature_array_from_raw_photometry(extra_features=["redshift", "log_mass"])
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
        depths = np.array([29] * len(fitter.raw_observation_names))
        depths = (10 ** (-0.4 * (depths - 8.9))) / 5 * Jy
        fitter.create_feature_array_from_raw_photometry(scatter_fluxes=True, depths=depths)

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
            0, 1, (5, len(fitter.raw_observation_names))
        ).astype(bool)
        fitter.create_feature_array_from_raw_photometry(
            simulate_missing_fluxes=True,
            missing_flux_options=missing_flux_options,
        )

        # Test removing a parameter
        fitter.create_feature_array_from_raw_photometry(parameters_to_remove=["mass"])

        # Test removing a feature
        fitter.create_feature_array_from_raw_photometry(
            photometry_to_remove=[fitter.raw_observation_names[0]]
        )

        # Test removing a feature that does not exist
        with pytest.raises(
            AssertionError,
            match="photometry_to_remove must be a list of filter names to remove.",
        ):
            fitter.create_feature_array_from_raw_photometry(photometry_to_remove=123)

        # Test a different limit
        fitter.create_feature_array_from_raw_photometry(norm_mag_limit=100)

        # Test asinh softening
        fitter.create_feature_array_from_raw_photometry(
            normed_flux_units="asinh", asinh_softening_parameters=5 * nJy
        )

        # Test depth based asinh softening
        with pytest.raises(
            AssertionError,
        ):
            fitter.create_feature_array_from_raw_photometry(
                normed_flux_units="asinh",
                asinh_softening_parameters="SNR_5",
                depths=depths,
            )

        fitter.create_feature_array_from_raw_photometry(
            normed_flux_units="asinh",
            asinh_softening_parameters="SNR_5",
            depths=depths,
            scatter_fluxes=True,
        )


class TestFullPipeline:
    """Test suite for full runthrough of grids and synference."""

    def test_full_lhc(self, lhc_basis_params, test_dir):
        """Test the full runthrough of LHC grid creation and synference."""
        # Create the GalaxyBasis with LHC parameters
        basis = GalaxyBasis(**lhc_basis_params)

        stellar_masses = np.random.uniform(7, 11, size=len(lhc_basis_params["redshifts"]))

        basis.create_mock_cat(
            log_stellar_masses=stellar_masses,
            emission_model_key="emergent",
            out_name="test_full_simple",
            out_dir=f"{test_dir}/test_output/",
            n_proc=1,
            overwrite=True,
        )

        # Initialize synference from the created grid
        fitter = SBI_Fitter.init_from_hdf5(
            model_name="test_sbi_lhc",
            hdf5_path=f"{test_dir}/test_output/test_full_simple.hdf5",
        )

        # Create feature array
        fitter.create_feature_array()

        assert fitter.has_features, (
            "synference did not create a feature array from the raw photometry."
        )

        fitter.run_single_sbi()


class TestSuppFunctions:
    """Test suite for the utility functions in the synthesizer package."""

    def param_functions(self, function):
        """Get a parameter function from the SUPP_FUNCTIONS module."""
        from synference import SUPP_FUNCTIONS

        return getattr(SUPP_FUNCTIONS, function)

    @pytest.fixture
    def test_galaxy(self, test_parametric_galaxy, mock_emission_model):
        """Fixture to create a mock Galaxy object for testing."""
        test_parametric_galaxy.stars.get_spectra(emission_model=mock_emission_model)
        test_parametric_galaxy.get_observed_spectra(cosmo=Planck18)
        filter_codes = ["JWST/NIRCam.F444W", "JWST/NIRCam.F115W"]
        filters = FilterCollection(filter_codes=filter_codes)
        test_parametric_galaxy.get_photo_fnu(filters)
        return test_parametric_galaxy

    def test_calculate_muv(self, test_galaxy):
        """Test the calculate_muv function."""
        func = self.param_functions("calculate_muv")
        muv = func(test_galaxy, Planck18)
        muv = muv["emergent"]
        assert muv is not None, "calculate_muv did not return a value."
        assert isinstance(muv, unyt_array), (
            f"calculate_muv did not return a unyt_array. Got {type(muv)} instead."
        )
        assert muv.units.dimensions == Jy.dimensions, (
            "calculate_muv did not return a value with the correct units."
        )

    def test_calculate_beta(self, test_galaxy):
        """Test the calculate_beta function."""
        func = self.param_functions("calculate_beta")
        beta = func(test_galaxy, emission_model_key="emergent")

        assert beta is not None, "calculate_beta did not return a value."
        assert isinstance(beta, float), "calculate_beta did not return a float."

    def test_calculate_sfr(self, test_galaxy):
        """Test the calculate_sfr function."""
        pytest.skip("Skipping calculate_sfr test as it requires a function not in Synthesizer.")
        func = self.param_functions("calculate_sfr")
        sfr = func(test_galaxy)

        assert sfr is not None, "calculate_sfr did not return a value."
        assert isinstance(sfr, unyt_array), "calculate_sfr did not return a unyt_array."
        assert sfr.units.dimensions == mass / time, (
            "calculate_sfr did not return a value with the correct dimensions."
        )

    def test_calculate_balmer_decrement(self, test_galaxy):
        """Test the calculate_balmer_break function."""
        func = self.param_functions("calculate_balmer_decrement")
        balmer_break = func(test_galaxy, emission_model_key="emergent")

        assert balmer_break is not None, "calculate_balmer_break did not return a value."
        assert isinstance(balmer_break, float), "calculate_balmer_break did not return a float."

    def test_calculate_colour(self, test_galaxy):
        """Test the calculate_colours function."""
        func = self.param_functions("calculate_colour")
        colour = func(test_galaxy, "V", "J", emission_model_key="emergent", rest_frame=True)

        assert colour is not None, "calculate_colour did not return a value."

        assert np.isfinite(colour).all(), "calculate_colour returned NaN or infinite values."

    def test_calculate_line_ew(self, test_galaxy, mock_emission_model):
        """Test the calculate_line_ew function."""
        func = self.param_functions("calculate_line_ew")
        ew = func(test_galaxy, mock_emission_model, "Ha", emission_model_key="emergent")

        assert ew is not None, "calculate_line_ew did not return a value."
        assert isinstance(ew, unyt_array), "calculate_line_ew did not return a unyt_array."
        assert ew.units == Angstrom, (
            "calculate_line_ew did not return a value with the correct units."
        )
        assert np.isfinite(ew), "calculate_line_ew returned NaN or infinite values."
        assert ew > 0 * Angstrom, "calculate_line_ew returned a non-positive equivalent width."

    def test_calculate_line_flux(self, test_galaxy, mock_emission_model):
        """Test the calculate_line_flux function."""
        func = self.param_functions("calculate_line_flux")
        flux = func(test_galaxy, mock_emission_model, "Ha")

        assert flux is not None, "calculate_line_flux did not return a value."
        assert isinstance(flux, unyt_array), "calculate_line_flux did not return a unyt_array."
        assert flux.units.dimensions == mass / time**3, (
            "calculate_line_flux did not return a value with the correct units. "
            f"Expected Jy, got {flux.units.dimensions}"
        )

    def test_calculate_d4000(self, test_galaxy):
        """Test the calculate_d4000 function."""
        pytest.skip("Skipping calculate_d4000 test as it requires a function not in Synthesizer.")
        func = self.param_functions("calculate_d4000")
        d4000 = func(test_galaxy)

        assert d4000 is not None, "calculate_d4000 did not return a value."
        assert isinstance(d4000, float), "calculate_d4000 did not return a float."
        assert 1 <= d4000 <= 2.5, "calculate_d4000 returned a value outside the expected range."

    def test_calculate_mass_weighted_age(self, test_galaxy):
        """Test the calculate_mass_weighted_age function."""
        pytest.skip(
            "Skipping calculate_mass_weighted_age test as it requires afunction not in Synthesizer."
        )
        func = self.param_functions("calculate_mass_weighted_age")
        age = func(test_galaxy)

        assert age is not None, "calculate_mass_weighted_age did not return a value."
        assert isinstance(age, unyt_array), (
            "calculate_mass_weighted_age did not return a unyt_array."
        )
        assert age.units == Myr, (
            "calculate_mass_weighted_age did not return a value with the correct units."
        )

