"""Test suite for the SBI_Fitter class."""

import os

import numpy as np
import pytest
from astropy.cosmology import Planck18
from synthesizer.instruments import FilterCollection
from unyt import Angstrom, Jy, Myr, nJy, unyt_array
from unyt.dimensions import mass, time

from synference import GalaxyBasis, SBI_Fitter



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

