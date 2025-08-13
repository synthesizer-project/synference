"""Tests for the uncertainty model classes in sbifitter."""

import os
import pickle
import shutil

import numpy as np
import pytest
from unyt import Jy, uJy

from sbifitter import (
    AsinhEmpiricalUncertaintyModel,
    DepthUncertaintyModel,
    GeneralEmpiricalUncertaintyModel,
    UncertaintyModel,
    load_unc_model_from_hdf5,
    save_unc_model_to_hdf5,
)

# =============================================================================
# Test Functions
# =============================================================================


@pytest.fixture(scope="module")
def temp_dir():
    """Create a temporary directory for test artifacts for the entire module."""
    dir_path = "temp_pytest_artifacts"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    yield dir_path
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)


@pytest.fixture(scope="module")
def mock_data():
    """Generate realistic mock photometric data for use in tests."""
    n_points = 1000
    true_flux_ab = np.linspace(20, 28, n_points)
    true_error_ab = 0.05 + np.exp((true_flux_ab - 26) / 1.5)
    fluxes = true_flux_ab + np.random.normal(0, 0.01, n_points)
    errors = true_error_ab + np.random.normal(0, 0.02, n_points)
    return fluxes, errors


def test_static_unit_conversions():
    """Test the static unit conversion methods in the base class."""
    ab_mag = 23.9
    expected_jy_val = 1e-6
    assert UncertaintyModel.ab_to_jy(ab_mag).to_value(Jy) == pytest.approx(expected_jy_val)

    flux_ujy = 1 * uJy
    expected_ab = 23.9
    assert UncertaintyModel.jy_to_ab(flux_ujy) == pytest.approx(expected_ab, abs=1e-2)

    flux_jy_val = 1e-6 * Jy
    mag_err = 0.1
    flux_err_jy = UncertaintyModel.ab_err_to_jy(mag_err, flux_jy_val)
    reverted_mag_err = UncertaintyModel.jy_err_to_ab(flux_err_jy, flux_jy_val)
    assert reverted_mag_err == pytest.approx(mag_err)


# --- Tests for DepthUncertaintyModel ---


def test_depth_model_initialization():
    """Test initialization of DepthUncertaintyModel."""
    depth_mag = 25.0
    model = DepthUncertaintyModel(depth_ab=depth_mag, depth_sigma_level=5)

    # The expected sigma is the flux at the 5-sigma depth, divided by 5.
    expected_sigma_jy = UncertaintyModel.ab_to_jy(depth_mag) / 5.0
    assert model.sigma.to_value(Jy) == pytest.approx(expected_sigma_jy.to_value(Jy))


def test_depth_model_apply_noise():
    """Check that the depth model applies noise with correct statistics."""
    model = DepthUncertaintyModel(depth_ab=25.0, depth_sigma_level=5)

    n_samples = 10000
    input_flux = np.full(n_samples, 1.0) * uJy  # 1 uJy

    noisy_flux = model.apply_noise(input_flux)
    noise = (noisy_flux - input_flux).to_value(Jy)

    assert np.mean(noise) == pytest.approx(0.0, abs=5e-9)
    assert np.std(noise) == pytest.approx(model.sigma.to_value(Jy), rel=5e-2)


def test_depth_model_apply_scalings():
    """Test that the depth model's scalings correctly convert units."""
    model = DepthUncertaintyModel(depth_ab=25.0)

    flux_ab, error_ab = 23.9, 0.1
    flux_jy, error_jy = model.apply_scalings(flux_ab, error_ab, flux_units="AB", out_units="Jy")

    expected_flux_jy = UncertaintyModel.ab_to_jy(flux_ab).to_value(Jy)
    assert flux_jy == pytest.approx(expected_flux_jy)


def test_depth_model_serialization(temp_dir):
    """Test HDF5 serialization for the DepthUncertaintyModel."""
    model_orig = DepthUncertaintyModel(depth_ab=26.5, depth_sigma_level=10)

    hdf5_path = os.path.join(temp_dir, "depth_model.hdf5")
    group_name = "depth_test"

    save_unc_model_to_hdf5(model_orig, hdf5_path, group_name, overwrite=True)
    model_loaded = load_unc_model_from_hdf5(hdf5_path, group_name)

    assert isinstance(model_loaded, DepthUncertaintyModel)
    assert model_orig.depth_ab == model_loaded.depth_ab
    assert model_orig.sigma.to_value(Jy) == pytest.approx(model_loaded.sigma.to_value(Jy))


# --- Tests for AsinhEmpiricalUncertaintyModel ---


def test_asinh_model_initialization(mock_data):
    """Test initialization of AsinhEmpiricalUncertaintyModel."""
    fluxes_ab, errors_ab = mock_data
    fluxes_jy = UncertaintyModel.ab_to_jy(fluxes_ab)
    errors_jy = UncertaintyModel.ab_err_to_jy(errors_ab, fluxes_jy)

    model = AsinhEmpiricalUncertaintyModel(
        observed_phot_jy=fluxes_jy, observed_phot_errors_jy=errors_jy
    )

    assert model.b is not None
    assert model._mu_sigma_interpolator is not None


def test_asinh_model_low_snr_and_negative_flux(mock_data, temp_dir):
    """Tests that the AsinhEmpiricalUncertaintyModel correctly handles low SNR data."""
    # 1. Create challenging mock data: fluxes centered around zero
    #    with significant errors, ensuring low SNR and negative values.
    n_points = 2000
    # Fluxes are normally distributed around zero
    flux_values_jy = np.random.normal(loc=1.0, scale=0.5, size=n_points) * uJy
    # Errors are constant and relatively large
    error_values_jy = np.full(n_points, 0.1) * uJy

    # 2. Initialize the model with this data.
    #    This should work without errors.
    model = AsinhEmpiricalUncertaintyModel(
        observed_phot_jy=flux_values_jy,
        observed_phot_errors_jy=error_values_jy,
        interpolation_flux_unit="uJy",
        return_noise=True,
    )
    assert model is not None, "Model failed to initialize with low SNR data."
    assert callable(model._mu_sigma_interpolator), "Interpolator was not created."

    # 3. Create specific test cases: a positive low-SNR flux, a zero flux,
    #    and a negative flux.
    test_fluxes_jy = np.array([0.5, 0.0, -0.5]) * uJy
    test_errors_jy = np.array([1.0, 1.0, 1.0]) * uJy

    # 4. Apply the model's scaling (unit conversion to asinh space).
    #    This should be robust and not produce NaNs.
    mag_asinh, mag_err_asinh = model.apply_scalings(test_fluxes_jy, test_errors_jy)

    # 5. Check the outputs for correctness.
    assert not np.any(np.isnan(mag_asinh)), "Output contains NaNs, asinh scaling failed."
    assert not np.any(np.isnan(mag_err_asinh)), "Output errors contain NaNs."

    test_fluxes_jy = np.array([0.5, 0.25, 0.1, 0.05, 0.0, -0.1, -0.5]) * uJy
    noisy_fluxes, errors = model.apply_noise(test_fluxes_jy)

    assert not np.any(np.isnan(noisy_fluxes)), "Output contains NaNs, asinh scaling failed."
    assert not np.any(np.isnan(errors)), "Output errors contains NaNs, asinh scaling failed."


def test_asinh_model_apply_scalings(mock_data):
    """Test that asinh model scalings correctly convert to asinh space."""
    fluxes_ab, errors_ab = mock_data
    fluxes_jy = UncertaintyModel.ab_to_jy(fluxes_ab)
    errors_jy = UncertaintyModel.ab_err_to_jy(errors_ab, fluxes_jy)

    model = AsinhEmpiricalUncertaintyModel(
        observed_phot_jy=fluxes_jy, observed_phot_errors_jy=errors_jy
    )

    # Test with a subset of the data
    test_flux_jy = fluxes_jy[:10]
    test_error_jy = errors_jy[:10]

    mag_asinh, mag_err_asinh = model.apply_scalings(test_flux_jy, test_error_jy)

    assert mag_asinh.shape == test_flux_jy.shape
    assert not np.any(np.isnan(mag_asinh))


# --- Tests for GeneralEmpiricalUncertaintyModel ---


def test_general_model_initialization_ab(mock_data):
    """Test initialization of GeneralEmpiricalUncertaintyModel with AB mags."""
    fluxes, errors = mock_data
    model = GeneralEmpiricalUncertaintyModel(
        observed_fluxes=fluxes, observed_errors=errors, flux_unit="AB", num_bins=15
    )
    assert model._mu_sigma_interpolator is not None
    assert model.flux_unit == "AB"
    assert model.interpolation_flux_unit == "AB"


def test_general_model_initialization_jy(mock_data):
    """Test initialization with physical flux units (uJy)."""
    fluxes_ab, errors_ab = mock_data
    fluxes_jy = UncertaintyModel.ab_to_jy(fluxes_ab)
    errors_jy = UncertaintyModel.ab_err_to_jy(errors_ab, fluxes_jy)

    model = GeneralEmpiricalUncertaintyModel(
        observed_fluxes=fluxes_jy.to_value("uJy"),
        observed_errors=errors_jy.to_value("uJy"),
        flux_unit="uJy",
        log_bins=True,  # More appropriate for flux space
    )
    assert model._mu_sigma_interpolator is not None
    assert model.flux_unit == "uJy"


def test_interpolation_unit_conversion(mock_data):
    """Test that using a different interpolation unit works as expected."""
    fluxes, errors = mock_data
    model = GeneralEmpiricalUncertaintyModel(
        observed_fluxes=fluxes,
        observed_errors=errors,
        flux_unit="AB",
        interpolation_flux_unit="uJy",
    )
    assert model.interpolation_flux_unit == "uJy"
    # The internal bin centers should now be in uJy, not AB magnitudes
    # AB mags would be ~24, while uJy values would be much smaller.
    assert np.mean(model.bin_centers) < 100


def test_already_binned_path(mock_data):
    """Test that the 'already_binned' flag correctly bypasses binning."""
    fluxes, errors = mock_data
    model1 = GeneralEmpiricalUncertaintyModel(observed_fluxes=fluxes, observed_errors=errors)

    # Create a new model using the binned data from the first
    model2 = GeneralEmpiricalUncertaintyModel(
        observed_fluxes=model1.bin_centers,
        observed_errors=None,  # Not needed
        already_binned=True,
        bin_median_errors=model1.median_error_in_bin,
        bin_std_errors=model1.std_error_in_bin,
    )

    # Check that the models are functionally identical
    test_flux = np.array([24, 25, 26])
    mu1 = model1.sample_uncertainty(test_flux)
    mu2 = model2.sample_uncertainty(test_flux)
    np.testing.assert_allclose(mu1, mu2, atol=0.3)


def test_apply_noise_statistical_properties(mock_data):
    """Check that the applied noise has the correct statistical properties."""
    fluxes, errors = mock_data
    model = GeneralEmpiricalUncertaintyModel(
        observed_fluxes=fluxes, observed_errors=errors, return_noise=True
    )

    test_flux, n_samples = 25.0, 10000
    input_fluxes = np.full(n_samples, test_flux)

    noisy_fluxes, _ = model.apply_noise(input_fluxes)
    noise = noisy_fluxes - test_flux

    # The expected standard deviation is the error the model samples for that flux
    expected_std = model.sample_uncertainty(np.array([test_flux]))[0]

    assert np.mean(noise) == pytest.approx(0.0, abs=0.1)
    assert np.std(noise) == pytest.approx(expected_std, rel=0.1)


def test_upper_limit_logic(mock_data):
    """Verify that the upper limit handling works as expected."""
    fluxes, errors = mock_data
    model = GeneralEmpiricalUncertaintyModel(
        observed_fluxes=fluxes,
        observed_errors=errors,
        flux_unit="AB",
        upper_limits=True,
        treat_as_upper_limits_below=1.0,  # Treat SNR < 1 as upper limits
        return_noise=True,
    )

    assert model.interpolation_flux_unit == "AB", f"{model.interpolation_flux_unit}"

    # A very faint flux that should always become an upper limit
    faint_flux = np.array([30.0])
    noisy_flux, _ = model.apply_noise(faint_flux)

    assert model.upper_limit_value is not None
    # The output flux should be close to the pre-calculated upper limit value
    assert noisy_flux[0] == pytest.approx(model.upper_limit_value, abs=1.0)  # Allow for scatter


def test_general_model_apply_scalings(mock_data):
    """Tests that the GeneralEmpiricalUncertaintyModel's apply_scalings method is correct."""
    # 1. Initialize a model with a known upper limit rule.
    fluxes, errors = mock_data
    model = GeneralEmpiricalUncertaintyModel(
        observed_fluxes=fluxes,
        observed_errors=errors,
        flux_unit="AB",
        upper_limits=True,
        treat_as_upper_limits_below=2.0,  # Treat SNR < 2.0 as an upper limit
    )

    # 2. Define test data: one high-SNR point and one low-SNR point.
    # High SNR: mag=22, err=0.1 -> SNR > 10
    # Low SNR: mag=27, err=1.5 -> SNR < 1
    test_fluxes_ab = np.array([22.0, 27.0])
    test_errors_ab = np.array([0.1, 1.5])

    # 3. Call apply_scalings to transform the data.
    scaled_fluxes, scaled_errors = model.apply_scalings(
        flux=test_fluxes_ab,
        error=test_errors_ab,
        flux_units="AB",
        out_units="AB",  # Keep units the same for a clear comparison
    )

    # 4. Assert the high-SNR point is unchanged.
    # Since input and output units are the same, the values should be identical.
    assert scaled_fluxes[0] == pytest.approx(test_fluxes_ab[0])
    assert scaled_errors[0] == pytest.approx(test_errors_ab[0])

    # 5. Assert the low-SNR point was replaced by the model's upper limit values.
    # The flux should be the model's pre-calculated upper limit value.
    assert scaled_fluxes[1] == pytest.approx(model.upper_limit_value)
    # The error should be the model's median error at that upper limit flux.
    expected_error_at_limit = model._mu_sigma_interpolator(model.upper_limit_value)
    assert scaled_errors[1] == pytest.approx(expected_error_at_limit)


def test_pickle_serialization(mock_data, temp_dir):
    """Test that the model can be pickled and unpickled correctly."""
    fluxes, errors = mock_data
    model_orig = GeneralEmpiricalUncertaintyModel(observed_fluxes=fluxes, observed_errors=errors)

    pkl_path = os.path.join(temp_dir, "model.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(model_orig, f)

    with open(pkl_path, "rb") as f:
        model_loaded = pickle.load(f)

    assert isinstance(model_loaded, GeneralEmpiricalUncertaintyModel)
    assert callable(model_loaded._mu_sigma_interpolator)

    test_flux = np.array([25.0])
    np.testing.assert_allclose(
        model_orig.sample_uncertainty(test_flux),
        model_loaded.sample_uncertainty(test_flux),
        atol=0.5,
    )


def test_hdf5_serialization(mock_data, temp_dir):
    """Test the HDF5 serialization and factory functions."""
    fluxes, errors = mock_data
    model_orig = GeneralEmpiricalUncertaintyModel(
        observed_fluxes=fluxes,
        observed_errors=errors,
        flux_unit="AB",
        upper_limits=True,
        treat_as_upper_limits_below=1.5,
    )

    hdf5_path = os.path.join(temp_dir, "model.hdf5")
    group_name = "test_model"

    save_unc_model_to_hdf5(model_orig, hdf5_path, group_name, overwrite=True)
    assert os.path.exists(hdf5_path)

    model_loaded = load_unc_model_from_hdf5(hdf5_path, group_name)

    assert isinstance(model_loaded, GeneralEmpiricalUncertaintyModel)
    assert callable(model_loaded._mu_sigma_interpolator)
    assert callable(model_loaded.log_snr_interpolator)
    assert model_orig.flux_unit == model_loaded.flux_unit
    assert model_orig.treat_as_upper_limits_below == model_loaded.treat_as_upper_limits_below

    test_flux = np.array([25.0])
    np.testing.assert_allclose(
        model_orig.sample_uncertainty(test_flux),
        model_loaded.sample_uncertainty(test_flux),
        atol=0.3,
    )


def test_apply_noise_preemptive_snr_check(mock_data):
    """Verifies that a source that is ALREADY low-SNR is never scattered."""
    fluxes, errors = mock_data
    model = GeneralEmpiricalUncertaintyModel(
        observed_fluxes=fluxes,
        observed_errors=errors,
        flux_unit="AB",
        upper_limits=True,
        treat_as_upper_limits_below=2.0,
        # Use a deterministic behaviour for this test
        upper_limit_flux_behaviour="upper_limit",
        return_noise=True,
    )

    # This flux is so faint its initial SNR will be << 2.0
    faint_flux = np.array([30.0])

    # Run the noise model many times
    outputs = [model.apply_noise(faint_flux, true_flux_units="AB") for _ in range(50)]
    output_fluxes = np.array([result[0][0] for result in outputs])

    # Since the source was caught by the pre-emptive mask, it should never
    # have had random noise added. The flux should be set deterministically
    # to the upper limit value every single time.
    assert len(np.unique(output_fluxes)) == 1
    assert output_fluxes[0] == pytest.approx(model.upper_limit_value)


@pytest.mark.parametrize(
    "flux_behaviour, expected_is_scattered",
    [
        ("upper_limit", False),
        (35.0, False),  # A numeric value is also deterministic
        ("scatter_limit", True),
    ],
)
def test_upper_limit_flux_behaviours(mock_data, flux_behaviour, expected_is_scattered):
    """Tests the different options for upper_limit_flux_behaviour."""
    fluxes, errors = mock_data
    model = GeneralEmpiricalUncertaintyModel(
        observed_fluxes=fluxes,
        observed_errors=errors,
        flux_unit="AB",
        upper_limits=True,
        treat_as_upper_limits_below=2.0,
        upper_limit_flux_behaviour=flux_behaviour,
        return_noise=True,
    )

    faint_flux = np.array([30.0])
    outputs = [model.apply_noise(faint_flux, true_flux_units="AB") for _ in range(50)]
    output_fluxes = np.array([result[0][0] for result in outputs])

    is_scattered = len(np.unique(output_fluxes)) > 1
    assert is_scattered == expected_is_scattered

    if not expected_is_scattered:
        # If deterministic, check it's the correct value
        expected_value = (
            model.upper_limit_value if flux_behaviour == "upper_limit" else float(flux_behaviour)
        )
        assert output_fluxes[0] == pytest.approx(expected_value)


@pytest.mark.parametrize(
    "err_behaviour, expected_error_func",
    [
        ("flux", lambda m: m._mu_sigma_interpolator(m.upper_limit_value)),
        ("upper_limit", lambda m: m.upper_limit_value),
        ("max", lambda m: m.max_flux_error),
        ("sig_1", lambda m: (2.5 / np.log(10)) / 1.0),  # Direct calculation for AB mags
        ("sig_3", lambda m: (2.5 / np.log(10)) / 3.0),  # Direct calculation for AB mags
    ],
)
def test_upper_limit_error_behaviours(mock_data, err_behaviour, expected_error_func):
    """Tests the different options for upper_limit_flux_err_behaviour."""
    fluxes, errors = mock_data
    model = GeneralEmpiricalUncertaintyModel(
        observed_fluxes=fluxes,
        observed_errors=errors,
        flux_unit="AB",
        interpolation_flux_unit="AB",  # Ensure we are in AB mag space for this test
        upper_limits=True,
        treat_as_upper_limits_below=2.0,
        upper_limit_flux_behaviour="upper_limit",  # Use deterministic flux for simplicity
        upper_limit_flux_err_behaviour=err_behaviour,
        max_flux_error=5.0 if err_behaviour == "max" else None,
        return_noise=True,
    )

    faint_flux = np.array([35.0])  # This flux is guaranteed to become an upper limit
    _, output_error = model.apply_noise(faint_flux, true_flux_units="AB")

    expected_error = expected_error_func(model)
    assert output_error[0] == pytest.approx(expected_error), (
        f"Expected error for {err_behaviour} did not match: {output_error[0]} != {expected_error}"
    )


if __name__ == "__main__":
    pytest.main([__file__])
