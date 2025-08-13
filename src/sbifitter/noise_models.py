"""
Noise models for simulating photometric fluxes and uncertainties.

This module provides a robust and serializable framework for creating and
applying various photometric noise models.
"""
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.table import Table
from scipy import stats
from scipy.interpolate import interp1d
from unyt import Jy, Unit, unyt_array, nJy

from .utils import f_jy_to_asinh, f_jy_err_to_asinh

# =============================================================================
# BASE CLASSES
# =============================================================================

class UncertaintyModel(ABC):
    """
    Abstract base class for photometric noise models.

    This class defines the common interface and provides static helper methods
    for photometric unit conversions. It is not meant to be instantiated directly.
    """

    def __init__(self, return_noise: bool = False, **kwargs: Any) -> None:
        self.return_noise = return_noise

    @abstractmethod
    def apply_noise(self, flux: np.ndarray | unyt_array) -> Union[
        np.ndarray | unyt_array,
        Tuple[np.ndarray | unyt_array, np.ndarray | unyt_array]
    ]:
        """Applies noise to the input flux."""
        pass

    @abstractmethod
    def serialize_to_hdf5(self, hdf5_group: h5py.Group):
        """Serializes the model's state into the given HDF5 group."""
        pass

    @classmethod
    @abstractmethod
    def _from_hdf5_group(cls, hdf5_group: h5py.Group) -> "UncertaintyModel":
        """Loads a model instance from an HDF5 group."""
        pass

    @staticmethod
    def ab_to_jy(magnitude: np.ndarray | float) -> unyt_array:
        """Converts AB magnitude to flux density in Janskys."""
        return (10 ** (-0.4 * (magnitude - 8.90))) * Jy

    @staticmethod
    def jy_to_ab(flux: unyt_array) -> np.ndarray:
        """Converts flux density in Janskys to AB magnitude."""
        return -2.5 * np.log10(flux.to_value(Jy)) + 8.90

    @staticmethod
    def ab_err_to_jy(magnitude_err: np.ndarray | float, flux_jy: unyt_array) -> unyt_array:
        """Converts AB magnitude uncertainty to flux uncertainty in Janskys."""
        return (flux_jy.to(Jy) * magnitude_err * np.log(10)) / 2.5

    @staticmethod
    def jy_err_to_ab(flux_err_jy: unyt_array, flux_jy: unyt_array) -> np.ndarray:
        """Converts flux uncertainty in Janskys to AB magnitude uncertainty."""
        return (2.5 / np.log(10)) * (flux_err_jy.to_value(Jy) / flux_jy.to_value(Jy))

class DepthUncertaintyModel(UncertaintyModel):
    """Applies Gaussian noise based on a fixed survey depth."""
    def __init__(self, depth_ab: float, depth_sigma_level: unyt_array = 5.0, **kwargs: Any):
        super().__init__(**kwargs)
        self.depth_ab = depth_ab
        self.depth_sigma_level = depth_sigma_level
        flux_limit_jy = self.ab_to_jy(self.depth_ab)
        self.sigma = (flux_limit_jy / self.depth_sigma_level).to(Jy)

    def apply_noise(self, flux: unyt_array, true_flux_units=None, **kwargs) -> Union[unyt_array, Tuple[unyt_array, unyt_array]]:
        
        if true_flux_units is not None:
            if true_flux_units == "AB":
                true_flux_jy = self.ab_to_jy(flux)
            else:
                true_flux_jy = (true_flux  * true_flux_units).to('Jy').value
        
        if len(kwargs) > 0:
            print(f'WARNING {kwargs} arguments will have no effect with this model')
        

        flux_jy = flux.to('Jy')

        if flux_jy.units.dimensions != Jy.dimensions:
            raise Exception("Input flux must be in Janskys (Jy).")
        noise = np.random.normal(loc=0.0, scale=self.sigma.to_value(Jy), size=flux_jy.shape) * Jy
        noisy_flux = flux_jy + noise
        if self.return_noise:
            uncertainty = np.ones_like(noisy_flux.value) * self.sigma
            return noisy_flux, uncertainty
        return noisy_flux

    def apply_scalings(self, flux: np.ndarray, error: np.ndarray, flux_units: str, out_units: str) -> Tuple[np.ndarray, np.ndarray]:
        """Applies only unit conversions, as this model has no other scalings."""
        if flux_units == out_units:
            return flux, error
        
        # Convert to a common intermediate unit (Jy)
        if flux_units == "AB":
            flux_jy = self.ab_to_jy(flux)
            error_jy = self.ab_err_to_jy(error, flux_jy)
        else:
            flux_jy = (flux * flux_units).to(Jy)
            error_jy = (error * flux_units).to(Jy)
        
        # Convert from Jy to the final output unit
        if out_units == "AB":
            return self.jy_to_ab(flux_jy), self.jy_err_to_ab(error_jy, flux_jy)
        else:
            return flux_jy.to_value(out_units), error_jy.to_value(out_units)

    def serialize_to_hdf5(self, hdf5_group: h5py.Group):
        """Serializes the model to an HDF5 group."""
        attrs = hdf5_group.attrs
        attrs["__class__"] = self.__class__.__name__
        attrs["depth_ab"] = self.depth_ab
        attrs["depth_sigma_level"] = self.depth_sigma_level
        attrs["return_noise"] = self.return_noise

    @classmethod
    def _from_hdf5_group(cls, hdf5_group: h5py.Group) -> "DepthUncertaintyModel":
        """Loads a model from an HDF5 group."""
        return cls(
            depth_ab=hdf5_group.attrs["depth_ab"],
            depth_sigma_level=hdf5_group.attrs["depth_sigma_level"],
            return_noise=hdf5_group.attrs["return_noise"],
        )

class EmpiricalUncertaintyModel(UncertaintyModel, ABC):
    """
    Abstract base for empirical uncertainty models from observed data.
    """
    def __init__(
        self,
        extrapolate: bool = False,
        min_samples_per_bin: int = 10,
        num_bins: int = 20,
        log_bins: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.extrapolate = extrapolate
        self._min_samples_per_bin = min_samples_per_bin
        self._num_bins = num_bins
        self._log_bins = log_bins
        self.bin_centers = None
        self.median_error_in_bin = None
        self.std_error_in_bin = None
        self._mu_sigma_interpolator = None
        self._sigma_sigma_interpolator = None

    def _compute_bins_from_data(self, fluxes: np.ndarray, errors: np.ndarray, precomputed_bins: Optional[np.ndarray] = None):
        if precomputed_bins is not None:
            bins = precomputed_bins
        else:
            valid_mask = np.isfinite(fluxes)
            if not np.any(valid_mask): raise ValueError("No valid finite data to build bins.")
            fluxes_for_bins = fluxes[valid_mask]
            if self._log_bins:
                positive_flux_mask = fluxes_for_bins > 0
                if not np.any(positive_flux_mask): raise ValueError("Log-binning requires positive flux values.")
                min_val, max_val = np.min(fluxes_for_bins[positive_flux_mask]), np.max(fluxes_for_bins)
                bins = np.logspace(np.log10(min_val), np.log10(max_val), self._num_bins + 1)
            else:
                min_val, max_val = np.min(fluxes_for_bins), np.max(fluxes_for_bins)
                bins = np.linspace(min_val, max_val, self._num_bins + 1)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median_err, bin_edges, _ = stats.binned_statistic(fluxes, errors, 'median', bins=bins)
            std_err, _, _ = stats.binned_statistic(fluxes, errors, np.std, bins=bins)
            counts, _, _ = stats.binned_statistic(fluxes, fluxes, 'count', bins=bins)

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        valid_bins_mask = counts >= self._min_samples_per_bin
        if np.sum(valid_bins_mask) < 2: raise ValueError("Could not create enough valid bins for interpolation.")

        self.bin_centers = bin_centers[valid_bins_mask]
        self.median_error_in_bin = median_err[valid_bins_mask]
        self.std_error_in_bin = std_err[valid_bins_mask]

    def _create_interpolators(self):
        if self.bin_centers is None or len(self.bin_centers) < 2:
            raise AttributeError("Binned data not found. Cannot create interpolators.")
        
        fill_median = "extrapolate" if self.extrapolate else (self.median_error_in_bin[0], self.median_error_in_bin[-1])
        fill_std = "extrapolate" if self.extrapolate else (self.std_error_in_bin[0], self.std_error_in_bin[-1])
        
        self._mu_sigma_interpolator = interp1d(
            x=self.bin_centers, y=self.median_error_in_bin, kind='linear',
            bounds_error=False, fill_value=fill_median
        )
        self._sigma_sigma_base_interpolator = interp1d(
            x=self.bin_centers, y=self.std_error_in_bin, kind='linear',
            bounds_error=False, fill_value=fill_std
        )
        # Assign the wrapper method, which is pickle-safe
        self._sigma_sigma_interpolator = self._non_negative_sigma_wrapper

    def _non_negative_sigma_wrapper(self, flux_values: np.ndarray) -> np.ndarray:
        """Pickle-safe wrapper to ensure sigma_sigma is never negative."""
        std_devs = self._sigma_sigma_base_interpolator(flux_values)
        return np.maximum(0, std_devs)

    def sample_uncertainty(self, flux_values: np.ndarray) -> np.ndarray:
        """Samples an uncertainty from the learned distribution p(sigma|f)."""
        mu_sigma = self._mu_sigma_interpolator(flux_values)
        sigma_sigma = self._sigma_sigma_interpolator(flux_values)
        a = (0 - mu_sigma) / np.where(sigma_sigma > 1e-9, sigma_sigma, 1)
        return stats.truncnorm.rvs(a=a, b=np.inf, loc=mu_sigma, scale=sigma_sigma, size=len(flux_values))

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state.pop('_mu_sigma_interpolator', None)
        state.pop('_sigma_sigma_interpolator', None)
        state.pop('_sigma_sigma_base_interpolator', None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        if self.bin_centers is not None:
            self._create_interpolators()

    def serialize_to_hdf5(self, hdf5_group: h5py.Group):
        """Serializes the common state of an empirical model."""
        attrs = hdf5_group.attrs
        attrs["__class__"] = self.__class__.__name__
        attrs["extrapolate"] = self.extrapolate
        attrs["min_samples_per_bin"] = self._min_samples_per_bin
        attrs["num_bins"] = self._num_bins
        attrs["log_bins"] = self._log_bins

        if self.bin_centers is not None:
            hdf5_group.create_dataset("bin_centers", data=self.bin_centers)
            hdf5_group.create_dataset("median_error_in_bin", data=self.median_error_in_bin)
            hdf5_group.create_dataset("std_error_in_bin", data=self.std_error_in_bin)

    @classmethod
    def _from_hdf5_group(cls, hdf5_group: h5py.Group) -> "EmpiricalUncertaintyModel":
        """Loads the common state for an empirical model."""
        init_args = {
            "extrapolate": hdf5_group.attrs.get("extrapolate", False),
            "min_samples_per_bin": hdf5_group.attrs.get("min_samples_per_bin", 10),
            "num_bins": hdf5_group.attrs.get("num_bins", 20),
            "log_bins": hdf5_group.attrs.get("log_bins", True),
        }
        # Create an empty instance by calling __init__ with no data
        instance = cls.__new__(cls)
        super(EmpiricalUncertaintyModel, instance).__init__(**init_args)

        # Manually populate the binned data and reconstruct interpolators
        if "bin_centers" in hdf5_group:
            instance.bin_centers = hdf5_group["bin_centers"][:]
            instance.median_error_in_bin = hdf5_group["median_error_in_bin"][:]
            instance.std_error_in_bin = hdf5_group["std_error_in_bin"][:]
            instance._create_interpolators()

        return instance

class AsinhEmpiricalUncertaintyModel(EmpiricalUncertaintyModel):
    """An empirical model for uncertainties in asinh magnitude space."""
    def __init__(self, observed_phot_jy: unyt_array, observed_phot_errors_jy: unyt_array, asinh_b_factor = 5.0, error_type: str = "empirical", **kwargs: Any):
        kwargs['log_bins'] = False
        super().__init__(**kwargs)
        self.error_type = error_type
        
        valid = np.isfinite(observed_phot_jy) & np.isfinite(observed_phot_errors_jy)
        flux_jy, error_jy = observed_phot_jy[valid], observed_phot_errors_jy[valid]
        
        self.b = asinh_b_factor * np.median(error_jy)
        mag_asinh = f_jy_to_asinh(flux_jy, self.b)
        mag_err_asinh = f_jy_err_to_asinh(flux_jy, error_jy, self.b)
        
        self._compute_bins_from_data(fluxes=mag_asinh, errors=mag_err_asinh)
        self._create_interpolators()

    def apply_noise(self, true_flux: unyt_array, true_flux_units=None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        if len(kwargs) > 0:
            print(f'WARNING {kwargs} arguments will have no effect with this model')
        
        if true_flux_units is not None:
            true_flux_jy = (true_flux  * true_flux_units).to('Jy').value
        else:
            true_flux_jy = true_flux
    
        true_mag_asinh = f_jy_to_asinh(true_flux_jy, self.b)
        sampled_err_asinh = self.sample_uncertainty(true_mag_asinh)
        noise = np.random.normal(loc=0.0, scale=sampled_err_asinh)
        noisy_mag_asinh = true_mag_asinh + noise
        final_err = sampled_err_asinh if self.error_type == "empirical" else self.sample_uncertainty(noisy_mag_asinh)
        return (noisy_mag_asinh, final_err) if self.return_noise else noisy_mag_asinh

    def apply_scalings(self, flux: unyt_array, error: unyt_array, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Converts flux and error from Jy to asinh magnitudes."""
        if len(kwargs) > 0:
            print(f'WARNING {kwargs} arguments will have no effect with this model')
        
        mag = f_jy_to_asinh(flux, self.b)
        mag_err = f_jy_err_to_asinh(flux, error, self.b)
        return mag, mag_err

    def serialize_to_hdf5(self, hdf5_group: h5py.Group):
        """Saves the asinh model, including its unique attributes."""
        # Call the parent method to save the binned data and common config
        super().serialize_to_hdf5(hdf5_group)
        
        # Save attributes specific to this class
        attrs = hdf5_group.attrs
        attrs["error_type"] = self.error_type
        attrs["b_value"] = self.b.to_value()
        attrs["b_units"] = str(self.b.units)

    @classmethod
    def _from_hdf5_group(cls, hdf5_group: h5py.Group) -> "AsinhEmpiricalUncertaintyModel":
        """Loads the asinh model, including its unique attributes."""
        # Call the parent method to load the common parts (binned data, etc.)
        instance = super(AsinhEmpiricalUncertaintyModel, cls)._from_hdf5_group(hdf5_group)
        
        # Recast the instance to the correct class
        instance.__class__ = cls
        
        # Load attributes specific to this class
        attrs = hdf5_group.attrs
        instance.error_type = attrs["error_type"]
        instance.b = unyt_array(attrs["b_value"], attrs["b_units"])
        
        return instance


class GeneralEmpiricalUncertaintyModel(EmpiricalUncertaintyModel):
    """
    A class to model and sample photometric uncertainties based on observations,
    with full support for all original features and robust serialization.
    """
    def __init__(
        self,
        observed_fluxes: np.ndarray,
        observed_errors: np.ndarray,
        flux_unit: str = "AB",
        interpolation_flux_unit: Optional[str] = None,
        already_binned: bool = False,
        bin_median_errors: Optional[np.ndarray] = None,
        bin_std_errors: Optional[np.ndarray] = None,
        flux_bins: Optional[np.ndarray] = None,
        min_flux_for_binning: Optional[float] = None,
        sigma_clip: float = 3.0,
        error_type: str = "empirical",
        upper_limits: bool = False,
        treat_as_upper_limits_below: Optional[float] = None,
        upper_limit_flux_behaviour: str = "scatter_limit",
        upper_limit_flux_err_behaviour: str = "flux",
        **kwargs: Any,
    ):
        # 1. Initialize parent and instance attributes
        super().__init__(**kwargs)
        self.flux_unit = flux_unit
        self.interpolation_flux_unit = interpolation_flux_unit if interpolation_flux_unit else flux_unit
        self.sigma_clip = sigma_clip
        self.error_type = error_type
        self.upper_limits = upper_limits
        self.treat_as_upper_limits_below = treat_as_upper_limits_below
        self.upper_limit_flux_behaviour = upper_limit_flux_behaviour
        self.upper_limit_flux_err_behaviour = upper_limit_flux_err_behaviour
        self.log_snr_interpolator = None
        self.upper_limit_value = None

        # 2. Handle the 'already_binned' case first
        if already_binned:
            self.bin_centers = observed_fluxes
            self.median_error_in_bin = bin_median_errors
            self.std_error_in_bin = bin_std_errors
            self._create_interpolators()
            # Note: For pre-binned data, SNR interpolator cannot be built from raw data.
            # This would need to be handled separately if required.
            return

        # 3. Process raw data if not already binned
        flux_to_process, error_to_process = self._convert_units(observed_fluxes, observed_errors)
        
        valid_mask = np.isfinite(flux_to_process) & np.isfinite(error_to_process) & (error_to_process > 0)
        if min_flux_for_binning is not None:
            valid_mask &= flux_to_process > min_flux_for_binning
        
        self._compute_bins_from_data(
            fluxes=flux_to_process[valid_mask],
            errors=error_to_process[valid_mask],
            precomputed_bins=flux_bins
        )
        
        if self.upper_limits:
            self._setup_upper_limit_interpolator(flux_to_process[valid_mask], error_to_process[valid_mask])

        self._create_interpolators()

    def _convert_units(self, fluxes: np.ndarray, errors: np.ndarray, fluxes_unit=None) -> Tuple[np.ndarray, np.ndarray]:
        """Helper to handle unit conversion for binning data."""
        
        if fluxes_unit is None:
            fluxes_unit = self.flux_unit

        if (self.interpolation_flux_unit == fluxes_unit) or (isinstance(self.interpolation_flux_unit, unyt_array) and isinstance(fluxes_unit, unyt_array) and (self.interpolation_flux_unit.dimensions == fluxes_unit.dimensions)):
            if self.interpolation_flux_unit == fluxes_unit:
                conversion = 1.0
            else:
                conversion = (self.interpolation_flux_unit/fluxes_unit).simplify()

            return fluxes*conversion, errors*conversion
        if fluxes_unit == 'AB': # AB to physical flux
            flux_jy = self.ab_to_jy(fluxes)
            error_jy = self.ab_err_to_jy(errors, flux_jy)
            return flux_jy.to_value(self.interpolation_flux_unit), error_jy.to_value(self.interpolation_flux_unit)
        else: # Physical flux to AB
            flux_with_units = fluxes * fluxes_unit
            error_with_units = errors * fluxes_unit
            return self.jy_to_ab(flux_with_units), self.jy_err_to_ab(error_with_units, flux_with_units)

    def _setup_upper_limit_interpolator(self, fluxes: np.ndarray, errors: np.ndarray):
        """Creates the SNR interpolator, always using physical flux units."""
        # This interpolator is ALWAYS flux vs SNR, so we convert to Jy
        # regardless of interpolation_flux_unit.
        if self.interpolation_flux_unit == "AB":
            flux_jy = self.ab_to_jy(fluxes)
            error_jy = self.ab_err_to_jy(errors, flux_jy)
        else:
            flux_jy = (fluxes * self.interpolation_flux_unit).to(Jy)
            error_jy = (errors * self.interpolation_flux_unit).to(Jy)
            
        with np.errstate(divide='ignore', invalid='ignore'):
            snr = (flux_jy / error_jy).value
        
        valid = np.isfinite(snr) & (snr > 0) & np.isfinite(flux_jy.value) & (flux_jy.value > 0)
        if np.sum(valid) < 2: return
            
        order = np.argsort(snr[valid])
        self._snr_x_data = np.log10(snr[valid][order])
        self._snr_y_data = np.log10(flux_jy.value[valid][order])
        
        self.log_snr_interpolator = interp1d(
            self._snr_x_data, self._snr_y_data,
            kind="linear", bounds_error=False, fill_value="extrapolate"
        )
        ul_flux_jy = 10**self.log_snr_interpolator(np.log10(self.treat_as_upper_limits_below))
        
        if self.interpolation_flux_unit == "AB":
            self.upper_limit_value = self.jy_to_ab(ul_flux_jy * Jy)
        else:
            self.upper_limit_value = (ul_flux_jy * Jy).to_value(self.interpolation_flux_unit)

    def apply_noise(self, true_flux: np.ndarray, true_flux_units=None, out_units=None) -> Tuple[np.ndarray, np.ndarray]:
        """Applies configured noise and upper limit rules to true flux values."""
        # 1. Convert input flux to the model's internal interpolation units
        flux_internal, _ = self._convert_units(true_flux, np.zeros_like(true_flux), true_flux_units)
        
        # 2. Sample uncertainty and add noise in internal units
        sampled_sigma_internal = self.sample_uncertainty(flux_internal)
        noise = stats.truncnorm.rvs(-self.sigma_clip, self.sigma_clip, 0, sampled_sigma_internal, size=len(true_flux))
        noisy_flux_internal = flux_internal + noise
        
        final_sigma_internal = sampled_sigma_internal
        if self.error_type == "observed":
            final_sigma_internal = self.sample_uncertainty(noisy_flux_internal)

        # 3. Handle upper limits if specified
        if self.upper_limits and self.upper_limit_value is not None:
            if self.interpolation_flux_unit == "AB":
                flux_jy = self.ab_to_jy(noisy_flux_internal)
                error_jy = self.ab_err_to_jy(final_sigma_internal, flux_jy)
            else:
                flux_jy = (noisy_flux_internal * self.interpolation_flux_unit).to(Jy)
                error_jy = (final_sigma_internal * self.interpolation_flux_unit).to(Jy)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                snr = (flux_jy / error_jy).value
            
            limit_mask = ~np.isfinite(snr) | (snr < self.treat_as_upper_limits_below)
            
            if np.any(limit_mask):
                noisy_flux_internal[limit_mask] = self.upper_limit_value
                # Simplified: for now, assign a constant error. More complex behaviours can be added.
                if self.upper_limit_flux_err_behaviour == 'flux':
                     final_sigma_internal[limit_mask] = self.sample_uncertainty(np.full_like(noisy_flux_internal[limit_mask], self.upper_limit_value))

        # 4. Convert results back to the original input units
        out_flux, out_sigma = self._convert_units_inverse(noisy_flux_internal, final_sigma_internal, out_units)
        
        return (out_flux, out_sigma) if self.return_noise else out_flux

    def _convert_units_inverse(self, fluxes: np.ndarray, errors: np.ndarray, out_unit=None) -> Tuple[np.ndarray, np.ndarray]:
        """Helper to convert from internal units back to the model's primary flux_unit."""
        
        if out_unit is None:
            out_unit = self.flux_unit

        if (self.interpolation_flux_unit == out_unit) or (isinstance(self.interpolation_flux_unit, unyt_array) and isinstance(out_unit, unyt_array) and (self.interpolation_flux_unit.dimensions == out_unit.dimensions)):
            if self.interpolation_flux_unit == out_unit:
                return fluxes, errors
            else:
                conversion = (out_unit/self.interpolation_flux_unit).simplify()
                return fluxes*conversion, errors*conversion

        # This is the reverse of _convert_units
        if self.interpolation_flux_unit == 'AB': # AB to physical flux
            flux_jy = self.ab_to_jy(fluxes)
            error_jy = self.ab_err_to_jy(errors, flux_jy)
            return flux_jy.to_value(out_unit), error_jy.to_value(out_unit)
        else: # Physical flux to AB
            flux_with_units = fluxes * self.interpolation_flux_unit
            error_with_units = errors  * self.interpolation_flux_unit
            return self.jy_to_ab(flux_with_units), self.jy_err_to_ab(error_with_units, flux_with_units)

    def serialize_to_hdf5(self, hdf5_group: h5py.Group):
        """Serializes the model's state into the given HDF5 group."""
        attrs = hdf5_group.attrs
        attrs["__class__"] = self.__class__.__name__
        
        # Save binned data and config
        if self.bin_centers is not None:
            hdf5_group.create_dataset("bin_centers", data=self.bin_centers)
            hdf5_group.create_dataset("median_error_in_bin", data=self.median_error_in_bin)
            hdf5_group.create_dataset("std_error_in_bin", data=self.std_error_in_bin)
        
        if self.log_snr_interpolator is not None:
            hdf5_group.create_dataset("snr_x_data", data=self._snr_x_data)
            hdf5_group.create_dataset("snr_y_data", data=self._snr_y_data)
        
        # Save all __init__ parameters for perfect reconstruction
        attrs["flux_unit"] = self.flux_unit
        attrs["interpolation_flux_unit"] = self.interpolation_flux_unit
        attrs["sigma_clip"] = self.sigma_clip
        attrs["error_type"] = self.error_type
        attrs["upper_limits"] = self.upper_limits
        attrs["treat_as_upper_limits_below"] = self.treat_as_upper_limits_below if self.treat_as_upper_limits_below is not None else "None"
        attrs["upper_limit_flux_behaviour"] = self.upper_limit_flux_behaviour
        attrs["upper_limit_flux_err_behaviour"] = self.upper_limit_flux_err_behaviour
        attrs["extrapolate"] = self.extrapolate
        attrs["min_samples_per_bin"] = self._min_samples_per_bin
        attrs["num_bins"] = self._num_bins
        attrs["log_bins"] = self._log_bins

    @classmethod
    def _from_hdf5_group(cls, hdf5_group: h5py.Group) -> "GeneralEmpiricalUncertaintyModel":
        """Loads a model instance from an HDF5 group."""
        attrs = hdf5_group.attrs
        
        # Use the already_binned=True path for clean reconstruction
        init_args = {
            "observed_fluxes": hdf5_group["bin_centers"][:],
            "observed_errors": None, # Not needed for this path
            "already_binned": True,
            "bin_median_errors": hdf5_group["median_error_in_bin"][:],
            "bin_std_errors": hdf5_group["std_error_in_bin"][:],
            "flux_unit": attrs["flux_unit"],
            "interpolation_flux_unit": attrs["interpolation_flux_unit"],
            "sigma_clip": attrs["sigma_clip"],
            "error_type": attrs["error_type"],
            "upper_limits": attrs["upper_limits"],
            "treat_as_upper_limits_below": None if attrs["treat_as_upper_limits_below"] == "None" else attrs["treat_as_upper_limits_below"],
            "upper_limit_flux_behaviour": attrs["upper_limit_flux_behaviour"],
            "upper_limit_flux_err_behaviour": attrs["upper_limit_flux_err_behaviour"],
            "extrapolate": attrs["extrapolate"],
            "min_samples_per_bin": attrs["min_samples_per_bin"],
            "num_bins": attrs["num_bins"],
            "log_bins": attrs["log_bins"],
        }
        
        instance = cls(**init_args)
        
        # Manually reconstruct the SNR interpolator
        if "snr_x_data" in hdf5_group:
            instance._snr_x_data = hdf5_group["snr_x_data"][:]
            instance._snr_y_data = hdf5_group["snr_y_data"][:]
            instance.log_snr_interpolator = interp1d(
                instance._snr_x_data, instance._snr_y_data,
                kind="linear", bounds_error=False, fill_value="extrapolate"
            )

        return instance

    def apply_scalings(self, flux: np.ndarray, error: np.ndarray, flux_units: str, out_units: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies deterministic model transformations (units, SNR cuts)
        to observed data without adding random noise.
        """
        # 1. Convert input flux to the model's internal interpolation units
        flux_internal, error_internal = self._convert_units(flux, error)

        # 2. Apply upper limit rule (SNR cut) without random scatter
        if self.upper_limits and self.upper_limit_value is not None:
            if self.interpolation_flux_unit == "AB":
                flux_jy = self.ab_to_jy(flux_internal)
                error_jy = self.ab_err_to_jy(error_internal, flux_jy)
            else:
                flux_jy = (flux_internal * u.Unit(self.interpolation_flux_unit)).to(u.Jy)
                error_jy = (error_internal * u.Unit(self.interpolation_flux_unit)).to(u.Jy)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                snr = (flux_jy / error_jy).value
            
            limit_mask = ~np.isfinite(snr) | (snr < self.treat_as_upper_limits_below)
            
            if np.any(limit_mask):
                flux_internal[limit_mask] = self.upper_limit_value
                # Replace error with the model's median error at the limit
                error_internal[limit_mask] = self._mu_sigma_interpolator(self.upper_limit_value)
        
        # 3. Convert results to the desired output units
        if self.interpolation_flux_unit == out_units:
            return flux_internal, error_internal
        else:
            return self._convert_units_inverse(flux_internal, error_internal, out_units)

# =============================================================================
# SERIALIZATION FACTORY FUNCTIONS
# =============================================================================

MODEL_CLASS_REGISTRY = {
    "DepthUncertaintyModel": DepthUncertaintyModel, 
    "AsinhEmpiricalUncertaintyModel": AsinhEmpiricalUncertaintyModel,
    "GeneralEmpiricalUncertaintyModel": GeneralEmpiricalUncertaintyModel,
}

def save_unc_model_to_hdf5(model: UncertaintyModel, filepath: str, group_name: str, overwrite: bool = False):
    """Saves a supported uncertainty model to an HDF5 file."""
    with h5py.File(filepath, "a") as f:
        if group_name in f:
            if overwrite: del f[group_name]
            else: raise ValueError(f"Group '{group_name}' already exists.")
        group = f.create_group(group_name)
        model.serialize_to_hdf5(group)

def load_unc_model_from_hdf5(filepath: str, group_name: str) -> UncertaintyModel:
    """Factory function to load any supported model from an HDF5 file."""
    with h5py.File(filepath, "r") as f:
        if group_name not in f: raise KeyError(f"Group '{group_name}' not found.")
        group = f[group_name]
        class_name = group.attrs.get("__class__")
        if class_name not in MODEL_CLASS_REGISTRY: raise TypeError(f"Unknown model class '{class_name}'.")
        return MODEL_CLASS_REGISTRY[class_name]._from_hdf5_group(group)

def create_uncertainty_models_from_EPOCHS_cat(
    file,
    bands,
    new_band_names=None,
    plot=False,
    old=False,
    hdu=0,
    save=False,
    save_path=None,
    **kwargs,
):
    """Create uncertainty models from an EPOCHS catalog file.

    Parameters
    ----------
    file : str
        Path to the EPOCHS catalog file.
    bands : str or list of str
        Band(s) to create uncertainty models for. If a string is provided,
        it will be converted to a list.
    new_band_names : list of str, optional
        New names for the bands in the uncertainty models. If not provided,
        the original band names will be used.
    plot : bool, optional
        Whether to plot the uncertainty models. Default is False.
    old : bool, optional
        If True, assumes the catalog is in the old format (without aperture corrections).
        Default is False.
    hdu : int, optional
        The HDU number to read from the FITS file. Default is 0.
    save_path : str, optional
        Path to save the plots if `plot` is True. If None, plots are not saved.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the EmpiricalUncertaintyModel.

    Returns:
    -------
    dict
        A dictionary of EmpiricalUncertaintyModel objects for each band.
    """

    from astropy import units as u

    if isinstance(bands, str):
        bands = [bands]

    if not isinstance(file, Table):
        table = Table.read(file, hdu=hdu)
    else:
        table = file
    unc_models = {}

    if new_band_names is not None:
        assert len(new_band_names) == len(
            bands
        ), f"""new_band_names length {len(new_band_names)} does not match bands
            length {len(bands)}. Cannot create uncertainty models."""
    else:
        new_band_names = bands

    for band, band_new_name in zip(bands, new_band_names):
        if f"loc_depth_{band}" not in table.colnames:
            print(table.colnames)
            raise ValueError(f"Column loc_depth_{band} not found in the table.")

        mag = table[f"MAG_APER_{band}_aper_corr"]

        flux = (u.Jy * table[f"FLUX_APER_{band}_aper_corr_Jy"]).to("Jy").value
        flux_err = (table[f"loc_depth_{band}"] * u.ABmag).to("Jy").value / 5

        if old:
            mag = mag[:, 0]
            flux = flux[:, 0]
            flux_err = flux[:, 0]

        mag_err = (2.5 * flux_err) / (flux * np.log(10))
        mask = (mag != -99) & (np.isfinite(mag)) & (mag_err >= 0)
        mag = mag[mask]
        mag_err = mag_err[mask]

        unc_kwargs = dict(
            num_bins=20,
            log_bins=False,
            error_type="observed",
            upper_limits=True,
            treat_as_upper_limits_below=1,
            upper_limit_flux_behaviour=40,
            upper_limit_flux_err_behaviour="sig_1",
        )

        unc_kwargs.update(kwargs)

        # So this behaviour is to mask any fluxes with SNR < 1 either
        # before or after the scattering,
        # , setting the error to 1 sigma.
        noise_model = GeneralEmpiricalUncertaintyModel(
            mag,
            mag_err,
            **unc_kwargs,
        )

        unc_models[band_new_name] = noise_model

        if plot:
            # bin and plot as contour
            plt.figure(figsize=(10, 6))

            plt.title(f"Uncertainty Model for {band_new_name}", fontsize=16)

            plt.scatter(mag, mag_err, alpha=0.05, color="black", s=0.15, zorder=10)

            plt.ylim(0, 1.2)
            mag = np.linspace(23, 40, 10000)
            noisy_flux, sampled_sigma = noise_model.apply_noise(mag)

            # plt.scatter(noisy_flux, sampled_sigma, alpha=0.1, color='green', s=0.1)
            plt.hexbin(
                noisy_flux,
                sampled_sigma,
                gridsize=50,
                cmap="Greens",
                mincnt=1,
                norm="log",
                extent=(23, 42, 0, 1.1),
                alpha=1,
                label=r"$p\left(\sigma_X \mid f_X\right)$",
            )
            plt.legend(loc="upper left", fontsize=12)

            plt.xlabel("Magnitude", fontsize=14)
            plt.ylabel(r"$\sigma_{\rm m, AB}$", fontsize=14)
            if save:
                save_band_name = band_new_name.replace("/", "_")
                plt.savefig(f"{save_path}/uncertainty_model_{save_band_name}.png", dpi=300)
            else:
                plt.show()
    return unc_models
