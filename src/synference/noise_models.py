"""Noise models for simulating photometric fluxes and uncertainties.

This module provides a robust and serializable framework for creating and
applying various photometric noise models.
"""

import json
import math
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from astropy.table import Table
from scipy import stats
from scipy.interpolate import interp1d
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from unyt import Jy, Unit, unyt_array

from .utils import f_jy_err_to_asinh, f_jy_to_asinh

# =============================================================================
# BASE CLASSES
# =============================================================================


class UncertaintyModel(ABC):
    """Abstract base class for photometric noise models.

    This class defines the common interface and provides static helper methods
    for photometric unit conversions. It is not meant to be instantiated directly.
    """

    def __init__(self, return_noise: bool = False, **kwargs: Any) -> None:
        """Initializes the uncertainty model."""
        self.return_noise = return_noise

    @abstractmethod
    def apply_noise(
        self, flux: np.ndarray | unyt_array
    ) -> Union[np.ndarray | unyt_array, Tuple[np.ndarray | unyt_array, np.ndarray | unyt_array]]:
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
        return np.abs((2.5 / np.log(10)) * (flux_err_jy.to_value(Jy) / flux_jy.to_value(Jy)))


class DepthUncertaintyModel(UncertaintyModel):
    """Applies Gaussian noise based on a fixed survey depth."""

    def __init__(
        self,
        depth_ab: float,
        depth_sigma_level: unyt_array = 5.0,
        min_flux_error: Optional[float] = None,
        max_flux_error: Optional[float] = None,
        **kwargs: Any,
    ):
        """Initializes the model with a fixed depth in AB magnitudes.

        Args:
            depth_ab (float): The depth of the survey in AB magnitudes.
            depth_sigma_level (unyt_array): The sigma level for the depth, default is 5.0.
            min_flux_error (Optional[float]): Minimum flux error to apply, default is 0.0.
                Should be in Janskys (Jy).
            max_flux_error (Optional[float]): Maximum flux error to apply, default is np.inf.
                Should be in Janskys (Jy).
            kwargs: Additional keyword arguments for the base class.

        Returns:
            None
        """
        super().__init__(**kwargs)
        self.depth_ab = depth_ab
        self.depth_sigma_level = depth_sigma_level
        flux_limit_jy = self.ab_to_jy(self.depth_ab)
        self.sigma = (flux_limit_jy / self.depth_sigma_level).to(Jy)

        self.min_flux_error = min_flux_error if min_flux_error is not None else 0.0
        self.max_flux_error = max_flux_error if max_flux_error is not None else np.inf

        assert isinstance(self.sigma, unyt_array), "sigma must be a unyt_array with units of Jy"
        assert not np.isnan(self.sigma.value), "sigma must not be NaN"

    def apply_noise(
        self, flux: unyt_array, true_flux_units=None, out_units=None, **kwargs
    ) -> Union[unyt_array, Tuple[unyt_array, unyt_array]]:
        """Applies Gaussian noise to the input flux."""
        if true_flux_units is not None:
            if true_flux_units == "AB":
                true_flux_jy = self.ab_to_jy(flux)
            else:
                if isinstance(flux, unyt_array):
                    assert true_flux_units == flux.units, (
                        "If true_flux_units is specified, "
                        "flux must be a unyt_array with the same units."
                    )
                    flux = flux.to_value(true_flux_units)

                if isinstance(true_flux_units, str):
                    true_flux_units = Unit(true_flux_units)

                true_flux_jy = (flux * true_flux_units).to("Jy")
        else:
            if not isinstance(flux, unyt_array):
                true_flux_jy = unyt_array(flux, "Jy")
            else:
                true_flux_jy = flux.to("Jy")

        if len(kwargs) > 0:
            print(f"WARNING {kwargs} arguments will have no effect with this model.")

        if true_flux_jy.units.dimensions != Jy.dimensions:
            raise Exception("Input flux must be in Janskys (Jy).")

        flux_jy = true_flux_jy.to("Jy")

        noise = np.random.normal(loc=0.0, scale=self.sigma.to_value(Jy), size=flux_jy.shape) * Jy
        noisy_flux = flux_jy + noise

        uncertainty = np.ones_like(noisy_flux.value) * self.sigma

        if out_units is not None:
            if out_units == "AB":
                uncertainty = self.jy_err_to_ab(uncertainty, noisy_flux)
                noisy_flux = self.jy_to_ab(noisy_flux)
            else:
                noisy_flux = noisy_flux.to_value(out_units)
                uncertainty = uncertainty.to_value(out_units)

        uncertainty = np.clip(uncertainty, self.min_flux_error, self.max_flux_error)

        if self.return_noise:
            return noisy_flux, uncertainty

        return noisy_flux

    def apply_scalings(
        self, flux: np.ndarray, error: np.ndarray, flux_units: str, out_units: str
    ) -> Tuple[np.ndarray, np.ndarray]:
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

        error_jy = np.clip(error_jy, self.min_flux_error, self.max_flux_error)

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
        attrs["min_flux_error"] = self.min_flux_error
        attrs["max_flux_error"] = self.max_flux_error

    @classmethod
    def _from_hdf5_group(cls, hdf5_group: h5py.Group) -> "DepthUncertaintyModel":
        """Loads a model from an HDF5 group."""
        return cls(
            depth_ab=hdf5_group.attrs["depth_ab"],
            depth_sigma_level=hdf5_group.attrs["depth_sigma_level"],
            return_noise=hdf5_group.attrs["return_noise"],
            min_flux_error=hdf5_group.attrs.get("min_flux_error", 0.0),
            max_flux_error=hdf5_group.attrs.get("max_flux_error", np.inf),
        )


class SpectralUncertaintyModel(UncertaintyModel):
    """Applies uncertanties to a spectrum based on a fixed error kernel or a provided table."""

    def __init__(self, error_kernel: np.ndarray, **kwargs: Any):
        """Initializes the model with a fixed error kernel.

        Args:
            error_kernel (np.ndarray): An array of uncertainties to apply to the spectrum.
            kwargs: Additional keyword arguments for the base class.

        Returns:
            None
        """
        super().__init__(**kwargs)
        self.error_kernel = error_kernel

    def apply_noise(
        self, flux: np.ndarray, **kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Applies Gaussian noise to the input flux based on the error kernel."""
        if len(kwargs) > 0:
            print(f"WARNING {kwargs} arguments will have no effect with this model.")

        if flux.shape != self.error_kernel.shape:
            raise ValueError("Input flux shape must match the error kernel shape.")

        noise = np.random.normal(loc=0.0, scale=self.error_kernel, size=flux.shape)
        noisy_flux = flux + noise

        if self.return_noise:
            return noisy_flux, self.error_kernel

        return noisy_flux

    def serialize_to_hdf5(self, hdf5_group: h5py.Group):
        """Serializes the model to an HDF5 group."""
        attrs = hdf5_group.attrs
        attrs["__class__"] = self.__class__.__name__
        attrs["return_noise"] = self.return_noise
        hdf5_group.create_dataset("error_kernel", data=self.error_kernel)

    @classmethod
    def _from_hdf5_group(cls, hdf5_group: h5py.Group) -> "SpectralUncertaintyModel":
        """Loads a model from an HDF5 group."""
        error_kernel = hdf5_group["error_kernel"][:]
        return cls(
            error_kernel=error_kernel,
            return_noise=hdf5_group.attrs["return_noise"],
        )


class EmpiricalUncertaintyModel(UncertaintyModel, ABC):
    """Abstract base for empirical uncertainty models from observed data."""

    def __init__(
        self,
        extrapolate: bool = False,
        min_samples_per_bin: int = 10,
        num_bins: int = 20,
        log_bins: bool = True,
        **kwargs: Any,
    ):
        """Initializes the empirical uncertainty model."""
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

    def _compute_bins_from_data(
        self, fluxes: np.ndarray, errors: np.ndarray, precomputed_bins: Optional[np.ndarray] = None
    ):
        if precomputed_bins is not None:
            bins = precomputed_bins
        else:
            valid_mask = np.isfinite(fluxes)
            if not np.any(valid_mask):
                raise ValueError("No valid finite data to build bins.")
            fluxes_for_bins = fluxes[valid_mask]
            if self._log_bins:
                positive_flux_mask = fluxes_for_bins > 0
                if not np.any(positive_flux_mask):
                    raise ValueError("Log-binning requires positive flux values.")
                min_val, max_val = (
                    np.min(fluxes_for_bins[positive_flux_mask]),
                    np.max(fluxes_for_bins),
                )
                bins = np.logspace(np.log10(min_val), np.log10(max_val), self._num_bins + 1)
            else:
                min_val, max_val = np.min(fluxes_for_bins), np.max(fluxes_for_bins)
                bins = np.linspace(min_val, max_val, self._num_bins + 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median_err, bin_edges, _ = stats.binned_statistic(fluxes, errors, "median", bins=bins)
            std_err, _, _ = stats.binned_statistic(fluxes, errors, np.std, bins=bins)
            counts, _, _ = stats.binned_statistic(fluxes, fluxes, "count", bins=bins)

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        valid_bins_mask = counts >= self._min_samples_per_bin
        if np.sum(valid_bins_mask) < 2:
            raise ValueError("Could not create enough valid bins for interpolation.")

        self.bin_centers = bin_centers[valid_bins_mask]
        self.median_error_in_bin = median_err[valid_bins_mask]
        self.std_error_in_bin = std_err[valid_bins_mask]

    def plot(self, ax: Optional[plt.Axes] = None):
        """Plots the binned median error and standard deviation."""
        fig, ax = plt.subplots() if ax is None else (ax.get_figure(), ax)
        if self.bin_centers is None or len(self.bin_centers) < 2:
            raise AttributeError("Binned data not found. Cannot plot.")

        ax.errorbar(
            self.bin_centers,
            self.median_error_in_bin,
            yerr=self.std_error_in_bin,
            fmt="o",
            label="Median Error",
            color="blue",
            alpha=0.7,
        )

        f = "Flux"
        if isinstance(self, AsinhEmpiricalUncertaintyModel):
            f = "Mag [asinh]"

        ax.set_xlabel(f)
        ax.set_ylabel("Error")
        ax.legend()
        return fig

    def _create_interpolators(self):
        if self.bin_centers is None or len(self.bin_centers) < 2:
            raise AttributeError("Binned data not found. Cannot create interpolators.")
        fill_median = (
            "extrapolate"
            if getattr(self, "extrapolate", False)
            else (self.median_error_in_bin[0], self.median_error_in_bin[-1])
        )
        fill_std = (
            "extrapolate"
            if getattr(self, "extrapolate", False)
            else (self.std_error_in_bin[0], self.std_error_in_bin[-1])
        )

        self._mu_sigma_interpolator = interp1d(
            x=self.bin_centers,
            y=self.median_error_in_bin,
            kind="linear",
            bounds_error=False,
            fill_value=fill_median,
        )
        self._sigma_sigma_base_interpolator = interp1d(
            x=self.bin_centers,
            y=self.std_error_in_bin,
            kind="linear",
            bounds_error=False,
            fill_value=fill_std,
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
        return stats.truncnorm.rvs(
            a=a, b=np.inf, loc=mu_sigma, scale=sigma_sigma, size=len(flux_values)
        )

    def __getstate__(self) -> Dict[str, Any]:
        """Returns the state of the model for serialization."""
        state = self.__dict__.copy()
        state.pop("_mu_sigma_interpolator", None)
        state.pop("_sigma_sigma_interpolator", None)
        state.pop("_sigma_sigma_base_interpolator", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restores the state of the model, including interpolators."""
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

    def __init__(
        self,
        # Raw data is now optional to allow for an empty instance during deserialization
        observed_phot_jy: Optional[unyt_array] = None,
        observed_phot_errors_jy: Optional[unyt_array] = None,
        asinh_b_factor: float = 5.0,
        error_type: str = "empirical",
        min_flux_error: Optional[float] = None,
        max_flux_error: Optional[float] = None,
        interpolation_flux_unit: str = "asinh",
        **kwargs: Any,
    ):
        """Initializes the model with observed photometric data in Jy.

        Args:
            observed_phot_jy (unyt_array): Observed photometric fluxes in Janskys.
            observed_phot_errors_jy (unyt_array): Observed photometric errors in Janskys.
            asinh_b_factor (float): The b factor for the asinh scaling, default is 5.0.
            error_type (str): Type of error model, either "empirical" or "theoretical".
            min_flux_error (Optional[float]): Minimum flux error to apply, default is 0.0.
            max_flux_error (Optional[float]): Maximum flux error to apply, default is np.inf.
                Units should be in asinh magnitudes.
                Currently limits are applied to returned arrays, not those used for scattering.
            interpolation_flux_unit (str): The unit for interpolation, default is "asinh".
                Can also be e.g. Jy.
            kwargs: Additional keyword arguments for the base class.

        Returns:
            None
        """
        super().__init__(**kwargs)
        self.error_type = error_type
        self.min_flux_error = min_flux_error if min_flux_error is not None else 0.0
        self.max_flux_error = max_flux_error if max_flux_error is not None else np.inf
        self.interpolation_flux_unit = interpolation_flux_unit
        self.b = None  # Initialize to None

        if observed_phot_jy is not None and observed_phot_errors_jy is not None:
            if not isinstance(observed_phot_jy, unyt_array):
                observed_phot_jy = unyt_array(observed_phot_jy, "Jy")
            if not isinstance(observed_phot_errors_jy, unyt_array):
                observed_phot_errors_jy = unyt_array(observed_phot_errors_jy, "Jy")

            valid = np.isfinite(observed_phot_jy) & np.isfinite(observed_phot_errors_jy)
            flux_jy, error_jy = observed_phot_jy[valid], observed_phot_errors_jy[valid]

            # The processed state IS saved to self.
            self.b = asinh_b_factor * np.median(error_jy)

            mag_asinh = f_jy_to_asinh(flux_jy, self.b)
            mag_err_asinh = f_jy_err_to_asinh(flux_jy, error_jy, self.b)

            if self.interpolation_flux_unit == "asinh":
                self._compute_bins_from_data(fluxes=mag_asinh, errors=mag_err_asinh)
            else:  # interpolation_flux_unit is a physical unit
                self._compute_bins_from_data(
                    fluxes=flux_jy.to_value(self.interpolation_flux_unit),
                    errors=error_jy.to_value(self.interpolation_flux_unit),
                )
            self._create_interpolators()

    def apply_noise(
        self, flux: unyt_array, true_flux_units: Optional[str] = None, **kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Applies noise to a flux, which is assumed to be in Jy or convertible."""
        if true_flux_units == "AB":
            true_flux_jy = self.ab_to_jy(flux)
            warnings.warn(
                "Using asinh model with AB input will not benefit from asinh scaling of neg fluxes."
            )
        elif true_flux_units is not None:
            if isinstance(flux, unyt_array):
                assert true_flux_units == flux.units, (
                    "If true_flux_units is specified, "
                    "flux must be a unyt_array with the same units."
                )
                true_flux_jy = flux.to("Jy")
            else:
                if isinstance(true_flux_units, str):
                    true_flux_units = Unit(true_flux_units)

                true_flux_jy = (flux * true_flux_units).to("Jy")
        else:
            true_flux_jy = flux  # Assumes input is already a unyt_array in Jy

        if self.interpolation_flux_unit == "asinh":
            true_mag_asinh = f_jy_to_asinh(true_flux_jy, self.b)
            sampled_err_asinh = self.sample_uncertainty(true_mag_asinh)
            noise = np.random.normal(loc=0.0, scale=sampled_err_asinh)
            noisy_mag_asinh = true_mag_asinh + noise
            final_err = (
                sampled_err_asinh
                if self.error_type == "empirical"
                else self.sample_uncertainty(noisy_mag_asinh)
            )
        else:  # Assumes interpolation is in physical flux units
            sampled_err_phys = self.sample_uncertainty(
                true_flux_jy.to_value(self.interpolation_flux_unit)
            )
            sampled_err_jy = unyt_array(sampled_err_phys, self.interpolation_flux_unit).to("Jy")
            noise = np.random.normal(loc=0.0, scale=sampled_err_jy.to_value())
            noisy_flux_jy = true_flux_jy + noise * Jy
            noisy_mag_asinh = f_jy_to_asinh(noisy_flux_jy, self.b)

            if self.error_type == "empirical":
                err_phys = self.sample_uncertainty(
                    noisy_flux_jy.to_value(self.interpolation_flux_unit)
                )
                final_err_jy = unyt_array(err_phys, self.interpolation_flux_unit).to("Jy")
            else:
                final_err_jy = sampled_err_jy
            final_err = f_jy_err_to_asinh(noisy_flux_jy, final_err_jy, self.b)

        final_err = np.clip(final_err, self.min_flux_error, self.max_flux_error)
        return (noisy_mag_asinh, final_err) if self.return_noise else noisy_mag_asinh

    def apply_scalings(
        self, flux: unyt_array, error: unyt_array, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Converts flux and error from Jy to asinh magnitudes."""
        if len(kwargs) > 0:
            print(
                f"WARNING {kwargs} arguments will have no effect with this model. "
                "Input must be in Jy."
            )

        if not isinstance(flux, unyt_array):
            flux = unyt_array(flux, "Jy")
        if not isinstance(error, unyt_array):
            error = unyt_array(error, "Jy")

        mag = f_jy_to_asinh(flux, self.b)
        mag_err = f_jy_err_to_asinh(flux, error, self.b)

        # Raise error if any mag_errs become nan when flux or mag is not NAN
        """if np.any(np.isnan(mag_err) & ~np.isnan(flux)) or np.any(
            np.isnan(mag_err) & ~np.isnan(mag)
        ):
            idx = np.where(np.isnan(mag_err) & ~np.isnan(flux))[0]
            raise ValueError(
                f"Conversion resulted in NaN magnitude errors for non-NaN fluxes at indices {idx}."
            )
        """
        # Clip errors to the specified limits
        mag_err = np.clip(mag_err, self.min_flux_error, self.max_flux_error)

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
        attrs["return_noise"] = self.return_noise
        attrs["min_flux_error"] = self.min_flux_error
        attrs["max_flux_error"] = self.max_flux_error
        attrs["interpolation_flux_unit"] = self.interpolation_flux_unit
        attrs["extrapolate"] = self.extrapolate

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
        instance.return_noise = attrs["return_noise"]
        instance.min_flux_error = attrs["min_flux_error"]
        instance.max_flux_error = attrs["max_flux_error"]
        instance.interpolation_flux_unit = attrs["interpolation_flux_unit"]
        instance.extrapolate = attrs.get("extrapolate", False)

        instance._log_bins = attrs.get("log_bins", True)
        instance._num_bins = attrs.get("num_bins", 20)
        instance._min_samples_per_bin = attrs.get("min_samples_per_bin", 10)

        instance._create_interpolators()

        return instance


class GeneralEmpiricalUncertaintyModel(EmpiricalUncertaintyModel):
    """General empirical uncertainty model for photometric fluxes."""

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
        sigma_clip: float = None,
        min_flux_error: float = 0.0,
        max_flux_error: float = np.inf,
        error_type: str = "empirical",
        upper_limits: bool = False,
        treat_as_upper_limits_below: Optional[float] = None,
        upper_limit_flux_behaviour: Union[str, float] = "scatter_limit",
        upper_limit_flux_err_behaviour: str = "flux",
        **kwargs: Any,
    ):
        """Initializes the model with observed fluxes and errors.

        Args:
            observed_fluxes (np.ndarray): Observed fluxes in the specified unit.
            observed_errors (np.ndarray): Observed errors in the same unit as fluxes.
            flux_unit (str): The unit of the observed fluxes, default is "AB".
            interpolation_flux_unit (Optional[str]): The unit for interpolation, default is None.
                If None, defaults to flux_unit.
            already_binned (bool): If True, assumes the data is already binned.
            bin_median_errors (Optional[np.ndarray]): Median errors for each bin,
                required if already binned is True.
            bin_std_errors (Optional[np.ndarray]): Standard deviation of errors for each bin,
                required if already_binned is True.
            flux_bins (Optional[np.ndarray]): Precomputed bins for fluxes, required if
                already_binned is False.
            min_flux_for_binning (Optional[float]): Minimum flux value to consider for binning
                (in the same unit as fluxes). If None, no minimum is applied.
            sigma_clip (Optional[float]): Sigma clipping threshold for outlier removal,
                default is None (no clipping).
            min_flux_error (float): Minimum flux error to apply, default is 0.0
                Should be in the same unit as fluxes.
            max_flux_error (float): Maximum flux error to apply, default is np.inf.
                Should be in the same unit as fluxes.
            error_type (str): Type of error model, either "empirical" or "theoretical".
            upper_limits (bool): If True, handles upper limits in the data.
            treat_as_upper_limits_below (Optional[float]): If specified, fluxes below this
                value are treated as upper limits. If None, no upper limit treatment is applied.
            upper_limit_flux_behaviour (Union[str, float]): Behaviour for upper limit fluxes.
                Can be "scatter_limit" to use the scatter limit, or a fixed value.
            upper_limit_flux_err_behaviour (str): Behaviour for upper limit flux errors.
                Can be "flux" to use the flux error, or "scatter_limit" to use the scatter limit.
            kwargs: Additional keyword arguments for the base class.

        Returns:
            None
        """
        # 1. Initialize parent and instance attributes
        super().__init__(**kwargs)
        self.flux_unit = flux_unit
        self.interpolation_flux_unit = (
            interpolation_flux_unit if interpolation_flux_unit else flux_unit
        )
        self.sigma_clip = sigma_clip
        self.min_flux_error = min_flux_error
        self.max_flux_error = max_flux_error
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

        valid_mask = (
            np.isfinite(flux_to_process) & np.isfinite(error_to_process) & (error_to_process > 0)
        )
        if min_flux_for_binning is not None:
            valid_mask &= flux_to_process > min_flux_for_binning

        self._compute_bins_from_data(
            fluxes=flux_to_process[valid_mask],
            errors=error_to_process[valid_mask],
            precomputed_bins=flux_bins,
        )

        if self.upper_limits:
            self._setup_upper_limit_interpolator(
                flux_to_process[valid_mask], error_to_process[valid_mask]
            )

        self._create_interpolators()

    def _convert_units(
        self, fluxes: np.ndarray, errors: np.ndarray, fluxes_unit=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Helper to handle unit conversion for binning data."""
        if fluxes_unit is None:
            fluxes_unit = self.flux_unit

        if (self.interpolation_flux_unit == fluxes_unit) or (
            isinstance(self.interpolation_flux_unit, unyt_array)
            and isinstance(fluxes_unit, unyt_array)
            and (self.interpolation_flux_unit.dimensions == fluxes_unit.dimensions)
        ):
            if self.interpolation_flux_unit == fluxes_unit:
                conversion = 1.0
            else:
                conversion = (self.interpolation_flux_unit / fluxes_unit).simplify()

            return fluxes * conversion, errors * conversion
        if fluxes_unit == "AB":  # AB to physical flux
            flux_jy = self.ab_to_jy(fluxes)
            error_jy = self.ab_err_to_jy(errors, flux_jy)
            return flux_jy.to_value(self.interpolation_flux_unit), error_jy.to_value(
                self.interpolation_flux_unit
            )
        else:  # Physical flux to AB
            if isinstance(fluxes, unyt_array):
                fluxes = fluxes.to_value(fluxes_unit)
            if isinstance(errors, unyt_array):
                errors = errors.to_value(fluxes_unit)
            flux_with_units = fluxes * fluxes_unit
            error_with_units = errors * fluxes_unit
            return self.jy_to_ab(flux_with_units), self.jy_err_to_ab(
                error_with_units, flux_with_units
            )

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

        with np.errstate(divide="ignore", invalid="ignore"):
            snr = (flux_jy / error_jy).value

        valid = np.isfinite(snr) & (snr > 0) & np.isfinite(flux_jy.value) & (flux_jy.value > 0)
        if np.sum(valid) < 2:
            return

        order = np.argsort(snr[valid])
        self._snr_x_data = np.log10(snr[valid][order])
        self._snr_y_data = np.log10(flux_jy.value[valid][order])

        self.log_snr_interpolator = interp1d(
            self._snr_x_data,
            self._snr_y_data,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        ul_flux_jy = 10 ** self.log_snr_interpolator(np.log10(self.treat_as_upper_limits_below))

        if self.interpolation_flux_unit == "AB":
            self.upper_limit_value = self.jy_to_ab(ul_flux_jy * Jy)
        else:
            self.upper_limit_value = (ul_flux_jy * Jy).to_value(self.interpolation_flux_unit)

    def apply_noise(
        self, flux: np.ndarray, true_flux_units: str = None, out_units=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Applies configured noise and upper limit rules to true flux values."""
        # 1. Convert input flux to the model's internal interpolation units
        flux_internal, _ = self._convert_units(flux, np.zeros_like(flux), true_flux_units)

        # 2. Sample the uncertainty based on the true (un-scattered) flux
        sampled_sigma_internal = self.sample_uncertainty(flux_internal)

        # Initialize noisy flux and final sigma with the true values
        noisy_flux_internal = np.copy(flux_internal)
        final_sigma_internal = np.copy(sampled_sigma_internal)

        # 3. FIX: Perform a pre-emptive SNR check BEFORE adding noise
        if self.upper_limits:
            # Identify sources that are already below the SNR threshold
            initial_limit_mask = self._get_snr_mask(flux_internal, sampled_sigma_internal)
            # We will not apply noise to these sources
            apply_noise_mask = ~initial_limit_mask
        else:
            # If upper limits are off, apply noise to everything
            initial_limit_mask = np.zeros_like(flux_internal, dtype=bool)
            apply_noise_mask = ~initial_limit_mask

        # 4. Add noise ONLY to the sources that passed the initial SNR check
        if np.any(apply_noise_mask):
            if self.sigma_clip is not None:
                noise = stats.truncnorm.rvs(
                    -self.sigma_clip, self.sigma_clip, 0, sampled_sigma_internal[apply_noise_mask]
                )
            else:
                noise = np.random.normal(loc=0.0, scale=sampled_sigma_internal[apply_noise_mask])
            noisy_flux_internal[apply_noise_mask] += noise

        # 5. Re-evaluate errors if 'observed' error type is used
        if self.error_type == "observed":
            final_sigma_internal = self.sample_uncertainty(noisy_flux_internal)

        # 6. Now, determine the final upper limit mask
        if self.upper_limits and self.upper_limit_value is not None:
            # Check the now-noisy fluxes to catch any that scattered into the low-SNR regime
            post_noise_limit_mask = self._get_snr_mask(noisy_flux_internal, final_sigma_internal)
            # The final mask includes both the pre-emptively caught sources and the newly scattered
            final_limit_mask = initial_limit_mask | post_noise_limit_mask

            if np.any(final_limit_mask):
                noisy_flux_internal = self._apply_flux_behaviour(
                    noisy_flux_internal, final_limit_mask, scatter=True
                )
                final_sigma_internal = self._apply_error_behaviour(
                    final_sigma_internal, final_limit_mask
                )

        # 7. Convert results back to the original input units
        out_flux, out_sigma = self._convert_units_inverse(
            noisy_flux_internal, final_sigma_internal, out_units
        )

        # 8. Apply final min/max clipping
        out_sigma = np.clip(out_sigma, self.min_flux_error, self.max_flux_error)

        return (out_flux, out_sigma) if self.return_noise else out_flux

    def _get_snr_mask(self, fluxes, errors):
        """Calculates a boolean mask for sources below the SNR threshold."""
        if self.interpolation_flux_unit == "AB":
            flux_jy = self.ab_to_jy(fluxes)
            error_jy = self.ab_err_to_jy(errors, flux_jy)
        else:
            flux_jy = (fluxes * self.interpolation_flux_unit).to(Jy)
            error_jy = (errors * self.interpolation_flux_unit).to(Jy)

        with np.errstate(divide="ignore", invalid="ignore"):
            snr = (flux_jy / error_jy).value

        return ~np.isfinite(snr) | (snr < self.treat_as_upper_limits_below)

    def _apply_flux_behaviour(
        self, fluxes: np.ndarray, mask: np.ndarray, scatter: bool
    ) -> np.ndarray:
        """Applies the configured flux rule to masked elements."""
        if self.upper_limit_flux_behaviour == "scatter_limit":
            # Only add random scatter if explicitly told to
            if scatter:
                scatter_std = self._sigma_sigma_interpolator(self.upper_limit_value)
                samples = stats.truncnorm.rvs(-3, 3, loc=0, scale=scatter_std, size=np.sum(mask))
                fluxes[mask] = self.upper_limit_value + samples
            else:
                # For apply_scalings, 'scatter_limit' is treated deterministically
                fluxes[mask] = self.upper_limit_value
        elif self.upper_limit_flux_behaviour == "upper_limit":
            fluxes[mask] = self.upper_limit_value
        else:  # Assumes a numeric value
            fluxes[mask] = float(self.upper_limit_flux_behaviour)
        return fluxes
        """Applies the configured flux rule to masked elements."""
        if self.upper_limit_flux_behaviour == "scatter_limit":
            scatter_std = self._sigma_sigma_interpolator(self.upper_limit_value)
            samples = stats.truncnorm.rvs(-3, 3, loc=0, scale=scatter_std, size=np.sum(mask))
            fluxes[mask] = self.upper_limit_value + samples
        elif self.upper_limit_flux_behaviour == "upper_limit":
            fluxes[mask] = self.upper_limit_value
        else:  # Assumes a numeric value
            fluxes[mask] = float(self.upper_limit_flux_behaviour)
        return fluxes

    def _apply_error_behaviour(self, errors: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Applies the configured error rule to masked elements."""
        behaviour = self.upper_limit_flux_err_behaviour

        if behaviour == "flux":
            errors[mask] = self._mu_sigma_interpolator(self.upper_limit_value)
        elif behaviour == "upper_limit":
            errors[mask] = self.upper_limit_value
        elif behaviour == "max":
            errors[mask] = self.max_flux_error
        elif behaviour.startswith("sig_"):
            sig_val = float(behaviour.split("_")[1])

            # FIX: If the model's internal units are AB magnitudes, the relationship
            # between SNR and mag_err is direct and independent of flux.
            if self.interpolation_flux_unit == "AB":
                # Use the direct formula: mag_err = (2.5 / ln(10)) / SNR
                error_val = (2.5 / np.log(10)) / sig_val
                errors[mask] = error_val
            else:
                # For physical flux units, we must still use the interpolator to find
                # a typical flux for that SNR, then find the error at that flux.
                if self.log_snr_interpolator is None:
                    raise ValueError(
                        "SNR interpolator is not available for 'sig_X' "
                        "error behaviour in flux space."
                    )

                flux_at_snr_jy = 10 ** self.log_snr_interpolator(np.log10(sig_val))
                flux_at_snr_internal = (flux_at_snr_jy * Jy).to_value(self.interpolation_flux_unit)
                errors[mask] = self._mu_sigma_interpolator(flux_at_snr_internal)

        return errors

    def _convert_units_inverse(
        self, fluxes: np.ndarray, errors: np.ndarray, out_unit=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Helper to convert from internal units back to the model's primary flux_unit."""
        if out_unit is None:
            out_unit = self.flux_unit

        if (self.interpolation_flux_unit == out_unit) or (
            isinstance(self.interpolation_flux_unit, unyt_array)
            and isinstance(out_unit, unyt_array)
            and (self.interpolation_flux_unit.dimensions == out_unit.dimensions)
        ):
            if self.interpolation_flux_unit == out_unit:
                return fluxes, errors
            else:
                conversion = (out_unit / self.interpolation_flux_unit).simplify()
                return fluxes * conversion, errors * conversion

        # This is the reverse of _convert_units
        if self.interpolation_flux_unit == "AB":  # AB to physical flux
            flux_jy = self.ab_to_jy(fluxes)
            error_jy = self.ab_err_to_jy(errors, flux_jy)
            return flux_jy.to_value(out_unit), error_jy.to_value(out_unit)
        else:  # Physical flux to AB
            flux_with_units = fluxes * self.interpolation_flux_unit
            error_with_units = errors * self.interpolation_flux_unit
            return self.jy_to_ab(flux_with_units), self.jy_err_to_ab(
                error_with_units, flux_with_units
            )

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
        attrs["sigma_clip"] = self.sigma_clip if self.sigma_clip is not None else "None"
        attrs["min_flux_error"] = self.min_flux_error if self.min_flux_error is not None else "None"
        attrs["max_flux_error"] = self.max_flux_error if self.max_flux_error is not None else "None"
        attrs["error_type"] = self.error_type
        attrs["upper_limits"] = self.upper_limits
        attrs["treat_as_upper_limits_below"] = (
            self.treat_as_upper_limits_below
            if self.treat_as_upper_limits_below is not None
            else "None"
        )
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
            "observed_errors": None,  # Not needed for this path
            "already_binned": True,
            "bin_median_errors": hdf5_group["median_error_in_bin"][:],
            "bin_std_errors": hdf5_group["std_error_in_bin"][:],
            "flux_unit": attrs["flux_unit"],
            "interpolation_flux_unit": attrs["interpolation_flux_unit"],
            "sigma_clip": None if attrs["sigma_clip"] == "None" else attrs["sigma_clip"],
            "min_flux_error": None
            if attrs["min_flux_error"] == "None"
            else attrs["min_flux_error"],
            "max_flux_error": None
            if attrs["max_flux_error"] == "None"
            else attrs["max_flux_error"],
            "error_type": attrs["error_type"],
            "upper_limits": attrs["upper_limits"],
            "treat_as_upper_limits_below": None
            if attrs["treat_as_upper_limits_below"] == "None"
            else attrs["treat_as_upper_limits_below"],
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
                instance._snr_x_data,
                instance._snr_y_data,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )

        return instance

    def apply_scalings(
        self, flux: np.ndarray, error: np.ndarray, flux_units: str, out_units: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Applies deterministic model transformations (units, SNR cuts)."""
        # 1. Convert input flux to the model's internal interpolation units
        flux_internal, error_internal = self._convert_units(flux, error)

        # 2. Apply upper limit rule (SNR cut) without random scatter
        if self.upper_limits and self.upper_limit_value is not None:
            limit_mask = self._get_snr_mask(flux_internal, error_internal)

            if np.any(limit_mask):
                # We call the behaviour helpers with scatter=False for a deterministic result
                flux_internal = self._apply_flux_behaviour(flux_internal, limit_mask, scatter=False)
                error_internal = self._apply_error_behaviour(error_internal, limit_mask)

        # 3. Convert results to the desired output units
        if self.interpolation_flux_unit != out_units:
            flux_internal, error_internal = self._convert_units_inverse(
                flux_internal, error_internal, out_units
            )

        # 4. Clip the final errors to the allowed range
        final_error = np.clip(error_internal, self.min_flux_error, self.max_flux_error)

        return flux_internal, final_error


# =============================================================================
# SCORE-BASED DIFFUSION UNCERTAINTY MODEL
# =============================================================================


class GaussianFourierProjection(nn.Module):
    """Maps a scalar input to a sinusoidal random Fourier feature embedding.

    Used to encode the diffusion time step into a fixed-dimensional vector
    suitable for conditioning a neural network. Random frequencies are drawn
    once at construction and kept fixed (non-trainable).
    """

    def __init__(self, embed_dim: int, scale: float = 30.0):
        """Initializes the projection with random Fourier frequencies.

        Args:
            embed_dim (int): Output embedding dimension. Must be even; the output
                is formed by concatenating ``embed_dim // 2`` sin and cos features.
            scale (float): Standard deviation of the Gaussian from which random
                frequencies are drawn. Higher values encode higher-frequency
                variation. Default is 30.0.
        """
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects x into the sinusoidal embedding space.

        Args:
            x (torch.Tensor): Input tensor of shape ``(..., 1)`` (e.g. diffusion
                time steps, one per batch element).

        Returns:
            torch.Tensor: Embedding of shape ``(..., embed_dim)`` formed by
                concatenating ``sin`` and ``cos`` projections along the last axis.
        """
        x_proj = x * self.W[None, :] * 2 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class _ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear(self.act(x))


class _RobustScoreNetwork(nn.Module):
    def __init__(
        self,
        n_filters: int,
        hidden_dim: int = 256,
        n_layers: int = 5,
        time_embed_dim: int = 64,
    ):
        super().__init__()
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
        )

        input_dim = n_filters * 2 + time_embed_dim
        self.proj_in = nn.Linear(input_dim, hidden_dim)

        self.res_blocks = nn.ModuleList([_ResidualBlock(hidden_dim) for _ in range(n_layers)])

        self.proj_out = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, n_filters))

    def forward(self, x: torch.Tensor, t: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t.unsqueeze(-1))
        h = self.proj_in(torch.cat([x, m, t_emb], dim=-1))

        for block in self.res_blocks:
            h = block(h)

        return self.proj_out(h)


class ScoreBasedUncertaintyModel:
    """Score-based diffusion model that learns the conditional uncertainty distribution p(σ|m).

    Trains a neural score network on pairs of observed magnitudes and flux
    uncertainties using a variance-preserving (VP) SDE with a linear beta
    schedule. After training, draws photometric uncertainty samples by running
    the reverse-time ODE or SDE, then uses those samples to corrupt input flux
    arrays in the same way as the other noise models.

    The model operates internally in log-uncertainty space and applies
    IQR-based normalisation to both the uncertainty and magnitude features
    before training, making it robust to outliers.
    """

    def __init__(
        self,
        filter_names: List[str],
        hidden_dim: int = 256,
        n_layers: int = 5,
        time_embed_dim: int = 64,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        T: float = 1.0,
        device: Optional[str] = None,
        return_noise: bool = False,
    ):
        """Initializes the score network, EMA wrapper, and SDE hyperparameters.

        Args:
            filter_names (List[str]): Ordered list of photometric band names
                (e.g. ``["u", "g", "r", "i", "z"]``). The number of filters is
                inferred as ``len(filter_names)`` and sets the input/output
                dimensionality of the score network.
            hidden_dim (int): Width of each hidden layer in the score network. Default 256.
            n_layers (int): Number of residual blocks in the score network. Default 5.
            time_embed_dim (int): Dimensionality of the Gaussian Fourier time embedding.
                Default 64.
            beta_min (float): Minimum noise schedule coefficient β(0). Default 0.1.
            beta_max (float): Maximum noise schedule coefficient β(T). Default 20.0.
            T (float): Total diffusion time horizon. Default 1.0.
            device (Optional[str]): Torch device string (e.g. ``"cuda"``, ``"cpu"``).
                Auto-detected from CUDA availability when ``None``.
            return_noise (bool): If ``True``, ``apply_noise`` returns a tuple
                ``(noisy_flux, sigma)`` instead of only the noisy flux. Default ``False``.
        """
        self.filter_names = list(filter_names)
        n_filters = len(self.filter_names)
        self.n_filters = n_filters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.time_embed_dim = time_embed_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
        self.return_noise = return_noise

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.score_net = _RobustScoreNetwork(n_filters, hidden_dim, n_layers, time_embed_dim).to(
            self.device
        )

        if self.device == "cuda":
            self.score_net = torch.compile(self.score_net, mode="reduce-overhead")

        self.ema_net = AveragedModel(self.score_net, multi_avg_fn=get_ema_multi_avg_fn(0.999))

        self._ln_sigma_median: Optional[torch.Tensor] = None
        self._ln_sigma_iqr: Optional[torch.Tensor] = None
        self._mag_median: Optional[torch.Tensor] = None
        self._mag_iqr: Optional[torch.Tensor] = None
        self._mag_max: Optional[torch.Tensor] = None
        self._mag_min: Optional[torch.Tensor] = None
        self._is_trained: bool = False

    def _beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + (t / self.T) * (self.beta_max - self.beta_min)

    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min * t + (t**2 / (2.0 * self.T)) * (self.beta_max - self.beta_min)

    def _marginal_params(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        A = self._alpha(t)
        s = torch.exp(-0.5 * A)
        sigma = torch.sqrt(torch.clamp(1.0 - torch.exp(-A), min=1e-8))
        return s, sigma

    def _normalise_ln_sigma(self, ln_sigma: torch.Tensor) -> torch.Tensor:
        return (ln_sigma - self._ln_sigma_median) / (self._ln_sigma_iqr + 1e-8)

    def _denormalise_ln_sigma(self, z: torch.Tensor) -> torch.Tensor:
        return z * self._ln_sigma_iqr + self._ln_sigma_median

    def _normalise_mag(self, mag: torch.Tensor) -> torch.Tensor:
        return (mag - self._mag_median) / (self._mag_iqr + 1e-8)

    def fit(
        self,
        magnitudes: np.ndarray,
        flux_uncertainties: np.ndarray,
        n_epochs: int = 1000,
        batch_size: int = 1024,
        learning_rate: float = 3e-4,
        t_eps: float = 1e-2,
        valid_fraction: float = 0.1,
        patience: int = 20,
        min_epochs: int = 200,
        min_delta: float = 1e-3,
        val_freq: int = 10,
        n_val_t: int = 20,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Trains the score network on observed magnitudes and flux uncertainties.

        Rows with non-positive or non-finite uncertainties are dropped before
        training. Normalisation statistics (median and IQR) are computed from
        the training set and stored on the instance for use during sampling.
        An EMA of the score network weights is maintained throughout and used
        for validation and sampling.

        Args:
            magnitudes (np.ndarray): Observed AB magnitudes, shape ``(N, n_filters)``.
            flux_uncertainties (np.ndarray): Observed flux uncertainties (σ) in
                Janskys, shape ``(N, n_filters)``. Must be positive and finite.
            n_epochs (int): Maximum number of training epochs. Default 1000.
            batch_size (int): Number of samples per gradient step. Default 1024.
            learning_rate (float): Initial Adam learning rate. Default 3e-4.
            t_eps (float): Minimum diffusion time to avoid numerical instability
                near t=0. Default 1e-2.
            valid_fraction (float): Fraction of data held out for validation.
                Default 0.1.
            patience (int): Number of validation evaluations without sufficient
                improvement before early stopping triggers. Default 20.
            min_epochs (int): Minimum training epochs before early stopping is
                active; allows EMA to warm up. Default 200.
            min_delta (float): Minimum relative validation-loss improvement
                required to reset the patience counter. Default 1e-3.
            val_freq (int): Epoch interval between validation evaluations.
                Default 10.
            n_val_t (int): Number of evenly spaced diffusion times used for
                deterministic validation loss evaluation. Default 20.
            verbose (bool): Print epoch-level progress. Default ``True``.

        Returns:
            Dict[str, List[float]]: Training history with keys ``"train_loss"``
                (every epoch) and ``"val_loss"`` (every ``val_freq`` epochs).

        Raises:
            ValueError: If ``magnitudes`` has the wrong shape or no valid samples
                remain after filtering.
        """
        if magnitudes.ndim != 2 or magnitudes.shape[1] != self.n_filters:
            raise ValueError(
                f"magnitudes must be (N, {self.n_filters}) for filters {self.filter_names}"
            )

        valid_mask = np.all((flux_uncertainties > 0) & np.isfinite(magnitudes), axis=1)
        if not np.any(valid_mask):
            raise ValueError("No valid training samples after filtering.")

        magnitudes = magnitudes[valid_mask]
        ln_sigma = np.log(flux_uncertainties[valid_mask])

        ln_sigma_median = np.median(ln_sigma, axis=0).astype(np.float32)
        ln_sigma_iqr = (
            np.percentile(ln_sigma, 75, axis=0) - np.percentile(ln_sigma, 25, axis=0)
        ).astype(np.float32)
        mag_median = np.median(magnitudes, axis=0).astype(np.float32)
        mag_iqr = (
            np.percentile(magnitudes, 75, axis=0) - np.percentile(magnitudes, 25, axis=0)
        ).astype(np.float32)

        mag_max = np.percentile(magnitudes, 99.9, axis=0).astype(np.float32)
        mag_min = np.percentile(magnitudes, 0.1, axis=0).astype(np.float32)

        self._ln_sigma_median = torch.tensor(ln_sigma_median, device=self.device)
        self._ln_sigma_iqr = torch.tensor(ln_sigma_iqr, device=self.device)
        self._mag_median = torch.tensor(mag_median, device=self.device)
        self._mag_iqr = torch.tensor(mag_iqr, device=self.device)
        self._mag_max = torch.tensor(mag_max, device=self.device)
        self._mag_min = torch.tensor(mag_min, device=self.device)

        ln_sigma_norm = (ln_sigma - ln_sigma_median) / (ln_sigma_iqr + 1e-8)
        mag_norm = (magnitudes - mag_median) / (mag_iqr + 1e-8)

        N = len(ln_sigma_norm)
        n_val = max(1, int(N * valid_fraction))
        idx = np.random.permutation(N)
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        x_train = torch.tensor(ln_sigma_norm[train_idx], dtype=torch.float32, device=self.device)
        m_train = torch.tensor(mag_norm[train_idx], dtype=torch.float32, device=self.device)
        x_val = torch.tensor(ln_sigma_norm[val_idx], dtype=torch.float32, device=self.device)
        m_val = torch.tensor(mag_norm[val_idx], dtype=torch.float32, device=self.device)

        # Pre-generate fixed validation noise. The validation loss becomes a deterministic
        # function of model parameters: same (t, eps) pairs are reused every eval step.
        # Using a stratified t grid (linspace) eliminates the variance from t sampling,
        # which is the dominant source of validation noise in score-based diffusion training.
        val_t_grid = torch.linspace(t_eps, self.T, n_val_t, device=self.device)
        val_eps_fixed = torch.randn(n_val_t, len(x_val), self.n_filters, device=self.device)

        optimizer = torch.optim.Adam(self.score_net.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=10, min_lr=1e-5
        )

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        best_state: Dict[str, torch.Tensor] = {}
        patience_counter = 0

        for epoch in range(n_epochs):
            self.score_net.train()
            perm = torch.randperm(len(x_train), device=self.device)
            x_ep, m_ep = x_train[perm], m_train[perm]

            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, len(x_ep), batch_size):
                x0 = x_ep[i : i + batch_size]
                m = m_ep[i : i + batch_size]
                B = len(x0)

                t = torch.rand(B, device=self.device) * (self.T - t_eps) + t_eps
                s, sigma = self._marginal_params(t)
                eps = torch.randn_like(x0)
                x_t = s.unsqueeze(-1) * x0 + sigma.unsqueeze(-1) * eps

                score = self.score_net(x_t, t, m)
                target = -eps / sigma.unsqueeze(-1)
                loss = (sigma.unsqueeze(-1) ** 2 * (score - target) ** 2).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.score_net.parameters(), max_norm=1.0)
                optimizer.step()
                self.ema_net.update_parameters(self.score_net)

                epoch_loss += loss.item()
                n_batches += 1

            train_loss = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(train_loss)

            if epoch % val_freq == 0:
                self.ema_net.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    n_val_terms = 0
                    # Iterate over the fixed t grid; for each t, evaluate the whole
                    # validation set in mini-batches using the corresponding fixed eps.
                    for t_idx in range(n_val_t):
                        t_val = val_t_grid[t_idx]
                        eps_full = val_eps_fixed[t_idx]
                        s_v_scalar, sigma_v_scalar = self._marginal_params(t_val.unsqueeze(0))
                        s_v_scalar = s_v_scalar.item()
                        sigma_v_scalar = sigma_v_scalar.item()

                        for i in range(0, len(x_val), batch_size):
                            x_v_batch = x_val[i : i + batch_size]
                            m_v_batch = m_val[i : i + batch_size]
                            eps_v = eps_full[i : i + batch_size]
                            B_val = len(x_v_batch)

                            t_v = t_val.expand(B_val)
                            x_t_v = s_v_scalar * x_v_batch + sigma_v_scalar * eps_v

                            score_v = self.ema_net(x_t_v, t_v, m_v_batch)
                            target_v = -eps_v / sigma_v_scalar
                            batch_loss = (
                                (sigma_v_scalar**2 * (score_v - target_v) ** 2).mean().item()
                            )

                            val_loss += batch_loss
                            n_val_terms += 1

                    val_loss /= n_val_terms

                history["val_loss"].append(val_loss)
                scheduler.step(val_loss)

                if verbose:
                    print(
                        f"Epoch {epoch:5d}/{n_epochs}  train={train_loss:.4f}  val={val_loss:.4f}"
                    )

                # Early stopping: only active after min_epochs to allow EMA to warm up
                # and avoid premature stopping during the initial fluctuating phase.
                if epoch < min_epochs:
                    # Still track best state during warmup so we have something sensible
                    # if training is interrupted, but don't count toward patience.
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_state = {
                            k: v.clone() for k, v in self.ema_net.module.state_dict().items()
                        }
                    continue

                # Relative improvement threshold: require val_loss to drop by at least
                # min_delta (fractionally) to count as genuine progress. Avoids resetting
                # patience on noise-level fluctuations.
                if val_loss < best_val_loss * (1.0 - min_delta):
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in self.ema_net.module.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(
                                f"Early stopping at epoch {epoch} "
                                f"({patience} evals without >{min_delta:.1%} improvement)."
                            )
                        break

        if best_state:
            self.ema_net.module.load_state_dict(best_state)
            self.score_net.load_state_dict(best_state)
        self._is_trained = True
        return history

    @torch.no_grad()
    def sample_uncertainty(
        self,
        magnitudes: np.ndarray,
        n_samples: int = 1,
        n_steps: int = 50,
        t_eps: float = 1e-2,
        method: str = "ode",
    ) -> np.ndarray:
        """Samples flux uncertainties conditioned on observed magnitudes.

        Runs the reverse-time diffusion process (ODE or SDE) from t=T to t=t_eps,
        then exponentiates the resulting normalised log-uncertainty to recover
        physical flux uncertainties in Janskys.

        Args:
            magnitudes (np.ndarray): Observed AB magnitudes. Shape ``(n_filters,)``
                for a single object or ``(N, n_filters)`` for a batch.
            n_samples (int): Number of independent uncertainty draws per object.
                Default 1.
            n_steps (int): Number of discrete integration steps. Default 50.
            t_eps (float): Minimum diffusion time; integration runs from T down
                to this value. Default 1e-2.
            method (str): Reverse-time solver — ``"ode"`` (probability-flow ODE,
                deterministic) or ``"sde"`` (Langevin SDE, stochastic).
                Default ``"ode"``.

        Returns:
            np.ndarray: Sampled flux uncertainties (σ) in Janskys. Shape is
                ``(N, n_filters)`` when ``n_samples=1``, ``(N, n_samples, n_filters)``
                otherwise. A leading batch dimension is stripped when the input
                was a single-object array.

        Raises:
            RuntimeError: If called before the model has been trained.
            ValueError: If ``magnitudes`` has the wrong shape, contains non-finite
                values, or ``method`` is not recognised.
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained before sampling. Call fit() first.")

        scalar_input = magnitudes.ndim == 1
        if scalar_input:
            magnitudes = magnitudes[np.newaxis]

        # Check input is finite and has correct shape
        if magnitudes.shape[1] != self.n_filters:
            raise ValueError(
                f"Input magnitudes must have shape (N, {self.n_filters}) "
                f"for filters {self.filter_names}"
            )
        if not np.all(np.isfinite(magnitudes)):
            raise ValueError("Input magnitudes must be finite.")

        N = len(magnitudes)

        # Clip to training range to avoid OOD extrapolation for very faint
        # (large mag) or very bright (small mag) simulated sources.
        mag_max_np = self._mag_max.cpu().numpy()
        mag_min_np = self._mag_min.cpu().numpy()
        magnitudes = np.clip(magnitudes, a_min=mag_min_np, a_max=mag_max_np)

        mag_t = torch.tensor(magnitudes, dtype=torch.float32, device=self.device)
        mag_norm = self._normalise_mag(mag_t).repeat_interleave(n_samples, dim=0)

        x = torch.randn(N * n_samples, self.n_filters, device=self.device)
        self.ema_net.eval()

        dt = (self.T - t_eps) / n_steps
        times = torch.linspace(self.T, t_eps, n_steps + 1, device=self.device)

        with torch.amp.autocast("cuda", enabled=(self.device == "cuda"), dtype=torch.bfloat16):
            for i in range(n_steps):
                t_val = times[i].item()
                t_tensor = torch.full(
                    (N * n_samples,), t_val, device=self.device, dtype=torch.float32
                )
                beta_t = self._beta(t_tensor).unsqueeze(-1)
                score = self.ema_net(x, t_tensor, mag_norm)

                if method == "ode":
                    dx_dt = -0.5 * beta_t * (x + score)
                    x = x - dx_dt * dt
                elif method == "sde":
                    drift = (beta_t / 2) * x + beta_t * score
                    z = torch.randn_like(x)
                    x = x + drift * dt + torch.sqrt(beta_t * dt) * z
                else:
                    raise ValueError(f"Unknown sampling method: {method}")

        ln_sigma = self._denormalise_ln_sigma(x)
        # Clamp to prevent float32 overflow: exp(88) ≈ 1.6e38 ≈ float32 max.
        # Physical sigma never exceeds ~1e5 Jy; exp(12) ≈ 1.6e5 Jy is a safe ceiling.
        ln_sigma = torch.clamp(ln_sigma, min=-80.0, max=12.0)
        sigma_np = torch.exp(ln_sigma).cpu().numpy().reshape(N, n_samples, self.n_filters)

        if n_samples == 1:
            sigma_np = sigma_np[:, 0, :]

        return sigma_np[0] if scalar_input else sigma_np

    def apply_noise(
        self,
        flux: np.ndarray,
        n_steps: int = 50,
        true_flux_units: str = None,
        method: str = "ode",
        **kwargs: Any,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Applies diffusion-model-sampled Gaussian noise to the input flux.

        Converts flux to AB magnitudes, draws one uncertainty sample per object
        via the reverse diffusion process, then adds zero-mean Gaussian noise
        with that uncertainty to the original flux.

        Args:
            flux (np.ndarray or unyt_array): Input flux values. If a plain
                numpy array, ``true_flux_units`` must be provided.
            n_steps (int): Number of reverse-diffusion integration steps. Default 50.
            true_flux_units (str): Units of ``flux`` (e.g. ``"Jy"``, ``"uJy"``).
                Required when ``flux`` is not a unyt_array. Default ``None``.
            method (str): Reverse-time solver — ``"ode"`` or ``"sde"``. Default ``"ode"``.
            **kwargs: Absorbed for interface compatibility; not used.

        Returns:
            Union[unyt_array, Tuple[unyt_array, unyt_array]]: Noisy flux in
                Janskys, or ``(noisy_flux_jy, sigma_jy)`` if the model was
                constructed with ``return_noise=True``.

        Raises:
            ValueError: If ``flux`` is a plain array and ``true_flux_units``
                is not provided.
        """
        if isinstance(flux, unyt_array):
            flux_jy = flux.to("Jy").value
        elif true_flux_units is not None:
            flux_jy = unyt_array(flux, units=true_flux_units).to("Jy").value
        else:
            raise ValueError("true_flux_units must be provided when flux is not a unyt_array.")
        safe_jy = np.where(flux_jy > 0, flux_jy, np.finfo(np.float32).tiny)
        magnitudes = (-2.5 * np.log10(safe_jy / 3631.0)).astype(np.float32)
        sigma = self.sample_uncertainty(magnitudes, n_samples=1, n_steps=n_steps, method=method)
        noise = np.random.normal(0.0, sigma)
        noisy_flux = unyt_array(flux_jy + noise, units="Jy")
        sigma_jy = unyt_array(sigma, units="Jy")

        if self.return_noise:
            return noisy_flux, sigma_jy
        return noisy_flux

    def serialize_to_hdf5(self, hdf5_group: h5py.Group) -> None:
        """Serializes the model's configuration and trained weights to an HDF5 group.

        Hyperparameters (including the ordered filter name list) are stored as
        HDF5 attributes. When the model has been trained, normalisation
        statistics and the EMA score-network weights are also written.
        Compiled-model wrappers (``torch.compile``) are unwrapped before weight
        extraction.

        Args:
            hdf5_group (h5py.Group): Open, writable HDF5 group to write into.
        """
        attrs = hdf5_group.attrs
        attrs["__class__"] = self.__class__.__name__
        attrs["filter_names"] = json.dumps(self.filter_names)
        attrs["hidden_dim"] = self.hidden_dim
        attrs["n_layers"] = self.n_layers
        attrs["time_embed_dim"] = self.time_embed_dim
        attrs["beta_min"] = self.beta_min
        attrs["beta_max"] = self.beta_max
        attrs["T"] = self.T
        attrs["return_noise"] = self.return_noise
        attrs["is_trained"] = self._is_trained

        if not self._is_trained:
            return

        hdf5_group.create_dataset("ln_sigma_median", data=self._ln_sigma_median.cpu().numpy())
        hdf5_group.create_dataset("ln_sigma_iqr", data=self._ln_sigma_iqr.cpu().numpy())
        hdf5_group.create_dataset("mag_median", data=self._mag_median.cpu().numpy())
        hdf5_group.create_dataset("mag_iqr", data=self._mag_iqr.cpu().numpy())
        hdf5_group.create_dataset("mag_max", data=self._mag_max.cpu().numpy())
        hdf5_group.create_dataset("mag_min", data=self._mag_min.cpu().numpy())

        weights_group = hdf5_group.create_group("score_network_weights")

        underlying = self.ema_net.module
        if hasattr(underlying, "_orig_mod"):
            underlying = underlying._orig_mod

        for key, tensor in underlying.state_dict().items():
            safe_key = key.replace(".", "__")
            weights_group.create_dataset(safe_key, data=tensor.cpu().numpy())

    @classmethod
    def _from_hdf5_group(cls, hdf5_group: h5py.Group) -> "ScoreBasedUncertaintyModel":
        """Loads a ScoreBasedUncertaintyModel from an HDF5 group.

        Reconstructs the model from the hyperparameters stored as HDF5 attributes,
        then restores the normalisation statistics and score-network weights if
        the model was trained before serialisation.

        Args:
            hdf5_group (h5py.Group): Open HDF5 group previously written by
                ``serialize_to_hdf5``.

        Returns:
            ScoreBasedUncertaintyModel: Fully restored model instance, ready
                for sampling if the serialised model was trained.
        """
        attrs = hdf5_group.attrs
        instance = cls(
            filter_names=json.loads(attrs["filter_names"]),
            hidden_dim=int(attrs["hidden_dim"]),
            n_layers=int(attrs["n_layers"]),
            time_embed_dim=int(attrs["time_embed_dim"]),
            beta_min=float(attrs["beta_min"]),
            beta_max=float(attrs["beta_max"]),
            T=float(attrs["T"]),
            return_noise=bool(attrs["return_noise"]),
        )

        if not attrs.get("is_trained", False):
            return instance

        dev = instance.device
        instance._ln_sigma_median = torch.tensor(
            hdf5_group["ln_sigma_median"][:], dtype=torch.float32, device=dev
        )
        instance._ln_sigma_iqr = torch.tensor(
            hdf5_group["ln_sigma_iqr"][:], dtype=torch.float32, device=dev
        )
        instance._mag_median = torch.tensor(
            hdf5_group["mag_median"][:], dtype=torch.float32, device=dev
        )
        instance._mag_iqr = torch.tensor(hdf5_group["mag_iqr"][:], dtype=torch.float32, device=dev)
        instance._mag_max = torch.tensor(hdf5_group["mag_max"][:], dtype=torch.float32, device=dev)
        instance._mag_min = torch.tensor(hdf5_group["mag_min"][:], dtype=torch.float32, device=dev)

        state_dict: Dict[str, torch.Tensor] = {}
        weights_group = hdf5_group["score_network_weights"]
        for safe_key in weights_group.keys():
            original_key = safe_key.replace("__", ".")
            state_dict[original_key] = torch.tensor(weights_group[safe_key][:], dtype=torch.float32)

        target = instance.score_net
        if hasattr(target, "_orig_mod"):
            target = target._orig_mod

        target.load_state_dict(state_dict)
        instance.ema_net.module.load_state_dict(state_dict)
        instance.score_net.to(dev)
        instance.ema_net.to(dev)
        instance._is_trained = True
        return instance


# =============================================================================
# SERIALIZATION FACTORY FUNCTIONS
# =============================================================================

MODEL_CLASS_REGISTRY = {
    "DepthUncertaintyModel": DepthUncertaintyModel,
    "AsinhEmpiricalUncertaintyModel": AsinhEmpiricalUncertaintyModel,
    "GeneralEmpiricalUncertaintyModel": GeneralEmpiricalUncertaintyModel,
    "ScoreBasedUncertaintyModel": ScoreBasedUncertaintyModel,
}


def save_unc_model_to_hdf5(
    model: UncertaintyModel, filepath: str, group_name: str, overwrite: bool = False
):
    """Saves a supported uncertainty model to an HDF5 file."""
    with h5py.File(filepath, "a") as f:
        if group_name in f:
            if overwrite:
                del f[group_name]
            else:
                raise ValueError(f"Group '{group_name}' already exists.")
        group = f.create_group(group_name)
        model.serialize_to_hdf5(group)


def load_unc_model_from_hdf5(filepath: str, group_name: str = "all") -> UncertaintyModel:
    """Factory function to load any supported model from an HDF5 file."""
    with h5py.File(filepath, "r") as f:
        if group_name == "all":
            models = {}
            for name in f.keys():
                group = f[name]
                if group.attrs is None or "__class__" not in group.attrs:
                    # Go one level deeper if needed
                    for name2 in group.keys():
                        group2 = group[name2]
                        class_name = group2.attrs.get("__class__")
                        models[f"{name}/{name2}"] = MODEL_CLASS_REGISTRY[
                            class_name
                        ]._from_hdf5_group(group2)  # noqa: E501
                else:
                    class_name = group.attrs.get("__class__")

                    if class_name not in MODEL_CLASS_REGISTRY:
                        raise TypeError(f"Unknown model class '{class_name}'.")
                    models[name] = MODEL_CLASS_REGISTRY[class_name]._from_hdf5_group(group)
            return models
        else:
            if group_name not in f:
                raise KeyError(f"Group '{group_name}' not found.")
            group = f[group_name]
            class_name = group.attrs.get("__class__")
            if class_name not in MODEL_CLASS_REGISTRY:
                raise TypeError(f"Unknown model class '{class_name}'.")
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
    model_class="general",
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
        loc_depth = table[f"loc_depth_{band}"]

        if old:
            mag = mag[:, 0]
            flux = flux[:, 0]
            flux_err = flux[:, 0]

        mag_err = (2.5 * flux_err) / (flux * np.log(10))
        mask = (mag != -99) & (np.isfinite(mag)) & (mag_err >= 0)
        mag = mag[mask]
        mag_err = mag_err[mask]
        base_unc_kwargs = {"return_noise": True, "error_type": "observed", "num_bins": 20}
        if model_class == "general":
            unc_kwargs = dict(
                log_bins=False,
                upper_limits=True,
                treat_as_upper_limits_below=1,
                upper_limit_flux_behaviour=40,
                upper_limit_flux_err_behaviour="sig_1",
            )
            unc_kwargs.update(base_unc_kwargs)
            unc_kwargs.update(kwargs)

            # So this behaviour is to mask any fluxes with SNR < 1 either
            # before or after the scattering,
            # , setting the error to 1 sigma.
            noise_model = GeneralEmpiricalUncertaintyModel(
                mag,
                mag_err,
                **unc_kwargs,
            )
        elif model_class == "depth":
            base_unc_kwargs.update(kwargs)
            if isinstance(loc_depth.data, np.ma.MaskedArray):
                loc_depth = loc_depth.data
            noise_model = DepthUncertaintyModel(
                np.nanmedian(loc_depth.data),
                depth_sigma_level=5.0,
                **base_unc_kwargs,
            )
        elif model_class == "asinh":
            base_unc_kwargs.update(kwargs)
            base_unc_kwargs["interpolation_flux_unit"] = "asinh"
            base_unc_kwargs["log_bins"] = True

            noise_model = AsinhEmpiricalUncertaintyModel(
                flux,
                flux_err,
                **base_unc_kwargs,
            )
        else:
            raise ValueError(
                f"Unknown model_class: {model_class}. Supported: 'general', 'depth', 'asinh'."
            )

        unc_models[band_new_name] = noise_model

        if plot:
            # bin and plot as contour
            plt.figure(figsize=(10, 6))

            plt.title(
                f"{model_class.capitalize()} Uncertainty Model for {band_new_name}", fontsize=16
            )

            if model_class == "depth" or model_class == "general":
                plt.scatter(mag, mag_err, alpha=0.05, color="black", s=0.15, zorder=10)
                plt.ylim(0, 1.2)
            elif model_class == "asinh":
                converted_mag, converted_mag_err = noise_model.apply_scalings(flux, flux_err)
                plt.scatter(
                    converted_mag, converted_mag_err, alpha=0.05, color="black", s=0.15, zorder=10
                )

            mag = np.linspace(23, 40, 10000)
            noisy_flux, sampled_sigma = noise_model.apply_noise(
                mag, true_flux_units="AB", out_units="AB"
            )

            # plt.scatter(noisy_flux, sampled_sigma, alpha=0.1, color='green', s=0.1)
            plt.hexbin(
                noisy_flux,
                sampled_sigma,
                gridsize=50,
                cmap="Greens",
                mincnt=1,
                norm="log",
                extent=(23, 42, 0, np.nanmax(sampled_sigma) * 1.1),
                alpha=1,
                label=r"$p\left(\sigma_X \mid f_X\right)$",
            )
            plt.legend(loc="upper left", fontsize=12)

            plt.xlabel("Magnitude", fontsize=14)
            plt.ylabel(r"$\sigma_{\rm m, AB}$", fontsize=14)
            if save:
                save_band_name = band_new_name.replace("/", "_")
                plt.savefig(
                    f"{save_path}/uncertainty_model_{model_class}_{save_band_name}.png", dpi=300
                )
            else:
                plt.show()
    return unc_models
