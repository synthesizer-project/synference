"""Noise models for photometric fluxes."""

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.table import Column, Table
from scipy import stats
from scipy.interpolate import interp1d
from unyt import Jy, Unit, unyt_array

from .utils import f_jy_err_to_asinh, f_jy_to_asinh


class UncertaintyModel:
    """base class for uncertainty models.

    This class defines the interface for uncertainty models that can
    apply noise to fluxes based on a given model.

    Attributes:
        per_filter (bool): If True, the model applies noise per filter.
            If False, it applies noise to the entire flux vector.
            Default is True.
        return_noise (bool): If True, the model returns the noise applied to the flux.
    """

    per_filter: bool = True
    return_noise: bool = False

    def __init__(self, per_filter: bool = True, return_noise: bool = False, **kwargs):
        """Initialize the uncertainty model.

        Args:
            per_filter: If True, the model applies noise per filter.
                        If False, it applies noise to the entire flux vector.
            return_noise: If True, the model returns the noise applied to the flux.
            **kwargs: Additional keyword arguments to pass to the model.
        """
        self.per_filter = per_filter
        self.return_noise = return_noise

        self.parameters = kwargs

    def apply_noise_to_flux(self):
        """Apply noise to the flux based on the model."""
        pass

    def _phot_ab_to_jy(self, flux: float) -> unyt_array:
        """Convert AB magnitude to Jy flux."""
        return 10 ** (-0.4 * (flux + 8.9)) * Jy

    def _phot_jy_to_ab(self, flux: unyt_array) -> float:
        """Convert Jy flux to AB magnitude."""
        return -2.5 * np.log10(flux.to(Jy).value) + 8.9

    def _phot_err_ab_to_jy(self, flux: unyt_array, error: np.ndarray) -> unyt_array:
        """Convert AB magnitude error to Jy flux error."""
        flux = flux.to(Jy)
        return (np.log(10) * flux * error) / 2.5

    def _phot_err_jy_to_ab(self, flux: unyt_array, error: unyt_array) -> np.ndarray:
        """Convert Jy flux error to AB magnitude error."""
        return 2.5 / np.log(10) * (error / flux.to(Jy).value)

    def _phot_jy_to_asinh(
        self, flux: unyt_array, asinh_softening_parameter: unyt_array
    ) -> unyt_array:
        """Convert Jy flux to asinh magnitude."""
        return f_jy_to_asinh(flux, asinh_softening_parameter)

    def _phot_err_jy_to_asinh(
        self, flux: unyt_array, error: unyt_array, asinh_softening_parameter: unyt_array
    ) -> unyt_array:
        """Convert Jy flux error to asinh magnitude error."""
        return f_jy_err_to_asinh(flux, error, asinh_softening_parameter)

    def handle_flux_conversion(
        self,
        model_flux: np.ndarray,
        model_flux_units: Union[str, Unit] = "AB",
        out_units: Optional[str] = "Jy",
    ):
        """Handle conversion of fluxes based on the model's requirements.

        Args:
            model_flux: The flux values to convert.
            model_flux_units: The units of the flux values.
                Default is "AB". Can be a string or a unyt Unit.
            out_units: The units to convert the fluxes to.
                If None, defaults to Jy.

        Returns:
            model_flux: The converted flux values.
            model_flux_units: The units of the converted flux values.
        """
        if isinstance(model_flux_units, str):
            model_flux_units = Unit(model_flux_units)

        if out_units is None:
            out_units = Jy

        if model_flux_units == "AB":
            model_flux = self._phot_ab_to_jy(model_flux)
            model_flux_units = Jy
        elif model_flux_units == Jy:
            model_flux = unyt_array(model_flux, units=Jy)
        elif isinstance(model_flux_units, Unit):
            model_flux = unyt_array(model_flux, units=model_flux_units)
        else:
            raise ValueError("model_flux_units must be 'AB', 'Jy', or a valid unyt Unit.")

        if out_units == "AB":
            model_flux = self._phot_jy_to_ab(model_flux)
            model_flux_units = "AB"
        elif out_units == Jy:
            model_flux = unyt_array(model_flux, units=Jy)
        elif isinstance(out_units, Unit):
            model_flux = unyt_array(model_flux, units=out_units)
        else:
            raise ValueError("out_units must be 'AB', 'Jy', or a valid unyt Unit.")

        return model_flux

    def handle_flux_error_conversion(
        self,
        model_flux: np.ndarray,
        model_flux_units: Union[str, Unit] = "AB",
        model_flux_error: np.ndarray = None,
        out_units: Optional[str] = "Jy",
    ):
        """Handle conversion of flux errors based on the model's requirements.

        Args:
            model_flux: The flux values to convert.
            model_flux_units: The units of the flux values.
                Default is "AB". Can be a string or a unyt Unit.
            model_flux_error: The flux error values to convert.
            out_units: The units to convert the flux errors to.
                If None, defaults to Jy.

        Returns:
            model_flux_error: The converted flux error values.
        """
        if model_flux_error is None:
            return None

        if isinstance(model_flux_units, str):
            model_flux_units = Unit(model_flux_units)

        if out_units is None:
            out_units = Jy

        if model_flux_units == "AB":
            model_flux_error = self._phot_err_ab_to_jy(model_flux, model_flux_error)
            model_flux_units = Jy
        elif model_flux_units == Jy:
            model_flux_error = unyt_array(model_flux_error, units=Jy)
        elif isinstance(model_flux_units, Unit):
            model_flux_error = unyt_array(model_flux_error, units=model_flux_units)
        else:
            raise ValueError("model_flux_units must be 'AB', 'Jy', or a valid unyt Unit.")

        if out_units == "AB":
            model_flux_error = self._phot_err_jy_to_ab(model_flux, model_flux_error)
            model_flux_units = "AB"
        elif out_units == Jy:
            model_flux_error = unyt_array(model_flux_error, units=Jy)
        elif isinstance(out_units, Unit):
            model_flux_error = unyt_array(model_flux_error, units=out_units)
        else:
            raise ValueError("out_units must be 'AB', 'Jy', or a valid unyt Unit.")

        return model_flux_error

    def serialize_to_hdf5(self, hdf5_file: str, group_name: str):
        """Serialize the model to an HDF5 file."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def __getstate__(self) -> Dict[str, Any]:
        """Prepare a serializable state dictionary for pickling.

        Converts array-like attributes to basic Python lists.
        """
        state: Dict[str, Any] = {}
        for attr in self.__dict__:
            # Use a default value of None if the attribute doesn't exist
            value = getattr(self, attr, None)

            # Use a single if/elif/else chain to handle conversions
            if isinstance(value, (np.ndarray, unyt_array, Column)):
                # For any array-like type, convert to a list for serialization
                state[attr] = np.array(value)
            elif callable(value):
                # Skip callable attributes (like methods)
                continue
            else:
                # Keep all other types (lists, ints, floats, str, etc.) as they are
                state[attr] = value

        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore the object's state from a pickled state dictionary."""
        # get args with inspect
        import inspect

        args = inspect.getfullargspec(self.__init__).args
        # Filter out 'self' from the args
        args = [arg for arg in args if arg != "self"]
        # Create a dictionary with the state values for the init args
        args = {arg: state.get(arg, None) for arg in args}

        self.__init__(**args)


class DepthUncertaintyModel(UncertaintyModel):
    """An uncertainty model that applies noise based on a depth value."""

    def __init__(self, depth: unyt_array, depth_sigma=5, **kwargs):
        """Initialize the model with a depth value.

        Args:
            depth: The depth value to use for noise application.
            depth_sigma: The standard deviation of the noise to apply.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        self.depth = depth
        self.depth_sigma = depth_sigma

        self.sigma = self.depth / self.depth_sigma

        kwargs = {
            "depth": self.depth,
            "depth_sigma": self.depth_sigma,
            "sigma": self.sigma,
            "per_filter": True,
            **kwargs,
        }

        super().__init__(**kwargs)

    def apply_noise_to_flux(
        self,
        model_flux: np.ndarray,
        model_flux_units: Union[str, Unit] = "AB",
    ):
        """Apply noise to the flux based on the depth value."""
        flux = self.handle_flux_conversion(
            model_flux=model_flux, model_flux_units=model_flux_units, out_units="Jy"
        )

        noise = np.random.normal(loc=0, scale=self.sigma, size=flux.shape)

        # Add the random values to the repeated photometry
        output_arr = flux + noise

        output_flux = self.handle_flux_conversion(
            model_flux=output_arr,
            model_flux_units="Jy",
            out_units=model_flux_units,
        )

        if self.return_noise:
            # Noise in this case is self.sigma - the
            # standard deviation of the Gaussian noise applied
            #
            sigma = np.ones(output_flux.shape) * self.sigma
            sigma = self.handle_flux_error_conversion(
                model_flux=output_arr,
                model_flux_units="Jy",
                model_flux_error=sigma,
                out_units=model_flux_units,
            )

            return output_flux, sigma

        return output_flux


class DiffusionUncertaintyModel(UncertaintyModel):
    """A class to model and sample photometric uncertainties based on observations.

    The model estimates p(sigma_X | f_X) as a
    Gaussian N(mu_sigma_X(f_X), sigma_sigma_X(f_X)),

    It uses a score-based diffusion model to model
    the uncertanity vector sigma_X, given
    observed fluxes f_X and errors sigma_X.
    Based on the Thorp et al. 2025 paper:
        https://arxiv.org/pdf/2506.12122
    """

    def __init__(self, **kwargs):
        """Initialize the diffusion uncertainty model."""
        super().__init__(**kwargs)
        raise NotImplementedError("DiffusionUncertaintyModel is not implemented yet. ")


class EmpiricalUncertaintyModel(UncertaintyModel):
    """A class to model and sample photometric uncertainties based on observations."""

    def __init__(
        self, error_type: str = "empirical", extrapolate_uncertanties: bool = False, **kwargs
    ):
        """Initialize the empirical uncertainty model.

        Parameters
        ----------
        error_type : str
            Type of error model to use, e.g., 'empirical', 'observed'.
            Default is 'empirical'.
        extrapolate_uncertanties : bool
            If True, allows extrapolation of uncertainties beyond the range of the flux bins.
            If False, uses the nearest bin value for extrapolation.
            Default is False.
        **kwargs : dict
            Additional keyword arguments passed to the parent class.
        """
        self.error_type = error_type
        self.extrapolate_uncertanties = extrapolate_uncertanties

        return super().__init__(**kwargs)

    def _setup_bins(self, num_bins: int = 20, log_bins: bool = True):
        # Calculate bins

        if log_bins:
            bins = np.logspace(
                np.log10(np.nanmin(self.mag)),
                np.log10(np.nanmax(self.mag)),
                num_bins + 1,
            )

        else:
            bins = np.linspace(np.nanmin(self.mag), np.nanmax(self.mag), num_bins + 1)

        self.flux_bins_centers: List[float] = []
        bin_median_errors: List[float] = []
        bin_std_errors: List[float] = []

        # To avoid negative issues, predict log flux_err

        for i in range(len(bins) - 1):
            low, high = bins[i], bins[i + 1]
            # Ensure the last bin includes the maximum value
            if i == len(self.mag) - 2:
                mask = (self.mag >= low) & (self.mag <= high)
            else:
                mask = (self.mag >= low) & (self.mag < high)

            errors_in_bin = self.mag_err[mask]

            if len(errors_in_bin) > 0:
                self.flux_bins_centers.append(low + (high - low) / 2.0)

                bin_median_errors.append(np.median(errors_in_bin))
                bin_std_errors.append(np.std(errors_in_bin))

        if len(self.flux_bins_centers) < 2:  # Need at least two points for interpolation
            raise ValueError(
                f"Could not create enough valid bins ({len(self.flux_bins_centers)}) "
                "for interpolation. Try adjusting the input data."
            )
        self.flux_bins_centers = np.array(self.flux_bins_centers)
        # Store the flux range for which the model is considered valid

        # Ignore bounds issues for now

        self._min_interp_flux = self.flux_bins_centers[0]
        self._max_interp_flux = self.flux_bins_centers[-1]

        if not self.extrapolate_uncertanties:
            fill_value = (bin_median_errors[0], bin_median_errors[-1])
        else:
            fill_value = "extrapolate"

        # Use 'bounds_error=False' and 'fill_value' to handle extrapolation.
        # For sigma_sigma_X (std of errors), it should not be negative.
        self.mu_sigma_interpolator: Callable[
            [Union[float, np.ndarray]], Union[float, np.ndarray]
        ] = interp1d(
            self.flux_bins_centers,
            bin_median_errors,
            kind="linear",
            bounds_error=False,
            fill_value=fill_value,
        )

        self.sigma_sigma_interpolator: Callable[
            [Union[float, np.ndarray]], Union[float, np.ndarray]
        ] = interp1d(
            self.flux_bins_centers,
            bin_std_errors,
            kind="linear",
            bounds_error=False,
            fill_value=(bin_std_errors[0], bin_std_errors[-1]),
        )

    def plot_sigma(self, ax=None, **kwargs):
        """Plots the interpolated mu_sigma_X and sigma_sigma_X.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, creates a new figure and axes.
        **kwargs : dict, optional
            Additional keyword arguments passed to the plot function.
        """
        if ax is None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
        flux_range = np.linspace(self._min_interp_flux, self._max_interp_flux, 1000)
        mu_sigma = self.mu_sigma_interpolator(flux_range)
        sigma_sigma = self.sigma_sigma_interpolator(flux_range)

        line = ax.plot(flux_range, mu_sigma, **kwargs)
        ax.fill_between(
            flux_range,
            mu_sigma - sigma_sigma,
            mu_sigma + sigma_sigma,
            alpha=0.2,
            color=line[0].get_color(),
        )

        ax.set_xlabel("Flux")
        ax.set_ylabel("Flux Uncertainty")

    def get_valid_flux_range(self) -> Tuple[float, float]:
        """Returns the flux range for which the interpolator was built."""
        return self._min_interp_flux, self._max_interp_flux

    def sample_uncertainty(
        self,
        true_flux: Union[float, np.ndarray],
        filter_negative: bool = True,
    ) -> Union[float, np.ndarray, None]:
        """Samples a 'fake' uncertainty (sigma_prime_X) for a given true flux.

        Parameters
        ----------
        true_flux : float or np.ndarray
            The true flux value(s) for which to sample the uncertainty.
        extrapolate : bool, optional
            If True, allows extrapolation beyond the range of the flux bins.
            If False, uses the nearest bin value for extrapolation.
        filter_negative : bool, optional
            If True, any sampled uncertainties that are negative will be set to NaN.

        Returns:
        -------
        sampled_sigmas : float or np.ndarray
            The sampled uncertainties for the given true flux value(s).

        """
        is_scalar = np.isscalar(true_flux)
        flux_array = np.atleast_1d(true_flux)
        sampled_sigmas = np.empty_like(flux_array)

        mu_sigma_values = self.mu_sigma_interpolator(flux_array)
        sigma_sigma_values = self.sigma_sigma_interpolator(flux_array)

        # Define the bounds for truncation (0 to infinity)
        lower_bound = 0
        upper_bound = np.inf

        # Calculate the bounds in terms of standard deviations from the mean
        # This is the format required by scipy.stats.truncnorm
        a = (lower_bound - mu_sigma_values) / sigma_sigma_values
        b = (upper_bound - mu_sigma_values) / sigma_sigma_values

        from scipy.stats import truncnorm

        # Sample from the truncated normal distribution
        sampled_sigmas = truncnorm.rvs(
            a=a,
            b=b,
            loc=mu_sigma_values,
            scale=sigma_sigma_values,
            size=flux_array.shape,
        )

        return sampled_sigmas[0] if is_scalar else sampled_sigmas

    def apply_noise_to_flux(self):
        """Apply noise to a flux based on the uncertainty model."""
        raise NotImplementedError(
            "apply_noise_to_flux is not implemented for EmpiricalUncertaintyModel. "
            "Use a subclass that implements this method."
        )

    def __call__(self, true_flux: Union[float, np.ndarray]):
        """Sample uncertainty for a given true flux.

        This method allows the model to be called like a function.
        """
        return self.apply_noise_to_flux(true_flux)


class AsinhEmpiricalUncertaintyModel(EmpiricalUncertaintyModel):
    """A class to model and sample photometric uncertainties based on observations.

    The model estimates p(sigma_X | f_X) as a
    Gaussian N(mu_sigma_X(f_X), sigma_sigma_X(f_X)),
    where mu_sigma_X and sigma_sigma_X are
    interpolated from binned statistics of
    observed (sigma_X, f_X) pairs.
    """

    def __init__(
        self,
        observed_phot: unyt_array,
        observed_phot_errors: unyt_array,
        asinh_softening_parameter: Optional[unyt_array] = None,
        asinh_sigma_level: float = 5.0,
        min_flux_error: Optional[float] = 0.0,
        sample_log_err: bool = False,
        **kwargs,
    ):
        """Initialize the AsinhEmpiricalUncertaintyModel.

        Parameters
        ----------
        observed_phot : unyt_array
            The observed photometric flux values.
        observed_phot_errors : unyt_array
            The observed photometric flux errors.
        asinh_softening_parameter : unyt_array, optional
            The softening parameter for the asinh transformation.
            If None, it will be set to 5 times the median flux error.
        asinh_sigma_level : float, optional
            The sigma level for the asinh transformation.
            Default is 5.0.
        min_flux_error : float, optional
            The minimum flux error to apply.
            If None, defaults to 0.0.
        sample_log_err : bool, optional
            If True, the uncertainties will be sampled in log space.
            If False, they will be sampled in linear space.
            Default is False.
        **kwargs : dict
            Additional keyword arguments passed to the parent class.
        """
        self.observed_phot = observed_phot
        self.observed_phot_errors = observed_phot_errors
        self.asinh_sigma_level = asinh_sigma_level
        self.min_flux_error = min_flux_error
        self.sample_log_err = sample_log_err

        kwargs = {
            "observed_phot": self.observed_phot,
            "observed_phot_errors": self.observed_phot_errors,
            "asinh_softening_parameter": asinh_softening_parameter,
            "asinh_sigma_level": self.asinh_sigma_level,
            "min_flux_error": self.min_flux_error,
            "sample_log_err": self.sample_log_err,
            **kwargs,
        }

        super().__init__(**kwargs)

        # Steps for an asinh model.

        # 1. From observed data (in flux space), find 5 sigma detection limit.
        # 2. Using this limit, calculate the asinh magnitude and errors.
        # 3. Interpolate the asinh magnitude and errors to create a model.

        if not isinstance(self.observed_phot, unyt_array):
            raise TypeError("observed_phot must be a unyt_array.")

        if not isinstance(self.observed_phot_errors, unyt_array):
            raise TypeError("observed_phot_errors must be a unyt_array.")

        if self.observed_phot.shape != self.observed_phot_errors.shape:
            raise ValueError("observed_phot and observed_phot_errors must have the same shape.")

        if self.extrapolate_uncertanties and not self.sample_log_err:
            print(
                "Warning! Extrapolating uncertainties with sample_log_err=False may lead to negative uncertainties."  # noqa: E501
            )

        # Filter out non-finite values and errors

        valid_mask = (
            np.isfinite(self.observed_phot)
            & np.isfinite(self.observed_phot_errors)
            & (self.observed_phot_errors > 0)
        )

        self.observed_phot = self.observed_phot[valid_mask]
        self.observed_phot_errors = self.observed_phot_errors[valid_mask]

        if asinh_softening_parameter is None:
            median_unc = np.median(self.observed_phot_errors)

            self.asinh_softening_parameter = self.asinh_sigma_level * median_unc
        else:
            self.asinh_softening_parameter = asinh_softening_parameter

        self.mag = self._phot_jy_to_asinh(
            self.observed_phot.to("Jy"), self.asinh_softening_parameter.to("Jy")
        )

        self.mag_err = self._phot_err_jy_to_asinh(
            self.observed_phot.to("Jy"),
            self.observed_phot_errors.to("Jy"),
            self.asinh_softening_parameter.to("Jy"),
        )

        if self.sample_log_err:
            self.mag_err = np.log10(self.mag_err)

        self._setup_bins(log_bins=False, num_bins=40)

    def apply_noise_to_flux(
        self,
        true_flux: unyt_array,
    ) -> Tuple[Union[float, np.ndarray, None], Union[float, np.ndarray, None]]:
        """Applies noise to a photometric measurement.

        Parameters
        ----------
        true_flux : float or np.ndarray
            The true flux value(s) to which noise will be applied.
        """
        true_flux = true_flux.copy()

        is_scalar = np.isscalar(true_flux)
        flux_array = np.atleast_1d(true_flux)

        true_mag = self._phot_jy_to_asinh(
            flux_array,
            self.asinh_softening_parameter.to(flux_array.units),
        )
        # Calculate the log of the uncertainty kernel
        log_sigma = self.sample_uncertainty(true_mag, filter_negative=not self.sample_log_err)

        if self.sample_log_err:
            sigma = 10**log_sigma
        else:
            sigma = log_sigma

        # Apply minimum flux error
        sigma[sigma < self.min_flux_error] = self.min_flux_error

        # Scatter the fluxes based on the sampled uncertainties
        noise = np.random.normal(loc=0, scale=sigma)

        noisy_mag = true_mag + noise
        # Ensure noise is not below the minimum error
        noisy_mag[np.isnan(sigma)] = np.nan

        if self.error_type == "observed":
            # Re-estimate the errors based on the scattered fluxes
            log_sigma = self.sample_uncertainty(noisy_mag, filter_negative=not self.sample_log_err)
            if self.sample_log_err:
                sigma = 10**log_sigma
            else:
                sigma = log_sigma

        if is_scalar:
            return noisy_mag[0], sigma[0]
        return noisy_mag, sigma


class GeneralEmpiricalUncertaintyModel(EmpiricalUncertaintyModel):
    """A class to model and sample photometric uncertainties based on observations.

    The model estimates p(sigma_X | f_X) as a
    Gaussian N(mu_sigma_X(f_X), sigma_sigma_X(f_X)),
    where mu_sigma_X and sigma_sigma_X are
    interpolated from binned statistics of
    observed (sigma_X, f_X) pairs.
    """

    def __init__(
        self,
        observed_fluxes: np.ndarray,
        observed_errors: np.ndarray,
        num_bins: int = 20,
        flux_bins: Optional[np.ndarray] = None,
        log_bins: bool = True,
        min_flux_for_binning: Optional[float] = None,
        min_samples_per_bin: int = 10,
        flux_unit: str = "AB",
        interpolation_flux_unit: str = None,
        min_flux_error: Optional[float] = None,
        error_type: str = "empirical",
        sigma_clip: float = 3.0,
        upper_limits: bool = False,
        treat_as_upper_limits_below: Optional[float] = None,
        upper_limit_flux_behaviour: str = "scatter_limit",
        upper_limit_flux_err_behaviour: str = "flux",
        max_flux_error: Optional[float] = None,
        already_binned: bool = False,
        bin_median_errors=None,
        bin_std_errors=None,
    ):
        """Uncertainity model from observed data.

        Args:
            observed_fluxes: 1D array of fluxes from a real survey.
            observed_errors: 1D array of corresponding flux uncertainties.
            num_bins: Number of bins to use for flux if flux_bins is not provided.
            flux_bins: Optional array defining the edges of flux bins.
                       If None, bins are created based on num_bins and log_bins.
            log_bins: If True and flux_bins is None, bins will be spaced
                      logarithmically. Otherwise, linearly.
            min_flux_for_binning: If provided, only fluxes above this value are
                                  used for creating the interpolation model.
                                  This can help avoid issues with very low/zero fluxes.
            min_samples_per_bin: Minimum number of samples required in a bin
                                 for it to be considered valid for interpolation.
            flux_unit: The unit of the fluxes, e.g., 'AB', 'Jy', etc.
            interpolation_flux_unit: The unit of the fluxes used for interpolation.
                                If None, defaults to flux_unit.
            min_flux_error: Minimum value for the estimated flux error.
                             If None, defaults to 0.0.
            error_type: What kind of error to return. If 'empirical', then sigma_X
                        is the standard deviation of
                        the Gaussian used to scatter the fluxes. If 'observed' then
                        we re-estimate the errors based on the scattered fluxes.
            sigma_clip: number of standard deviations to clip the sampled uncertainties.
            upper_limits: If True, treat the model fluxes below treat_as_upper_limits_below
                    as upper limits. If False, all fluxes are treated as normal.
            treat_as_upper_limits_below: If provided, fluxes below this SNR
                    are treated as upper limits.
            upper_limit_flux_behaviour: How to handle upper limits fluxes.
                Options are 'scatter_limit', 'upper_limit':
                - scatter_limit: Use the upper limit value as the flux and scatter the
                    flux using the model.
                - upper_limit: Use the upper limit value as the flux with no scattering.
                - a float/int - Use this value as the upper limit flux.
            upper_limit_flux_err_behaviour: How to handle upper limits flux errors.
            Options are 'flux', 'upper_limit'.:
                - flux: Use the flux_err scatter at the flux value.
                - 'upper_limit': Use the upper limit value as the flux error.
                - 'max' : Use the maximum flux error set by max_flux_error.
                - 'sig_{val}': Find the error at a specific value, e.g., 'sig_1'
                    will find the error at 1 sigma.
            max_flux_error: Maximum flux error to allow. If None, no maximum is applied.
            already_binned: If True, the observed_fluxes and observed_errors are already binned
            and do not need to be binned again. This is useful for cases where the
            fluxes and errors are already pre-processed and ready for interpolation.
            bin_median_errors: Optional pre-computed median errors for each bin.
                only used if already_binned is True.
            bin_std_errors: Optional pre-computed standard deviations for each bin.
                Only used if already_binned is True.
        """
        if len(observed_fluxes) != len(observed_errors):
            raise ValueError("observed_fluxes and observed_errors must have the same length.")
        self.sigma_clip = sigma_clip

        self.observed_fluxes = observed_fluxes
        self.observed_errors = observed_errors
        self.num_bins = num_bins
        self.flux_bins = flux_bins
        self.log_bins = log_bins
        if min_flux_for_binning == "None":
            min_flux_for_binning = None
        self.min_flux_for_binning = min_flux_for_binning
        self.min_samples_per_bin = min_samples_per_bin
        self.flux_unit = flux_unit
        self.min_flux_error = min_flux_error if min_flux_error is not None else 0.0
        self.upper_limit_flux_behaviour = upper_limit_flux_behaviour

        if interpolation_flux_unit is None:
            self.interpolation_flux_unit = flux_unit
        else:  # Use provided interpolation flux unit
            self.interpolation_flux_unit = interpolation_flux_unit

        # convert self.observed_fluxes and self.observed_errors to interpolation_flux_unit

        if self.interpolation_flux_unit != self.flux_unit:
            if isinstance(self.interpolation_flux_unit, Unit) and isinstance(self.flux_unit, Unit):
                # If both are unyt units, convert them
                observed_fluxes = (
                    (self.observed_fluxes * self.interpolation_flux_unit).to(self.flux_unit).value
                )
                observed_errors = (
                    (self.observed_errors * self.interpolation_flux_unit).to(self.flux_unit).value
                )

            elif isinstance(self.interpolation_flux_unit, Unit) and self.flux_unit == "AB":
                # Convert from AB magnitudes to to unyt fluxes

                observed_fluxes = (10 ** (-0.4 * (self.observed_fluxes + 8.9))) * Jy
                observed_fluxes = observed_fluxes.to(self.interpolation_flux_unit)
                observed_errors = (
                    self.observed_errors * np.log(10) * observed_fluxes
                ) / 2.5  # Convert errors to Jy
                observed_errors = observed_errors.to(self.interpolation_flux_unit).value
                observed_fluxes = observed_fluxes.value
            elif self.interpolation_flux_unit == "AB" and isinstance(self.flux_unit, Unit):
                # Convert from unyt fluxes to AB magnitudes
                observed_errors = 2.5 / np.log(10) * (self.observed_errors / self.observed_fluxes)
                observed_fluxes = (
                    -2.5 * np.log10((self.observed_fluxes * self.flux_unit).to("Jy").value) + 8.9
                )
            else:
                raise ValueError("interpolation_flux_unit must be a valid unyt unit or 'AB'.")

            self.observed_fluxes = observed_fluxes
            self.observed_errors = observed_errors
            self.flux_unit = self.interpolation_flux_unit

        valid_mask = (
            np.isfinite(observed_fluxes) & np.isfinite(observed_errors) & (observed_errors > 0)
        )
        if min_flux_for_binning is not None:
            valid_mask &= observed_fluxes > min_flux_for_binning

        fluxes = observed_fluxes[valid_mask]
        errors = observed_errors[valid_mask]

        assert error_type in ["empirical", "observed"], (
            "error_type must be either 'empirical' or 'observed'."
        )
        self.error_type = error_type

        if upper_limits:
            assert treat_as_upper_limits_below is not None, (
                "If upper_limits is True, treat_as_upper_limits_below must be provided."
            )

        self.max_observed_flux_err = np.max(errors)

        self.upper_limits = upper_limits
        self.treat_as_upper_limits_below = treat_as_upper_limits_below
        self.upper_limit_flux_behaviour = upper_limit_flux_behaviour
        self.upper_limit_flux_err_behaviour = upper_limit_flux_err_behaviour

        if upper_limits:
            if flux_unit == "AB":
                # convert fluxes to Jy if they are in AB magnitudes
                ftemp = 10 ** (-0.4 * (fluxes + 8.9))  # Convert AB magnitudes to Jy
                etemp = (errors * np.log(10) * ftemp) / 2.5  # Convert errors to Jy
            else:
                ftemp = fluxes
                etemp = errors

            self.snr_interpolator = interp1d(ftemp / etemp, ftemp)
            upper_limit_value = self.snr_interpolator(treat_as_upper_limits_below)
            # Convert upper_limit_value back to the original flux unit
            if flux_unit == "AB":
                upper_limit_value = -2.5 * np.log10(upper_limit_value) + 8.9
        else:
            upper_limit_value = None

        self.upper_limit_value = upper_limit_value

        self.max_flux_error = max_flux_error if max_flux_error is not None else np.inf

        if not already_binned:
            if len(fluxes) < min_samples_per_bin * 2:  # Need at least two bins for interpolation
                raise ValueError(
                    f"Not enough valid data points ({len(fluxes)}) to build the model "
                    f"with min_samples_per_bin={min_samples_per_bin}. "
                    "Consider adjusting min_flux_for_binning or providing more data."
                )

            if flux_bins is None:
                if log_bins:
                    # Ensure fluxes are positive for log binning
                    positive_flux_mask = fluxes > 0
                    if not np.any(positive_flux_mask):
                        raise ValueError(
                            """No positive fluxes available for log binning.
                            Try linear bins or check data."""
                        )
                    min_f = np.min(fluxes[positive_flux_mask])
                    max_f = np.max(fluxes[positive_flux_mask])
                    if min_f <= 0:  # Should be caught by positive_flux_mask, but as safeguard
                        min_f = (
                            np.partition(fluxes[positive_flux_mask], 1)[1]
                            if len(fluxes[positive_flux_mask]) > 1
                            else 1e-9
                        )
                    flux_bins = np.logspace(np.log10(min_f), np.log10(max_f), num_bins + 1)
                else:
                    min_f = np.min(fluxes)
                    max_f = np.max(fluxes)
                    flux_bins = np.linspace(min_f, max_f, num_bins + 1)

            self.flux_bins_centers: List[float] = []
            bin_median_errors: List[float] = []
            bin_std_errors: List[float] = []

            for i in range(len(flux_bins) - 1):
                low_f, high_f = flux_bins[i], flux_bins[i + 1]
                # Ensure the last bin includes the maximum value
                if i == len(flux_bins) - 2:
                    mask = (fluxes >= low_f) & (fluxes <= high_f)
                else:
                    mask = (fluxes >= low_f) & (fluxes < high_f)

                errors_in_bin = errors[mask]

                if len(errors_in_bin) >= min_samples_per_bin:
                    self.flux_bins_centers.append(low_f + (high_f - low_f) / 2.0)  # Bin center
                    bin_median_errors.append(np.median(errors_in_bin))
                    bin_std_errors.append(np.std(errors_in_bin))

            if len(self.flux_bins_centers) < 2:  # Need at least two points for interpolation
                raise ValueError(
                    f"Could not create enough valid bins ({len(self.flux_bins_centers)}) "
                    f"for interpolation with min_samples_per_bin={min_samples_per_bin}. "
                    "Try reducing num_bins, adjusting flux_bins, or min_flux_for_binning."
                )
        else:
            self.flux_bins_centers = flux_bins

        self.flux_bins_centers = np.array(self.flux_bins_centers)
        # Store the flux range for which the model is considered valid
        self._min_interp_flux = self.flux_bins_centers[0]
        self._max_interp_flux = self.flux_bins_centers[-1]

        self.flux_unit = flux_unit
        self.min_flux_error = min_flux_error if min_flux_error is not None else 0.0

        # Use 'bounds_error=False' and 'fill_value' to handle extrapolation.
        # For sigma_sigma_X (std of errors), it should not be negative.
        # We use the value from the closest bin if extrapolating.

        self.mu_sigma_interpolator_clip: Callable[
            [Union[float, np.ndarray]], Union[float, np.ndarray]
        ] = interp1d(
            self.flux_bins_centers,
            bin_median_errors,
            kind="linear",
            bounds_error=False,
            fill_value=(bin_median_errors[0], bin_median_errors[-1]),
        )
        mu_sigma_interpolator_extrap: Callable[
            [Union[float, np.ndarray]], Union[float, np.ndarray]
        ] = interp1d(
            self.flux_bins_centers,
            bin_median_errors,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )

        def mu_sigma_interpolator(flux_values):
            output = np.empty_like(flux_values, dtype=float)
            extrapolate_mask = flux_values > self._max_interp_flux
            output[~extrapolate_mask] = self.mu_sigma_interpolator_clip(
                flux_values[~extrapolate_mask]
            )
            output[extrapolate_mask] = mu_sigma_interpolator_extrap(flux_values[extrapolate_mask])
            return output

        self.mu_sigma_interpolator_extrap = mu_sigma_interpolator
        self.mu_sigma_interpolator = self.mu_sigma_interpolator_clip

        self.sigma_sigma_interpolator_clip: Callable[
            [Union[float, np.ndarray]], Union[float, np.ndarray]
        ] = interp1d(
            self.flux_bins_centers,
            bin_std_errors,
            kind="linear",
            bounds_error=False,
            fill_value=(bin_std_errors[0], bin_std_errors[-1]),
        )
        sigma_sigma_interpolator_extrap: Callable[
            [Union[float, np.ndarray]], Union[float, np.ndarray]
        ] = interp1d(
            self.flux_bins_centers,
            bin_std_errors,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )

        def sigma_sigma_interpolator(flux_values):
            output = np.empty_like(flux_values, dtype=float)
            extrapolate_mask = flux_values > self._max_interp_flux
            output[~extrapolate_mask] = self.sigma_sigma_interpolator_clip(
                flux_values[~extrapolate_mask]
            )
            output[extrapolate_mask] = sigma_sigma_interpolator_extrap(
                flux_values[extrapolate_mask]
            )

            return output

        self.sigma_sigma_interpolator_extrap = sigma_sigma_interpolator

        self.sigma_sigma_interpolator = self.sigma_sigma_interpolator_clip

        original_sigma_sigma_interpolator = self.sigma_sigma_interpolator

        def non_negative_sigma_sigma_interpolator(flux_values):
            std_devs = original_sigma_sigma_interpolator(flux_values)
            if isinstance(std_devs, np.ndarray):
                std_devs[std_devs < 0] = 0
            elif std_devs < 0:  # scalar
                std_devs = 0
            return std_devs

        self.sigma_sigma_interpolator = non_negative_sigma_sigma_interpolator

    def apply_noise_to_flux(
        self,
        true_flux: Union[float, np.ndarray],
        true_flux_units: Optional[str] = None,
        out_units: Optional[str] = None,
        asinh_softening_parameter: Optional[float] = None,
    ) -> Tuple[Union[float, np.ndarray, None], Union[float, np.ndarray, None]]:
        """Applies noise to a photometric measurement.

        Parameters
        ----------
        true_flux : float or np.ndarray
            The true flux value(s) to which noise will be applied.
        true_flux_units : str, optional
            The units of the true flux. If None, defaults to the object's flux_unit.
        out_units : str, optional
            The units to return the noisy flux in. If None, defaults to the object's
            flux_unit.
            Can be "AB", "asinh" or any valid unyt unit.
        asinh_softening_parameter : float, optional
            If out_units is "asinh", this parameter is used to soften the
            asinh transformation.

        """
        true_flux = copy.deepcopy(true_flux)

        if out_units == "asinh":
            assert asinh_softening_parameter is not None, (
                "If out_units is 'asinh', asinh_softening_parameter must be provided."
            )

        if self.flux_unit != true_flux_units:
            if self.flux_unit == "AB":
                if not isinstance(true_flux, unyt_array):
                    true_flux = unyt_array(true_flux, units=true_flux_units).to("Jy").value
                else:
                    true_flux = true_flux.to("Jy").value

                true_flux = -2.5 * np.log10(true_flux) + 8.9
            else:
                if true_flux_units == "AB":
                    true_flux = 10 ** (-0.4 * (true_flux - 8.9)) * Jy
                    true_flux = true_flux.to(self.flux_unit).value

                else:
                    true_flux = (
                        unyt_array(true_flux, units=true_flux_units).to(self.flux_unit).value
                    )

        is_scalar = np.isscalar(true_flux)
        flux_array = np.atleast_1d(true_flux)

        flux_array = np.array(flux_array, dtype=float)  # Ensure it's a float array for calculations

        sampled_sigma_prime = self.sample_uncertainty(
            flux_array,
        )

        # Apply minimum flux error
        sampled_sigma_prime[sampled_sigma_prime < self.min_flux_error] = self.min_flux_error

        # If upper limit, calculate a mask here so we don't apply crazy 10+ mag scatters
        if self.upper_limits:
            if self.flux_unit == "AB":
                # Convert to Jy for upper limit calculations
                temp_flux_array = 10 ** (-0.4 * (flux_array - 8.9)) * Jy
                temp_sig = (np.log(10) * temp_flux_array * sampled_sigma_prime) / 2.5
            else:
                temp_flux_array = unyt_array(flux_array, units=self.flux_unit).to("Jy").value
                temp_sig = unyt_array(sampled_sigma_prime, units=self.flux_unit).to("Jy").value
            snr = (temp_flux_array / temp_sig).value.astype(float)
            print(type(snr), snr.dtype, snr.shape)
            print(self.treat_as_upper_limits_below, type(self.treat_as_upper_limits_below))
            umask = (snr < self.treat_as_upper_limits_below) | (
                np.isnan(temp_sig) | np.isnan(temp_flux_array)
            )
            # print(umask)

        else:
            umask = np.zeros_like(flux_array, dtype=bool)  # No upper limit mask if not set

        noisy_flux_array = flux_array.copy()
        noisy_flux_array[~umask] = flux_array[~umask] + stats.truncnorm.rvs(
            loc=0,
            scale=sampled_sigma_prime[~umask],
            a=-self.sigma_clip,
            b=self.sigma_clip,
        )
        # Ensure noise is not below the minimum error
        noisy_flux_array[np.isnan(sampled_sigma_prime)] = np.nan  # Handle NaNs from sampling

        if self.error_type == "observed":
            # Re-estimate the errors based on the scattered fluxes
            sampled_sigma_prime = self.sample_uncertainty(
                noisy_flux_array,
            )

        if self.upper_limits:
            # If upper limits are set, treat fluxes below the threshold as upper limits
            # Calculate SNR, and apply upper limit condition.
            snr_limit = self.treat_as_upper_limits_below

            if self.flux_unit == "AB":
                # print('AB:', noisy_flux_array, sampled_sigma_prime)
                temp_flux_array = 10 ** (-0.4 * (noisy_flux_array - 8.9)) * Jy
                # Convert error back into Jy correctly
                temp_sigma_prime = (np.log(10) * temp_flux_array * sampled_sigma_prime) / 2.5
            else:
                temp_flux_array = unyt_array(noisy_flux_array, units=self.flux_unit).to("Jy").value
                temp_sigma_prime = (
                    unyt_array(sampled_sigma_prime, units=self.flux_unit).to("Jy").value
                )

            snr = temp_flux_array / temp_sigma_prime
            m = snr < snr_limit

            # print(m, snr, temp_flux_array, temp_sigma_prime)

            m = (
                m
                | umask
                | ~np.isfinite(temp_flux_array)
                | np.isinf(noisy_flux_array)
                | np.isnan(temp_sigma_prime)
            )  # Ensure we mask out NaNs and infinities

            # Set upper limit flux behaviour
            if self.upper_limit_flux_behaviour == "scatter_limit":
                scatter_lim = self.sigma_sigma_interpolator(self.upper_limit_value)
                samples = stats.truncnorm.rvs(loc=0, scale=scatter_lim, a=-3, b=3, size=np.sum(m))
                noisy_flux_array[m] = self.upper_limit_value + samples
            elif self.upper_limit_flux_behaviour == "upper_limit":
                noisy_flux_array[m] = self.upper_limit_value
            elif isinstance(self.upper_limit_flux_behaviour, (int, float)):
                # If it is a float, use it as the upper limit value
                noisy_flux_array[m] = float(self.upper_limit_flux_behaviour)
            else:
                raise ValueError(
                    f"""Unknown upper_limit_flux_behaviour:
                    {self.upper_limit_flux_behaviour}
                    Must be 'scatter_limit', 'upper_limit', or a float/int value."""
                )

            # Set upper limit flux error behaviour
            if self.upper_limit_flux_err_behaviour == "flux":
                val_lim = self.mu_sigma_interpolator(
                    self.upper_limit_value
                )  # Use the interpolated sigma for the upper limit value but include scatter
                sampled_sigma_prime[m] = val_lim
            elif self.upper_limit_flux_err_behaviour == "upper_limit":
                sampled_sigma_prime[m] = self.upper_limit_value
            elif self.upper_limit_flux_err_behaviour == "max":
                sampled_sigma_prime[m] = self.max_flux_error
            elif self.upper_limit_flux_err_behaviour.startswith("sig_"):
                sig_val = float(self.upper_limit_flux_err_behaviour.split("_")[1])

                if self.flux_unit == "AB":
                    # val is in Jy, convert to AB
                    val = 2.5 / (sig_val * np.log(10))
                else:
                    val = self.snr_interpolator(sig_val)
                    # Find the error at this value
                    val = self.mu_sigma_interpolator(val)
                sampled_sigma_prime[m] = val

            else:
                raise ValueError(
                    f"""Unknown upper_limit_flux_err_behaviour:
                    {self.upper_limit_flux_err_behaviour}. Must be 'flux',
                    'upper_limit', 'max', or 'sig_{val}'."""
                )

        # convert back to original units if necessary

        if out_units is None:
            out_units = true_flux_units

        if self.flux_unit != out_units:
            if self.flux_unit == "AB":
                raise NotImplementedError()
            else:
                noisy_flux_array = unyt_array(noisy_flux_array, units=self.flux_unit)
                sampled_sigma_prime = unyt_array(sampled_sigma_prime, units=self.flux_unit)

                if out_units == "AB":
                    # Convert to AB magnitude
                    sampled_sigma_prime = (
                        -2.5
                        * noisy_flux_array.to(Jy).value
                        / (np.log(10) * sampled_sigma_prime.to(Jy).value)
                    )
                    noisy_flux_array = -2.5 * np.log10(noisy_flux_array.to(Jy).value) + 8.9
                elif out_units == "asinh":
                    if isinstance(asinh_softening_parameter, unyt_array):
                        f_b = asinh_softening_parameter.to(Jy)
                    else:
                        # Assume it is a sigma value, and calculate
                        # error in the same way as we calculated the SNR
                        # before.
                        f_b = [self.snr_interpolator(i) for i in asinh_softening_parameter]
                        f_b = unyt_array(f_b, units=self.flux_unit).to(Jy)

                    # Convert to asinh magnitude
                    noisy_flux_array = f_jy_err_to_asinh(
                        sampled_sigma_prime.to(Jy),
                        noisy_flux_array.to(Jy),
                        f_b=f_b,
                    )
                    sampled_sigma_prime = f_jy_to_asinh(
                        sampled_sigma_prime.to(Jy),
                        f_b=f_b,
                    )

                else:
                    sampled_sigma_prime = sampled_sigma_prime.to(true_flux_units).value
                    noisy_flux_array = noisy_flux_array.to(true_flux_units).value

        # Apply min/max in original/output units.
        sampled_sigma_prime[sampled_sigma_prime < self.min_flux_error] = self.min_flux_error
        sampled_sigma_prime[sampled_sigma_prime > self.max_flux_error] = self.max_flux_error

        if is_scalar:
            return noisy_flux_array[0], sampled_sigma_prime[0]
        return noisy_flux_array, sampled_sigma_prime

    def serialize_to_hdf5(
        self,
        hdf5_file: str,
        group_name: str = "empirical_uncertainty_model",
        overwrite: bool = False,
    ) -> None:
        """Serializes the model to an HDF5 file.

        Args:
            hdf5_file: Path to the HDF5 file where the model will be saved.
            group_name: Name of the group in the HDF5 file where the model will be stored.
            overwrite: If True, overwrite the existing group if it exists.
        """
        import h5py

        with h5py.File(hdf5_file, "a") as f:
            if overwrite and group_name in f:
                del f[group_name]
            group = f.create_group(group_name)
            group.create_dataset("flux_bins_centers", data=self.flux_bins_centers)
            group.create_dataset(
                "mu_sigma_values",
                data=self.mu_sigma_interpolator(self.flux_bins_centers),
            )
            group.create_dataset(
                "sigma_sigma_values",
                data=self.sigma_sigma_interpolator(self.flux_bins_centers),
            )
            group.attrs["num_bins"] = len(self.flux_bins_centers)
            group.attrs["flux_unit"] = self.flux_unit
            group.attrs["min_flux_error"] = self.min_flux_error
            group.attrs["max_flux_error"] = self.max_flux_error
            group.attrs["upper_limits"] = self.upper_limits
            group.attrs["error_type"] = self.error_type
            group.attrs["sigma_clip"] = self.sigma_clip
            group.attrs["min_samples_per_bin"] = self.min_samples_per_bin
            group.attrs["log_bins"] = self.log_bins
            group.attrs["min_flux_for_binning"] = str(self.min_flux_for_binning)
            group.attrs["interpolation_flux_unit"] = self.interpolation_flux_unit

            if self.upper_limits:
                group.attrs["treat_as_upper_limits_below"] = int(self.treat_as_upper_limits_below)
                group.attrs["upper_limit_value"] = self.upper_limit_value
                group.attrs["upper_limit_flux_behaviour"] = self.upper_limit_flux_behaviour
                group.attrs["upper_limit_flux_err_behaviour"] = self.upper_limit_flux_err_behaviour

    @classmethod
    def deserialize_from_hdf5(cls, hdf5_file: str, group_name: str = "all"):
        """Deserializes the model from an HDF5 file.

        Args:
            hdf5_file: Path to the HDF5 file from which the model will be loaded.
            group_name: Name of the group in the HDF5 file where the model is stored.

        Returns:
            An instance of EmpiricalUncertaintyModel initialized with the data
            from the HDF5 file.
        """
        import h5py

        output = {}
        with h5py.File(hdf5_file, "r") as f:
            if group_name == "all":
                group_names = list(f.keys())
            else:
                group_names = [group_name]
            for group_name in group_names:
                group = f[group_name]
                flux_bins_centers = group["flux_bins_centers"][:]
                mu_sigma_values = group["mu_sigma_values"][:]
                sigma_sigma_values = group["sigma_sigma_values"][:]
                num_bins = group.attrs["num_bins"]
                flux_unit = group.attrs["flux_unit"]
                min_flux_error = group.attrs["min_flux_error"]
                max_flux_error = group.attrs["max_flux_error"]
                upper_limits = group.attrs.get("upper_limits", False)
                treat_as_upper_limits_below = group.attrs.get("treat_as_upper_limits_below", None)
                if treat_as_upper_limits_below is not None:
                    treat_as_upper_limits_below = float(treat_as_upper_limits_below)
                upper_limit_flux_behaviour = group.attrs.get(
                    "upper_limit_flux_behaviour", "scatter_limit"
                )
                upper_limit_flux_err_behaviour = group.attrs.get(
                    "upper_limit_flux_err_behaviour", "flux"
                )
                error_type = group.attrs.get("error_type", "empirical")
                sigma_clip = group.attrs.get("sigma_clip", 3.0)
                log_bins = group.attrs.get("log_bins", True)
                min_flux_for_binning = group.attrs.get("min_flux_for_binning", None)
                interpolation_flux_unit = group.attrs.get("interpolation_flux_unit", flux_unit)
                min_samples_per_bin = group.attrs.get("min_samples_per_bin", 10)

                model = cls(
                    observed_fluxes=flux_bins_centers,
                    observed_errors=np.zeros_like(flux_bins_centers),  # Dummy
                    num_bins=num_bins,
                    flux_bins=flux_bins_centers,
                    log_bins=log_bins,
                    min_samples_per_bin=min_samples_per_bin,
                    flux_unit=flux_unit,
                    min_flux_error=min_flux_error,
                    error_type=error_type,
                    sigma_clip=sigma_clip,
                    upper_limits=upper_limits,
                    treat_as_upper_limits_below=treat_as_upper_limits_below,
                    upper_limit_flux_behaviour=upper_limit_flux_behaviour,
                    upper_limit_flux_err_behaviour=upper_limit_flux_err_behaviour,
                    max_flux_error=max_flux_error,
                    interpolation_flux_unit=interpolation_flux_unit,
                    min_flux_for_binning=min_flux_for_binning,
                    already_binned=True,
                    bin_median_errors=mu_sigma_values,
                    bin_std_errors=sigma_sigma_values,
                )
                output[group_name] = model

        if len(output) == 1:
            return output[group_names[0]]

        return output


def create_uncertainity_models_from_EPOCHS_cat(
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
            noisy_flux, sampled_sigma = noise_model.apply_noise_to_flux(mag, true_flux_units="AB")

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
