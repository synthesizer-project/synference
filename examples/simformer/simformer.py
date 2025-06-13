import copy
import inspect
from typing import (
    Callable,
    Dict,
    List,
    Tuple,
    Type,
    Union,
)  # For GalaxySimulator type hints

import corner
import jax
import jax.numpy as jnp
import numpy as np
import torch
from astropy.cosmology import Cosmology, Planck18
from omegaconf import OmegaConf  # To create DictConfig-like objects if needed
from scoresbibm.methods.score_transformer import train_transformer_model
from scoresbibm.tasks.base_task import InferenceTask
from synthesizer.emission_models import (
    EmissionModel,
    TotalEmission,
)
from synthesizer.emission_models.attenuation import Calzetti2000
from synthesizer.grid import Grid
from synthesizer.instruments import FilterCollection, Instrument
from synthesizer.parametric import (
    SFH,
    Galaxy,
    Stars,
    ZDist,
)  # Need concrete SFH, ZDist classes
from unyt import (
    Jy,
    Msun,
    Myr,
    Unit,
    uJy,
    unyt_array,
    unyt_quantity,
    yr,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Helper Class for Prior ---
class GalaxyPrior:
    """
    A simple prior distribution handler for galaxy parameters.
    Samples uniformly from ranges specified for each parameter.
    """

    def __init__(
        self,
        prior_ranges: Dict[str, Tuple[float, float]],
        param_order: List[str],
    ):
        self.prior_ranges = prior_ranges
        self.param_order = param_order
        self.theta_dim = len(param_order)

        self.distributions = []
        for param_name in self.param_order:
            low, high = self.prior_ranges[param_name]
            self.distributions.append(
                torch.distributions.Uniform(
                    torch.tensor(float(low)), torch.tensor(float(high))
                )
            )

    def sample(self, sample_shape: Tuple[int], sample_lhc=False) -> torch.Tensor:
        """
        Generates samples from the prior.
        Args:
            sample_shape: A tuple containing the number of samples, e.g., (num_samples,).
        Returns:
            A PyTorch tensor of shape (num_samples, theta_dim).
        """
        if not sample_lhc:
            num_samples = sample_shape[0]
            samples_per_param = [
                dist.sample((num_samples, 1)) for dist in self.distributions
            ]
        else:
            # Use boundaries, but sample from Latin Hypercube
            from scipy.stats.qmc import LatinHypercube

            sampler = LatinHypercube(d=self.theta_dim)
            lhc_samples = sampler.random(n=sample_shape[0])
            samples_per_param = []
            for i, dist in enumerate(self.distributions):
                low, high = self.prior_ranges[self.param_order[i]]
                # Scale LHC samples to the range of the distribution
                scaled_samples = low + (high - low) * lhc_samples[:, i : i + 1]
                samples_per_param.append(
                    torch.tensor(scaled_samples, dtype=torch.float32)
                )

        return torch.cat(samples_per_param, dim=1)

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Calculates the log probability of theta under the prior.
        Assumes theta is a batch of shape (num_samples, theta_dim).
        """
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)  # Make it (1, theta_dim)

        log_probs_per_param = []
        for i, dist in enumerate(self.distributions):
            log_probs_per_param.append(dist.log_prob(theta[:, i]))

        # Sum log_probs for independent parameters
        return torch.sum(torch.stack(log_probs_per_param, dim=1), dim=1)


class GalaxySimulator(object):
    """Class for simulating photometry/spectra of galaxies.

    This class is designed to work with a grid of galaxies, an instrument,
    and an emission model. It can simulate photometry and spectra based on
    the provided star formation history (SFH) and metallicity distribution
    (ZDist) models. The class can also handle uncertainties in the photometry
    using an empirical uncertainty model.
    It supports various configurations such as normalization methods,
    output types, and inclusion of photometric errors.
    It can also apply noise models to the simulated photometry.
    """

    def __init__(
        self,
        sfh_model: Type[SFH.Common],
        zdist_model: Type[ZDist.Common],
        grid: Grid,
        instrument: Instrument,
        emission_model: EmissionModel,
        emission_model_key: str,
        emitter_params: dict = {"stellar": [], "galaxy": []},
        cosmo: Cosmology = Planck18,
        param_order: Union[None, list] = None,
        param_units: dict = {},
        param_transforms: dict[callable] = {},
        out_flux_unit: str = "nJy",
        required_keys=["redshift", "log_mass"],
        extra_functions: List[callable] = None,
        normalize_method: str = None,
        output_type: str = "photo_fnu",
        include_phot_errors: bool = False,
        set_self=False,
        depths: Union[np.ndarray, unyt_array] = None,
        depth_sigma: int = 5,
        noise_models: Union[None, Dict[str, Callable]] = None,
        fixed_params: dict = {},
        min_flux: float = 50.0,
        min_flux_unit: str = "ABmag",
    ) -> None:
        """Parameters

        ----------

        sfh_model : Type[SFH.Common]
            SFH model to use. Must be a subclass of SFH.Common.
        zdist_model : Type[ZDist.Common]
            ZDist model to use. Must be a subclass of ZDist.Common.
        grid : Grid
            Grid object to use. Must be a subclass of Grid.
        instrument : Instrument
            Instrument object to use for photometry/spectra.
            Must be a subclass of Instrument.
        emission_model : EmissionModel
            Emission model to use. Must be a subclass of EmissionModel.
        emission_model_key : str
            Emission model key to use e.g. 'total', 'intrinsic'.
        emitter_params : dict
            Dictionary of parameters to pass to the emitter.
            Keys are 'stellar' and 'galaxy'.
        cosmo : Cosmology = Planck18
            Cosmology object to use. Must be a subclass of Cosmology.
        param_order : Union[None, list] = None
            Order of parameters to use.
            If None, will use the order of the keys in the params dictionary.
        param_units : dict
            Dictionary of parameter units to use.
            Keys are the parameter names and values are the units.
        param_transforms : dict
            Dictionary of parameter transforms to use.
            Keys are the parameter names and values are the transforms functions.
            Can be used to fit say Av when model requires tau_v, or for unit conversion.
        out_flux_unit : str
            Output flux unit to use. Default is 'nJy'. Can be 'AB', 'Jy', 'nJy', 'uJy',...
        required_keys : list
            List of required keys to use. Default is ['redshift', 'log_mass'].
            Typically this can be ignored.
        extra_functions : List[callable]
            List of extra functions to call after the simulation.
            Inputs to each function are determined by the function signature and can be
            any of the following:
            galaxy, spec, fluxes, sed, stars, emission_model, cosmo.
            The output of each function is appended to an array
            and returned in the output tuple.
        normalize_method : str
            Normalization method to use. If None, no normalization is applied.
            Only works on photo_fnu currently.
            Currently can be either the name of a filter, or a callable function
            with the same rules as extra_functions.
        output_type : str
            Output type to use. Default is 'photo_fnu'. Can be 'photo_fnu', 'photo_lnu',
              'fnu', 'lnu', or a list of any of these.
        include_phot_errors : bool
            Whether to include photometric errors in the output. Default is False.
        depths : Union[np.ndarray, unyt_array] = None
            Depths to use for the photometry. If None, no depths are applied.
            Default is None.
        depth_sigma : int
            Sigma to use for the depths. Default is 5.
            This is used to calculate the 1 sigma error in nJy.
            If depths is None, this is ignored.
        noise_models : Union[None, Dict[str, EmpiricalUncertaintyModel]] = None
            List of noise models to use for the photometry. If None, no noise is applied.
            Default is None.
        fixed_params : dict
            Dictionary of fixed parameters to use.
            Keys are the parameter names and values are the fixed values.
            This is used to fix parameters in the simulation.
            If None, no parameters are fixed. Default is None.
        min_flux : float
            Minimum flux to use for the photometry. If the flux is below this value,
            it is set to this value.
            This is used to avoid negative fluxes in the photometry. Default is 50.
        min_flux_unit : str
            Unit of the minimum flux. Default is 'ABmag'.
             Can be 'ABmag', 'Jy', 'nJy', 'uJy', 'mJy'.
            This is used to convert the minimum flux to the output flux unit.

        """
        assert isinstance(grid, Grid), (
            f"Grid must be a subclass of Grid. Got {type(grid)} instead."
        )
        self.grid = grid

        assert isinstance(
            instrument, Instrument
        ), f"""Instrument must be a subclass of Instrument.
            Got {type(instrument)} instead."""
        assert isinstance(
            emission_model, EmissionModel
        ), f"""Emission model must be a subclass of EmissionModel.
            Got {type(emission_model)} instead."""

        self.emission_model = emission_model
        self.emission_model_key = emission_model_key
        self.instrument = instrument
        self.cosmo = cosmo
        self.param_order = param_order
        self.param_units = param_units
        self.param_transforms = param_transforms
        self.out_flux_unit = out_flux_unit
        self.required_keys = required_keys
        self.extra_functions = extra_functions
        self.normalize_method = normalize_method
        self.fixed_params = fixed_params
        self.min_flux = min_flux
        self.min_flux_unit = min_flux_unit

        if noise_models is not None:
            assert isinstance(noise_models, dict), (
                f"Noise models must be a dictionary. Got {type(noise_models)} instead."
            )
            # Check all filters in noise_models are in the instrument filters
            for filter_code in instrument.filters.filter_codes:
                if filter_code not in noise_models:
                    raise ValueError(
                        f"""Filter {filter_code} not found in noise models.
                          Cannot apply noise models to photometry."""
                    )
        self.noise_models = noise_models
        self.include_phot_errors = include_phot_errors

        assert not (
            self.depths is not None and self.noise_models is not None
        ), """Cannot use both depths and noise models at
              the same time. Choose one or the other."""
        assert not (
            self.depths is None and self.noise_models is None and self.include_phot_errors
        ), """Cannot include photometric errors without depths or noise models.
            Set include_phot_errors to False or provide depths or noise models."""

        if depths is not None:
            assert len(depths) == len(
                instrument.filters.filter_codes
            ), f"""Depths array length {len(depths)} does not match number of filters
                {len(instrument.filters.filter_codes)}. Cannot create photometry."""
        self.depths = depths
        self.depth_sigma = depth_sigma

        assert isinstance(output_type, list) or output_type in [
            "photo_fnu",
            "photo_lnu",
            "fnu",
            "lnu",
        ], f"""Output type {output_type} not recognised.
              Must be one of ['photo_fnu', 'photo_lnu', 'fnu', 'lnu']"""
        if not isinstance(output_type, list):
            output_type = [output_type]
        self.output_type = output_type

        self.sfh_model = sfh_model
        sig = inspect.signature(sfh_model).parameters

        self.sfh_params = []
        self.optional_sfh_params = []

        for key in sig.keys():
            if sig[key].default != inspect._empty:
                self.optional_sfh_params.append(key)
            else:
                self.sfh_params.append(key)

        self.zdist_model = zdist_model

        sig = inspect.signature(zdist_model).parameters
        self.zdist_params = []
        self.optional_zdist_params = []
        for key in sig.keys():
            if sig[key].default != inspect._empty:
                self.optional_zdist_params.append(key)
            else:
                self.zdist_params.append(key)

        self.emitter_params = emitter_params
        self.emission_model.save_spectra(emission_model_key)

        self.total_possible_keys = (
            self.sfh_params
            + self.zdist_params
            + self.optional_sfh_params
            + self.optional_zdist_params
            + required_keys
        )

    def simulate(self, params):
        """Simulate photometry from the given parameters.

        Parameters
        ----------
        params : dict or array-like
            Dictionary of parameters or an array-like object with the parameters
            in the order specified by self.param_order.
            If self.param_order is None, params must be a dictionary.

        Returns:
        -------
        dict
            Dictionary of photometry outputs. Keys are the output types specified
            in self.output_type. Values are the corresponding photometry arrays.
        """
        params = copy.deepcopy(params)
        params.update(self.fixed_params)

        if not isinstance(params, dict):
            if self.param_order is None:
                raise ValueError(
                    """simulate() input requires a dictionary unless param_order is set.
                      Cannot create photometry."""
                )
            assert len(params) == len(
                self.param_order
            ), f"""Parameter array length {len(params)} does not match parameter order
                  length {len(self.param_order)}. Cannot create photometry."""
            params = {i: j for i, j in zip(self.param_order, params)}

        for key in self.required_keys:
            if key not in params:
                raise ValueError(
                    f"Missing required parameter {key}. Cannot create photometry."
                )

        mass = 10 ** params["log_mass"] * Msun

        # Check if we have sfh_params and zdist_params

        for key in params:
            if key in self.param_units:
                params[key] = params[key] * self.param_units[key]

        for key in self.param_transforms:
            if key in params:
                params[key] = self.param_transforms[key](params[key])

        sfh = self.sfh_model(
            **{i: params[i] for i in self.sfh_params},
            **{i: params[i] for i in self.optional_sfh_params if i in params},
        )
        zdist = self.zdist_model(
            **{i: params[i] for i in self.zdist_params},
            **{i: params[i] for i in self.optional_zdist_params if i in params},
        )

        # Get param names which aren't in the sfh or zdist models or the required keys
        param_names = [i for i in params.keys() if i not in self.total_possible_keys]

        # Check if any param names named here are in emitter param dictionry lusts

        found_params = []
        for key in param_names:
            found = False
            for param in self.emitter_params:
                if key in self.emitter_params[param]:
                    found = True
                    break
            if not found:
                raise ValueError(
                    f"""Parameter {key} not found in emitter params.
                    Cannot create photometry."""
                )

            else:
                found_params.append(key)

        # Check we understand all the parameters

        assert len(found_params) == len(
            self.emitter_params
        ), f"""Found {len(found_params)} parameters but expected
            {len(self.emitter_params)}. Cannot create photometry."""

        stellar_keys = {}
        if "stellar" in self.emitter_params:
            for key in found_params:
                stellar_keys[key] = params[key]

        # print(type(stellar_keys['tau_v']))
        # stellar_keys['tau_v'] = 0.0
        # print(stellar_keys)

        stars = Stars(
            log10ages=self.grid.log10ages,
            metallicities=self.grid.metallicity,
            sf_hist=sfh,
            metal_dist=zdist,
            initial_mass=mass,
            **stellar_keys,
        )

        galaxy_keys = {}
        if "galaxy" in self.emitter_params:
            for key in found_params:
                galaxy_keys[key] = params[key]

        galaxy = Galaxy(
            stars=stars,
            redshift=params["redshift"],
            **galaxy_keys,
        )

        # Get the spectra for the galaxy
        spec = galaxy.stars.get_spectra(self.emission_model)
        outputs = {}

        if "sfh" in self.output_type:
            stars_sfh = stars.get_sfh()
            stars_sfh = stars_sfh / np.diff(10 ** (self.grid.log10age), prepend=0) / yr
            time = 10 ** (self.grid.log10age) * yr
            time = time.to("Myr")

            # Check if any NANS in SFH. If there are, print sfh
            # if np.isnan(sfr).any():
            #    print(f"SFH has NANS: {sfh.parameters}")

            outputs["sfh"] = stars_sfh
            outputs["sfh_time"] = time
            outputs["redshift"] = galaxy.redshift
            # Put an absolute SFH time here as well given self.cosmo and redshift
            outputs["sfh_time_abs"] = (
                self.cosmo.age(galaxy.redshift).to("Myr").value * Myr
            )
            outputs["sfh_time_abs"] = outputs["sfh_time_abs"] - time

        if "lnu" in self.output_type:
            fluxes = spec.lnu
            outputs["lnu"] = fluxes

        if "photo_lnu" in self.output_type:
            fluxes = galaxy.stars.spectra[self.emission_model_key].get_photo_lnu(
                self.instrument.filters
            )
            fluxes = fluxes.photo_lnu
            outputs["photo_lnu"] = fluxes

        if "fnu" in self.output_type or "photo_fnu" in self.output_type:
            # Apply IGM and distance
            galaxy.get_observed_spectra(self.cosmo)

            if "photo_fnu" in self.output_type:
                fluxes = galaxy.stars.spectra[self.emission_model_key].get_photo_fnu(
                    self.instrument.filters
                )
                outputs["photo_fnu"] = fluxes.photo_fnu
                outputs["photo_wav"] = fluxes.filters.pivot_lams

                fluxes = galaxy.stars.spectra[self.emission_model_key].fnu
                outputs["fnu"] = fluxes
                outputs["fnu_wav"] = galaxy.stars.spectra[self.emission_model_key].lam * (
                    1 + galaxy.redshift
                )

        if self.out_flux_unit == "AB":

            def convert(f):
                return -2.5 * np.log10(f.to(Jy).value) + 8.9

            if "photo_fnu" in self.output_type:
                fluxes = convert(outputs["photo_fnu"])
                outputs["photo_fnu"] = fluxes
            if "fnu" in self.output_type:
                fluxes = convert(outputs["fnu"])
                outputs["fnu"] = fluxes
        else:
            if "photo_fnu" in self.output_type:
                outputs["photo_fnu"] = fluxes.to(self.out_flux_unit).value
            if "fnu" in self.output_type:
                outputs["fnu"] = fluxes.to(self.out_flux_unit).value

        if len(self.output_type) == 1:
            fluxes = outputs[self.output_type[0]]
        else:
            outputs["filters"] = self.instrument.filters
            return outputs

        def inspect_func(func, locals):
            parameters = inspect.signature(func).parameters
            inputs = {}
            possible_inputs = [
                "galaxy",
                "spec",
                "fluxes",
                "sed",
                "stars",
                "emission_model",
                "cosmo",
            ]
            for key in parameters:
                if key in possible_inputs:
                    if hasattr(self, key):
                        inputs[key] = getattr(self, key)
                    else:
                        inputs[key] = locals[key]
                else:
                    inputs[key] = params[key]
            return inputs

        if self.extra_functions is not None:
            output = []
            for func in self.extra_functions:
                if isinstance(func, tuple):
                    func, args = func
                    output.append(func(galaxy, *args))
                else:
                    inputs = inspect_func(func, locals())
                    output.append(func(**inputs))

        fluxes, errors = self._scatter(fluxes, flux_units=self.out_flux_unit)

        if self.normalize_method is not None:
            if callable(self.normalize_method):
                args = inspect_func(self.normalize_method, locals())
                norm = self.normalize_method(**args)
                if isinstance(norm, dict):
                    if self.emission_model_key in norm:
                        norm = norm[self.emission_model_key]
            else:
                norm = self.normalize_method

            fluxes = self._normalize(fluxes, method=norm, norm_unit=self.out_flux_unit)

        if self.include_phot_errors:
            fluxes = np.concatenate((fluxes, errors))

        return fluxes

    def _normalize(self, fluxes, method=None, norm_unit="AB", add_norm_pos=-1):
        if method is None:
            return fluxes

        if norm_unit == "AB":
            func = np.subtract
        else:
            func = np.divide

        if isinstance(method, str):
            if method in self.instrument.filters.filter_codes:
                # Get position of filter in filter codes
                filter_pos = self.instrument.filters.filter_codes.index(method)
                norm = fluxes[filter_pos]
                fluxes = func(fluxes, norm)
            else:
                raise ValueError(
                    f"""Filter {method} not found in filter codes.
                    Cannot normalize photometry."""
                )
        elif isinstance(method, (unyt_array, unyt_quantity)):
            if norm_unit == "AB":
                # Convert to AB
                method = -2.5 * np.log10(method.to(Jy).value) + 8.9

            norm = method
            fluxes = func(fluxes, norm)
        elif callable(method):
            norm = method(fluxes)
            fluxes = func(fluxes, norm)

        if add_norm_pos is not None:
            # Insert the normalization value at this position
            if add_norm_pos == -1:
                fluxes = np.append(fluxes, norm)
            else:
                fluxes = np.insert(fluxes, add_norm_pos, norm)

        if self.min_flux_unit is not None:
            if isinstance(self.min_flux_unit, Unit):
                min_flux = (
                    unyt_array(self.min_flux, self.min_flux_unit)
                    .to(self.out_flux_unit)
                    .value
                )
            elif isinstance(self.min_flux_unit, str):
                if self.min_flux_unit == "ABmag":
                    if self.out_flux_unit == "ABmag":
                        min_flux = self.min_flux
                    else:
                        min_flux = (10 ** ((self.min_flux - 23.9) / -2.5)) * uJy
                else:
                    raise ValueError(
                        f"""Minimum flux unit {self.min_flux_unit} not recognized.
                        Cannot normalize photometry."""
                    )
            else:
                raise ValueError(
                    f"""Minimum flux unit {self.min_flux_unit} not recognized.
                    Must be a string or unyt array."""
                )
            # Set any fluxes below the minimum to the minimum (inverse for ABmag)
            if self.out_flux_unit == "ABmag":
                fluxes = np.minimum(fluxes, min_flux)
            else:
                fluxes = np.maximum(fluxes, min_flux)

            print(min_flux, "min flux")

        return fluxes

    def _scatter(
        self,
        fluxes: np.ndarray,
        flux_units: str = "nJy",
    ):
        """Scatters the fluxes based on the provided depths or noise models.

        Parameters
        ----------
        fluxes : np.ndarray
            The fluxes to scatter.
        flux_units : str, optional
            The units of the fluxes. Default is 'nJy'.

        Returns:
        -------
        np.ndarray, np.ndarray
            The scattered fluxes and their corresponding errors.
            If depths are not provided, returns the original fluxes and None for errors.
        """
        if self.depths is not None:
            depths = self.depths
            if depths is None:
                return fluxes

            depths = np.array(depths)

            m = fluxes.shape

            if isinstance(fluxes, unyt_array):
                assert (
                    fluxes.units == flux_units
                ), f"""Fluxes units {fluxes.units} do not match flux units
                    {flux_units}. Cannot scatter photometry."""

            if flux_units == "AB":
                fluxes = (10 ** ((fluxes - 23.9) / -2.5)) * uJy

            # Convert depths based on units
            if self.out_flux_unit == "AB" and not hasattr(depths, "unit"):
                # Convert from AB magnitudes to microjanskys
                depths_converted = (10 ** ((depths - 23.9) / -2.5)) * uJy
                depths_std = depths_converted.to(uJy).value / self.depth_sigma
            else:
                depths_std = depths.to(self.out_flux_unit).value / self.depth_sigma
            # Pre-allocate output array with correct dimensions
            output_arr = np.zeros(m)

            # Generate all random values at once for better performance
            random_values = np.random.normal(loc=0, scale=depths_std, size=(m)) * uJy
            # Add the random values to the fluxes
            output_arr = fluxes + random_values

            errors = depths_std

            if flux_units == "AB":
                # Convert back to AB magnitudes
                output_arr = -2.5 * np.log10(output_arr.to(Jy).value) + 8.9
                errors = -2.5 * depths_std / (np.log(10) * fluxes.to(uJy).value)

            return output_arr, errors

        elif self.noise_models is not None:
            # Apply noise models to the fluxes
            scattered_fluxes = np.zeros_like(fluxes, dtype=float)
            errors = np.zeros_like(fluxes, dtype=float)
            for i, filter_code in enumerate(self.instrument.filters.filter_codes):
                noise_model = self.noise_models.get(filter_code)
                flux = fluxes[i]

                scattered_flux, sigma = noise_model.apply_noise_to_flux(
                    true_flux=flux,
                    true_flux_units=flux_units,
                    out_units=self.out_flux_unit,
                )

                scattered_fluxes[i] = scattered_flux
                errors[i] = sigma

            return scattered_fluxes, errors

        else:
            # No depths or noise models, return the fluxes as is
            return fluxes, None

    def __call__(self, params):
        """Call the simulator with parameters to get photometry."""
        return self.simulate(params)

    def __repr__(self):
        """String representation of the PhotometrySimulator."""
        return f"""PhotometrySimulator({self.sfh_model},
                                        {self.zdist_model},
                                        {self.grid},
                                        {self.instrument},
                                        {self.emission_model},
                                        {self.emission_model_key})"""

    def __str__(self):
        """String representation of the PhotometrySimulator."""
        return self.__repr__()


# --- Custom Simformer Task ---
class GalaxyPhotometryTask(InferenceTask):
    """
    A Simformer InferenceTask for the GalaxySimulator.
    """

    _theta_dim: int
    _x_dim: int

    def __init__(
        self,
        name: str = "galaxy_photometry_task",
        backend: str = "jax",
        prior_dict: Dict[str, Tuple[float, float]] = None,
        param_names_ordered: List[str] = None,
        run_simulator_fn: Callable = None,
        num_filters: int = None,
    ):
        super().__init__(name, backend)

        if (
            prior_dict is None
            or param_names_ordered is None
            or run_simulator_fn is None
            or num_filters is None
        ):
            raise ValueError(
                """prior_dict, param_names_ordered, run_simulator_fn,
                and num_filters must be provided."""
            )

        self.param_names_ordered = param_names_ordered
        self._theta_dim = len(param_names_ordered)
        self._x_dim = num_filters

        self.prior_dist = GalaxyPrior(
            prior_ranges=prior_dict, param_order=self.param_names_ordered
        )
        self.run_simulator_fn = run_simulator_fn

    def get_theta_dim(self) -> int:
        return self._theta_dim

    def get_x_dim(self) -> int:
        return self._x_dim

    def get_prior(self):
        """Returns the prior distribution object."""
        return self.prior_dist

    def get_simulator(self):
        """
        Returns a callable that takes a batch of thetas and returns a batch of xs.
        The provided run_simulator_fn processes one sample at a time
        and returns a torch tensor.
        This wrapper will handle batching.
        """

        def batched_simulator(
            thetas_batch_torch: torch.Tensor,
        ) -> torch.Tensor:
            xs_list = []
            for i in range(thetas_batch_torch.shape[0]):
                theta_sample_torch = thetas_batch_torch[i, :]
                # run_simulator_fn expects numpy array if it's not a dict,
                # and handles tensor conversion internally.
                # It returns a tensor of shape [1, num_filters].
                x_sample_torch = self.run_simulator_fn(
                    theta_sample_torch, return_type="tensor"
                )
                xs_list.append(x_sample_torch)
            return torch.cat(xs_list, dim=0)  # Shape will be (num_samples, num_filters)

        return batched_simulator

    def get_data(self, num_samples: int, **kwargs) -> Dict[str, jnp.ndarray]:
        """Generates and returns a dictionary of {'theta': thetas, 'x': xs} as JAX arr"""
        prior = self.get_prior()
        simulator = self.get_simulator()  # This is our batched_simulator

        # Sample thetas (parameters) using the prior
        # GalaxyPrior.sample returns a PyTorch tensor
        thetas_torch = prior.sample((num_samples,), **kwargs)

        # Simulate xs (photometry) using the parameters
        # batched_simulator also returns a PyTorch tensor
        xs_torch = simulator(thetas_torch)

        if self.backend == "jax":
            thetas_out = jnp.array(thetas_torch.cpu().numpy())
            xs_out = jnp.array(xs_torch.cpu().numpy())
        elif self.backend == "numpy":
            thetas_out = thetas_torch.cpu().numpy()
            xs_out = xs_torch.cpu().numpy()
        else:  # "torch" or other
            thetas_out = thetas_torch
            xs_out = xs_torch

        return {"theta": thetas_out, "x": xs_out}

    def get_node_id(self) -> jnp.ndarray:
        """Returns an array identifying the nodes (dimensions) of theta and x."""
        dim = self.get_theta_dim() + self.get_x_dim()
        if self.backend == "torch":  # Should align with SBIBMTask if that's a reference
            return torch.arange(dim)
        else:  # JAX or numpy
            return jnp.arange(dim)

    def get_base_mask_fn(self) -> Callable:
        """Defines the base attention mask for the transformer."""
        theta_dim = self.get_theta_dim()
        x_dim = self.get_x_dim()

        # Parameters only attend to themselves (or causal if ordered)
        thetas_self_mask = jnp.eye(theta_dim, dtype=jnp.bool_)

        # Data can attend to previous/current data points (causal within x)
        # Or use jnp.ones if full self-attention within x is desired.
        xs_self_mask = jnp.tril(jnp.ones((x_dim, x_dim), dtype=jnp.bool_))

        # Data can attend to all parameters
        xs_attend_thetas_mask = jnp.ones((x_dim, theta_dim), dtype=jnp.bool_)

        # Parameters do not attend to data
        thetas_attend_xs_mask = jnp.zeros((theta_dim, x_dim), dtype=jnp.bool_)

        base_mask = jnp.block(
            [
                [thetas_self_mask, thetas_attend_xs_mask],
                [xs_attend_thetas_mask, xs_self_mask],
            ]
        )
        base_mask = base_mask.astype(jnp.bool_)

        def base_mask_fn(node_ids, node_meta_data):
            # Handles potential permutation/subsetting of nodes
            return base_mask[jnp.ix_(node_ids, node_ids)]

        return base_mask_fn


if __name__ == "__main__":
    # Define sfh and zdist instances or classes as used by GalaxySimulator
    sfh_model_class = SFH.LogNormal
    zdist_model_class = ZDist.DeltaConstant

    # Example: Define global 'device' if not already defined
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    grid_dir = "/home/tharvey/work/synthesizer_grids/"  # This path needs to be accessible
    grid_name = "bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps.hdf5"

    # --- This part needs functional grid, instrument etc. ---

    grid = Grid(grid_name, grid_dir=grid_dir)
    filter_codes = [
        "JWST/NIRCam.F090W",
        "JWST/NIRCam.F115W",
        "JWST/NIRCam.F150W",
        "JWST/NIRCam.F162M",
        "JWST/NIRCam.F182M",
        "JWST/NIRCam.F200W",
        "JWST/NIRCam.F210M",
        "JWST/NIRCam.F250M",
        "JWST/NIRCam.F277W",
        "JWST/NIRCam.F300M",
        "JWST/NIRCam.F335M",
        "JWST/NIRCam.F356W",
        "JWST/NIRCam.F410M",
        "JWST/NIRCam.F444W",
    ]
    filterset = FilterCollection(filter_codes)
    instrument = Instrument("JWST", filters=filterset)
    emission_model_instance = TotalEmission(
        grid=grid,
        fesc=0.0,
        fesc_ly_alpha=0.1,
        dust_curve=Calzetti2000(),
        dust_emission_model=None,
    )
    emitter_params_dict = {"stellar": ["tau_v"]}
    galaxy_simulator_instance = GalaxySimulator(
        sfh_model=sfh_model_class,  # Pass the class
        zdist_model=zdist_model_class,  # Pass the class
        grid=grid,
        instrument=instrument,
        emission_model=emission_model_instance,
        emission_model_key="total",
        emitter_params=emitter_params_dict,
        param_units={"peak_age": Myr, "max_age": Myr},  # Ensure Myr is defined
        normalize_method=None,
        output_type="photo_fnu",
        out_flux_unit="ABmag",
    )

    inputs_list = [
        "redshift",
        "log_mass",
        "log10metallicity",
        "tau_v",
        "peak_age",
        "max_age",
        "tau",
    ]
    filter_codes_list = [
        "JWST/NIRCam.F090W",
        "JWST/NIRCam.F115W",
        "JWST/NIRCam.F150W",
        "JWST/NIRCam.F162M",
        "JWST/NIRCam.F182M",
        "JWST/NIRCam.F200W",
        "JWST/NIRCam.F210M",
        "JWST/NIRCam.F250M",
        "JWST/NIRCam.F277W",
        "JWST/NIRCam.F300M",
        "JWST/NIRCam.F335M",
        "JWST/NIRCam.F356W",
        "JWST/NIRCam.F410M",
        "JWST/NIRCam.F444W",
    ]

    priors_ranges_dict = {
        "redshift": (5.0, 10.0),
        "log_mass": (7.0, 11.0),
        "log10metallicity": (-3.0, -1.3),
        "tau_v": (0.0, 2),
        "peak_age": (
            0.0,
            500.0,
        ),  # Ensure float for peak_age if used with torch.tensor(float(low))
        "max_age": (500.0, 1000.0),
        "tau": (0.3, 1.5),
    }

    def run_simulator_glob(params, return_type="tensor"):
        if isinstance(params, torch.Tensor):
            params = params.cpu().numpy()
        if isinstance(params, dict):
            pass  # assumes params are correctly keyed
        elif isinstance(params, (list, tuple, np.ndarray)):
            params = np.squeeze(params)
            params = {inputs_list[i]: params[i] for i in range(len(inputs_list))}

        phot = galaxy_simulator_instance(
            params
        )  # This line requires galaxy_simulator_instance

        if return_type == "tensor":
            return torch.tensor(phot[np.newaxis, :], dtype=torch.float32).to(device)
        else:
            return phot

    galaxy_task = GalaxyPhotometryTask(
        prior_dict=priors_ranges_dict,
        param_names_ordered=inputs_list,
        run_simulator_fn=run_simulator_glob,
        num_filters=len(filter_codes_list),
    )

    # Test data generation
    print(f"Theta dim: {galaxy_task.get_theta_dim()}")
    print(f"X dim: {galaxy_task.get_x_dim()}")
    data_batch = galaxy_task.get_data(num_samples=3)
    print("Sampled theta (JAX):", data_batch["theta"])
    print("Shape of theta:", data_batch["theta"].shape)
    print("Sampled x (JAX):", data_batch["x"])
    print("Shape of x:", data_batch["x"].shape)

    # Test prior sampling directly
    prior_for_test = galaxy_task.get_prior()
    theta_samples_torch = prior_for_test.sample((2,))
    print("Direct prior samples (Torch):", theta_samples_torch)
    print(
        "Log prob of prior samples:",
        prior_for_test.log_prob(theta_samples_torch),
    )

    # Test base mask function
    mask_fn = galaxy_task.get_base_mask_fn()
    node_ids_example = jnp.arange(galaxy_task.get_theta_dim() + galaxy_task.get_x_dim())
    applied_mask = mask_fn(node_ids=node_ids_example, node_meta_data=None)
    print("Base mask applied to node_ids:", applied_mask)

    # Model Configuration (from config/method/model/score_transformer.yaml)
    #
    model_config_dict = {
        "name": "ScoreTransformer",
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 4,
        "d_feedforward": 256,
        "dropout": 0.1,
        "max_len": 5000,  # Adjust based on theta_dim + x_dim
        "tokenizer": {"name": "LinearTokenizer", "encoding_dim": 64},
        "use_output_scale_fn": True,
        # Add other model-specific parameters as per the YAML
    }

    # SDE Configuration (e.g., from config/method/sde/vpsde.yaml)
    #
    sde_config_dict = {
        "name": "VPSDE",  # or "VESDE"
        "beta_min": 0.1,
        "beta_max": 20.0,
        "num_steps": 1000,
        "T_min": 1e-05,
        "T_max": 1.0,
        # "likelihood_weighting": False,
        # "schedule_name": "linear",
        # Add other SDE-specific parameters
    }

    # Training Configuration (from config/method/train/train_score_transformer.yaml)
    #
    train_config_dict = {
        "learning_rate": 1e-4,  # Initial learning rate for training # used
        "min_learning_rate": 1e-6,  # Minimum learning rate for training # used
        "z_score_data": True,  # Whether to z-score the data # used
        "total_number_steps_scaling": 5,  # Scaling factor for total number of steps
        "max_number_steps": 1e8,  # Maximum number of steps for training # used
        "min_number_steps": 1e4,  # Minimum number of steps for training # used
        "training_batch_size": 64,  # Batch size for training # used
        "val_every": 100,  # Validate every 100 steps # used
        "clip_max_norm": 10.0,  # Gradient clipping max norm # used
        "condition_mask_fn": {
            "name": "joint"
        },  # Use the base mask function defined in the task
        "edge_mask_fn": {"name": "none"},
        "validation_fraction": 0.1,  # Fraction of data to use for validation # used
        "val_repeat": 5,  # Number of times to repeat validation # used
        "stop_early_count": 5,  # Number of steps to wait before stopping early # used
        "rebalance_loss": False,  # Whether to rebalance the loss # used
    }

    method_config_dict = {
        "device": str(device),  # Ensure this matches device setup
        "sde": sde_config_dict,
        "model": model_config_dict,
        "train": train_config_dict,
    }

    # Convert the main method_cfg to OmegaConf DictConfig
    method_cfg = OmegaConf.create(method_config_dict)

    print("Instantiating GalaxyPhotometryTask...")
    galaxy_task = GalaxyPhotometryTask(
        prior_dict=priors_ranges_dict,
        param_names_ordered=inputs_list,
        run_simulator_fn=run_simulator_glob,  # function to run the simulator
        num_filters=len(filter_codes_list),
        backend="jax",  # Or "torch"
    )
    print("Task instantiated.")

    # --- 2. Generate Data ---
    num_training_simulations = 5000  # Example number
    print(f"Generating {num_training_simulations} training simulations...")
    # .get_data() returns a dict with JAX arrays if backend is "jax"
    training_data = galaxy_task.get_data(num_samples=num_training_simulations)
    theta_train = training_data["theta"]
    x_train = training_data["x"]
    print(f"Data generated: theta shape {theta_train.shape}, x shape {x_train.shape}")

    # (Optional) Generate validation data if train_config.val_split is 0
    num_validation_simulations = 20
    validation_data = galaxy_task.get_data(num_samples=num_validation_simulations)
    theta_val = validation_data["theta"]
    x_val = validation_data["x"]

    # --- 3. Set RNG Seed for JAX ---
    rng_seed_for_training = 0
    master_rng_key = jax.random.PRNGKey(rng_seed_for_training)

    # --- 4. Train the Model ---

    print("Starting training...")
    trained_score_model = train_transformer_model(
        task=galaxy_task,
        data=training_data,  # Expects dict {"theta": ..., "x": ...} with JAX arrays
        method_cfg=method_cfg,  # The OmegaConf object created above
        rng=master_rng_key,
    )
    print(
        "Training finished. Model returned by train_transformer_model:",
        type(trained_score_model),
    )
    plot_corner = True
    # Take test observation
    theta_dim = galaxy_task.get_theta_dim()
    x_dim = galaxy_task.get_x_dim()
    # Mask for posterior: theta is unknown (0), x is known (1)
    posterior_condition_mask = jnp.array([0] * theta_dim + [1] * x_dim, dtype=jnp.bool_)
    for i, xobs in enumerate(x_val):
        x_val = jnp.array([xobs], dtype=jnp.float32)
        samples = trained_score_model.sample_batched(
            num_samples=1000,
            x_o=x_val,
            rng=master_rng_key,
            condition_mask=posterior_condition_mask,
        )
        if plot_corner:
            import corner
            import matplotlib.pyplot as plt

            truth = jnp.array(theta_val[i], dtype=jnp.float32)
            corner.corner(
                samples,
                labels=galaxy_task.param_names_ordered,
                show_titles=True,
                truths=truth,
                quantiles=[0.16, 0.5, 0.84],
                title_kwargs={"fontsize": 12},
            )
            plt.savefig(
                f"/home/tharvey/work/ltu-ili_testing/models/simformer/plots/corner_plot_{i}.png"
            )

    import pickle

    with open("trained_galaxy_score_model_params.pkl", "wb") as f:
        pickle.dump(trained_score_model.score_model_params, f)
    print("Model parameters saved (example).")
