"""SBI Inference classes for SED fitting."""

import copy
import glob
import json
import logging
import os
import pickle
import queue
import signal
import threading
import time
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import ili
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import sbi
import tarp
import torch
import torch.nn as nn
from astropy.table import Table
from astropy.visualization import hist
from ili.dataloaders import NumpyLoader, SBISimulator
from ili.inference import InferenceRunner
from ili.utils.samplers import (
    DirectSampler,
    EmceeSampler,
    PyroSampler,
    VISampler,
)
from ili.validation.metrics import PlotSinglePosterior, PosteriorCoverage
from ili.validation.runner import ValidationRunner
from joblib import dump, load
from optuna.trial import TrialState
from scipy import stats
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm, trange
from unyt import Jy, nJy, um, unyt_array, unyt_quantity

try:  # sbi > 0.22.0
    from sbi.inference.posteriors import EnsemblePosterior  # noqa F401
except ImportError:  # sbi < 0.22.0
    pass

from . import logger
from .custom_runner import CustomIndependentUniform

# astropy, scipy, matplotlib, tqdm, synthesizer, unyt, h5py, numpy,
# ili, torch, sklearn, optuna, joblib, pandas, tarp, astropy.table
from .library import GalaxySimulator
from .noise_models import (
    AsinhEmpiricalUncertaintyModel,
    EmpiricalUncertaintyModel,
    UncertaintyModel,
    load_unc_model_from_hdf5,
    save_unc_model_to_hdf5,
)
from .utils import (
    FilterArithmeticParser,
    TimeoutException,
    analyze_feature_contributions,
    asinh_err_to_f_jy,
    asinh_to_f_jy,
    asinh_to_snr,
    compare_methods_feature_importance,
    create_database_universal,
    create_sqlite_db,
    detect_outliers,
    detect_outliers_pyod,
    f_jy_err_to_asinh,
    f_jy_to_asinh,
    load_library_from_hdf5,
    make_serializable,
    move_to_device,
    optimize_sfh_xlimit,
)

code_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sample_with_timeout(sampler, nsteps, x, timeout_seconds):
    """Sample with timeout using threading."""
    result_queue = queue.Queue()
    exception_queue = queue.Queue()

    def target():
        try:
            result = sampler.sample(nsteps=nsteps, x=x, progress=False)
            result_queue.put(result)
        except Exception as e:
            exception_queue.put(e)

    thread = threading.Thread(target=target)
    thread.daemon = True  # Dies when main thread dies
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        # Thread is still running, sampling timed out
        raise TimeoutException(f"Sampling timed out after {timeout_seconds} seconds")

    # Check if there was an exception
    if not exception_queue.empty():
        raise exception_queue.get()

    # Get the result
    if not result_queue.empty():
        return result_queue.get()
    else:
        raise TimeoutException("No result returned")


class SBI_Fitter:
    """Class to fit a model to the data using the ltu-ili package.

    Datasets are loaded from HDF5 files. The data is then split into training
    and testing sets.
    Flexible models including ensembles are supported.

    name: str
        The name of the model.
    parameter_names: list
        The names of the parameters to fit.
    raw_observation_names: list
        The names of the photometry filters in the grid.
    raw_observation_grid: np.ndarray
        The raw photometry or spectral grid to use for fitting.
    parameter_array: np.ndarray
        The parameter array to use for fitting.
    raw_observation_units: list
        The units of the raw observation grid.
    simulator: callable
        The simulator function to use for generating synthetic data.
    feature_array: np.ndarray
        The feature array to use for fitting.
    feature_names: list
        The names of the features to use for fitting.
    feature_units: list
        The units of the features to use for fitting.
    library_path: str
        The path to the library file.
    supplementary_parameters: np.ndarray
        Any supplementary parameters to include in the fitting.
    supplementary_parameter_names: list
        The names of the supplementary parameters.
    supplementary_parameter_units: list
        The units of the supplementary parameters.
    device: str
        The device to use for fitting. Default is 'cuda' if available,
        otherwise 'cpu'.
    observation_type: str
        The type of model to use for fitting. Can be 'photometry' or 'spectra'.
        Default is 'photometry'.

    """

    device = "cuda" if hasattr(torch, "cuda") and torch.cuda.is_available() else "cpu"

    def __init__(
        self,
        name: str,
        parameter_names: list,
        raw_observation_names: list,
        raw_observation_grid: np.ndarray = None,
        parameter_array: np.ndarray = None,
        parameter_units: list = None,
        raw_observation_units: list = nJy,
        simulator: callable = None,
        feature_array: np.ndarray = None,
        feature_names: list = None,
        feature_units: list = None,
        library_path: str = None,
        supplementary_parameters: np.ndarray = None,
        supplementary_parameter_names: list = None,
        supplementary_parameter_units: list = None,
        device: str = device,
        observation_type: str = "photometry",
    ) -> None:
        """Class for SBI Fitting.

        Description:
        name: str
            The name of the model.
        parameter_names: list
            The names of the parameters to fit.
        raw_observation_names: list
            The names of the photometry filters in the grid.
        raw_observation_grid: np.ndarray
            The raw observation grid to use for fitting.
        parameter_array: np.ndarray
            The parameter array to use for fitting.
        raw_observation_units: list
            The units of the raw observation grid.
        simulator: callable
            The simulator function to use for generating synthetic data.
        feature_array: np.ndarray
            The feature array to use for fitting.
        feature_names: list
            The names of the features to use for fitting.
        feature_units: list
            The units of the features to use for fitting.
        library_path: str
            The path to the library file.
        supplementary_parameters: np.ndarray
            Any supplementary parameters to include in the fitting.
        supplementary_parameter_names: list
            The names of the supplementary parameters.
        supplementary_parameter_units: list
            The units of the supplementary parameters.
        device: str
            The device to use for fitting. Default is 'cuda' if available,
            otherwise 'cpu'.
        observation_type: str
            The type of model to use for fitting. Can be 'photometry' or 'spectra'.
            Default is 'photometry'.

        """
        self.name = name
        self.raw_observation_grid = raw_observation_grid
        self.raw_observation_units = raw_observation_units
        self.raw_observation_names = raw_observation_names
        self.parameter_array = parameter_array
        self.parameter_names = parameter_names
        self.parameter_units = parameter_units
        self.observation_type = observation_type

        self.simulator = simulator
        self.has_simulator = simulator is not None

        """
        assert (self.simulator is not None) or (self.raw_observation_grid is not None), (
            "Either a simulator or raw photometry grid must be provided."
        )
        """

        # This allows you to subset the parameters to fit
        # if you want to marginalize over some parameters.
        # See self.update_parameter_array() for more details.
        self.fitted_parameter_array = None
        self.fitted_parameter_names = parameter_names
        self.fitted_parameter_units = parameter_units

        if parameter_names is not None:
            self.simple_fitted_parameter_names = [i.split("/")[-1] for i in parameter_names]
        else:
            self.simple_fitted_parameter_names = None

        # Feature array and names
        self.feature_array = feature_array
        self.feature_names = feature_names
        self.feature_units = feature_units
        self.provided_feature_parameters = []  # This stores the parameters
        # which are provided as features e.g. redshift.
        # They are removed from the parameter array and names.

        # Supplementary parameters
        if supplementary_parameters is None:
            supplementary_parameters = []
        if supplementary_parameter_names is None:
            supplementary_parameter_names = []
        if supplementary_parameter_units is None:
            supplementary_parameter_units = []

        self.supplementary_parameters = supplementary_parameters
        self.supplementary_parameter_names = supplementary_parameter_names
        self.supplementary_parameter_units = supplementary_parameter_units

        # Grid path
        self.library_path = library_path

        self.has_features = (self.feature_array is not None) and (self.feature_names is not None)

        # This is a dictionary of flags for the feature array,
        # e.g. {'missing_flux': True}.
        self.feature_array_flags = {}

        self.posteriors = None
        self.stats = None
        self._feature_scalar = None
        self._target_scalar = None

        self._train_indices = None
        self._test_indices = None
        self._train_fraction = None
        self._train_args = None
        self._ensemble_model_types = None
        self._ensemble_model_args = None
        self._prior = None
        self._X_train = None
        self._y_train = None
        self._X_test = None
        self._y_test = None

        self.min_flux_pc_error = None
        self.phot_depths = None

        # Set the device
        if device is not None:
            self.device = device

        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "CUDA is not available. Falling back to CPU. "
                "Please check your PyTorch installation."
            )
            self.device = "cpu"

    @classmethod
    def init_from_hdf5(
        cls,
        model_name: str,
        hdf5_path: str,
        return_output=False,
        **kwargs,
    ):
        r"""Initialize the SBI fitter from an HDF5 file.

        Parameters:
            hdf5_path: Path to the HDF5 file.
            model_name: Name of the model to be used.
            return_output: If True, returns the output dictionary from the HDF5 file.
            \**kwargs: Additional keyword arguments to pass to the SBI_Fitter constructor.

        Returns:
            An instance of the SBI_Fitter class.
        """
        # Needs to load the training data and parameters from HDF5 file.
        # Training data if unnormalized and not setup as correct features yet.

        if not os.path.isfile(hdf5_path):
            if os.path.exists(f"{code_path}/libraries/{hdf5_path}"):
                hdf5_path = f"{code_path}/libraries/{hdf5_path}"

        try:
            output = load_library_from_hdf5(hdf5_path)
        except Exception as e:
            logger.warning(
                f"Error loading HDF5 file {hdf5_path}: {e} "
                "Won't be able to fit model without library."
            )
            return cls(
                name=model_name,
                raw_observation_grid=None,
                raw_observation_names=None,
                parameter_array=None,
                parameter_names=None,
                parameter_units=None,
                raw_observation_units=None,
                feature_array=None,
                feature_names=None,
                feature_units=None,
                library_path=hdf5_path,
                **kwargs,
            )

        if return_output:
            return output

        if "photometry" in output:
            observation_type = "photometry"
            raw_observation_grid = output["photometry"]

        elif "spectra" in output:
            observation_type = "spectra"
            raw_observation_grid = output["spectra"]
        else:
            raise ValueError("HDF5 file must contain 'photometry' or 'spectra' data.")
        raw_observation_names = output["filter_codes"]
        parameter_array = output["parameters"].T
        parameter_names = output["parameter_names"]
        raw_observation_units = output["photometry_units"]
        parameter_units = output["parameter_units"]

        if "supplementary_parameters" in output:
            supplementary_parameters = output["supplementary_parameters"]
            supplementary_parameter_names = output["supplementary_parameter_names"]
            supplementary_parameter_units = output["supplementary_parameter_units"]
        else:
            supplementary_parameters = None
            supplementary_parameter_names = []
            supplementary_parameter_units = []

        return cls(
            name=model_name,
            raw_observation_grid=raw_observation_grid,
            raw_observation_names=raw_observation_names,
            parameter_array=parameter_array,
            parameter_names=parameter_names,
            parameter_units=parameter_units,
            raw_observation_units=raw_observation_units,
            feature_array=None,
            feature_names=None,
            feature_units=None,
            library_path=hdf5_path,
            supplementary_parameters=supplementary_parameters,
            supplementary_parameter_names=supplementary_parameter_names,
            supplementary_parameter_units=supplementary_parameter_units,
            observation_type=observation_type,
            **kwargs,
        )

    @classmethod
    def load_saved_model(
        cls,
        model_file,
        library_path: Optional[str] = None,
        model_name: str = None,
        load_arrays: bool = True,
        **kwargs,
    ):
        """Load a prefit SBI model from a file.

        Args:
            model_file (str): Path to the saved model file.
            library_path (Optional[str], optional): Optional path to the library file.
                If not provided, it will be loaded from the model
                file parameters. Defaults to None.
            model_name (Optional[str], optional): Optional name of the model.
                If not provided, it will be loaded from the model
                file parameters. Defaults to None.
            load_arrays (bool, optional): Whether to load the feature and
                parameter arrays. Defaults to True.
            **kwargs (Any): Additional keyword arguments to pass to the
                SBI_Fitter constructor.

        Returns:
            SBI_Fitter:
                An instance of SBI_Fitter initialized with the loaded model.
        """
        if not os.path.exists(model_file):
            if os.path.exists(f"{code_path}/models/{model_file}"):
                model_file = f"{code_path}/models/{model_file}"
            else:
                raise ValueError(f"Model file {model_file} does not exist.")

        if library_path is None or model_name is None:
            level: int = logger.getEffectiveLevel()
            try:
                logger.setLevel(logging.CRITICAL)
                # The original call to load the model
                posterior, stats, params = cls.load_model_from_pkl(
                    cls, model_file, set_self=False, load_arrays=False
                )
            finally:
                # This block is guaranteed to run, restoring the original level
                logger.setLevel(level)
            if model_name is None:
                model_name = params.get("name", "default_name")

            if model_name == "default_name":
                logger.warning("Model name not found in file. Using default name 'default_name'.")

            if library_path is None:
                library_path = params.get("library_path", params.get("grid_path", None))

                if library_path is None:
                    raise ValueError("Library path not found in model file. Please provide it.")

        # Initialize the SBI_Fitter with the loaded parameters
        fitter = cls.init_from_hdf5(
            model_name=model_name, hdf5_path=library_path, return_output=False, **kwargs
        )

        fitter.load_model_from_pkl(model_file, set_self=True, load_arrays=load_arrays)

        return fitter

    def update_parameter_array(
        self,
        parameters_to_remove: list = [],
        delete_rows=[],
        n_scatters: int = 1,
        parameters_to_add: list = [],
        parameter_transformations: Dict[str, Callable] = None,
    ) -> None:
        """Updates parameter array based on feature array creation.

        This function removes parameters from the parameter array and names
        based on the provided feature parameters and parameters to remove.
        It also handles the case where the parameter array needs to be duplicated
        for multiple scatters.

        Parameters:
            parameters_to_remove: List of parameters to remove from the parameter array.
            delete_rows: List of rows to delete from the parameter array.
            n_scatters: Number of scatters to duplicate the parameter array for.
            parameters_to_add: List of parameters to add to the parameter array.
                Only parameters from self.supplementary_parameters are currently supported
        """
        if len(delete_rows) > 0:
            logger.info(f"{len(delete_rows)} rows to delete from parameter array.")
        self.fitted_parameter_array = copy.deepcopy(self.parameter_array)
        self.fitted_parameter_names = copy.deepcopy(self.parameter_names)
        self.fitted_parameter_units = copy.deepcopy(self.parameter_units)

        params = np.unique(self.provided_feature_parameters + parameters_to_remove)
        for param in params:
            if param in self.fitted_parameter_names:
                index = list(self.fitted_parameter_names).index(param)
                self.fitted_parameter_array = np.delete(self.fitted_parameter_array, index, axis=1)
                self.fitted_parameter_names = np.delete(self.fitted_parameter_names, index)
                self.simple_fitted_parameter_names = [
                    i.split("/")[-1] for i in self.fitted_parameter_names
                ]
                if self.fitted_parameter_units is not None:
                    self.fitted_parameter_units = np.delete(self.fitted_parameter_units, index)

        if len(parameters_to_add) > 0:
            # Add parameters from supplementary_parameters
            for param in parameters_to_add:
                if param in self.supplementary_parameter_names:
                    index = list(self.supplementary_parameter_names).index(param)
                    new_param = self.supplementary_parameters[index]
                    self.fitted_parameter_array = np.column_stack(
                        (self.fitted_parameter_array, new_param)
                    )
                    self.fitted_parameter_names = np.append(self.fitted_parameter_names, param)
                    self.parameter_names = np.append(self.parameter_names, param)
                    if (
                        self.fitted_parameter_units is not None
                        and self.supplementary_parameter_units is not None
                    ):
                        self.fitted_parameter_units = np.append(
                            self.fitted_parameter_units,
                            self.supplementary_parameter_units[index],
                        )
                    self.simple_fitted_parameter_names = np.append(
                        self.simple_fitted_parameter_names, param.split("/")[-1]
                    )
                else:
                    raise ValueError(
                        f"Can't add {param} to parameter array - not found in supplementary parameters."  # noqa: E501
                        f" Available parameters: {self.supplementary_parameter_names}"  # noqa: E501
                    )

        if n_scatters > 1:
            self.fitted_parameter_array = np.repeat(self.fitted_parameter_array, n_scatters, axis=0)

        # Remove any rows in delete_rows
        if len(delete_rows) > 0:
            self.fitted_parameter_array = np.delete(
                self.fitted_parameter_array, delete_rows, axis=0
            )

        # Apply any parameter transformations
        if parameter_transformations is not None:
            for param, transform in parameter_transformations.items():
                if param in self.fitted_parameter_names:
                    logger.info(
                        f"Applying {transform.__name__} transformation to parameter {param}."  # noqa: E501
                    )
                    index = list(self.fitted_parameter_names).index(param)
                    name = transform.__name__
                    self.fitted_parameter_array[:, index] = transform(
                        self.fitted_parameter_array[:, index]
                    )
                    self.fitted_parameter_names[index] = f"{name}_{param}"
                    if (
                        self.fitted_parameter_units[index] is not None
                        and str(self.fitted_parameter_units[index]) != "dimensionless"
                    ):
                        # Update the units if available
                        self.fitted_parameter_units[index] = (
                            f"{name}({self.fitted_parameter_units[index]})"
                        )
                    logger.info(f"Applied transformation: {self.fitted_parameter_names[index]}")
                else:
                    raise ValueError(
                        f"Parameter {param} not found in fitted parameter names for transformation."
                    )

    def _apply_depths(
        self,
        depths: unyt_array,
        photometry_array: np.ndarray,
        N_scatters: int = 5,
        depth_sigma: int = 5,
        return_errors: bool = False,
        min_flux_pc_error: float = 0.0,
    ):
        """Apply depths to the photometry array by scattering the fluxes.

        If depths is 2D, one depth array is randomly selected
        for each photometry row and scatter.

        Parameters:
        -----------
        depths : unyt_array
            Array of depth values. Can be:
            - 1D array of shape (m,) for consistent depths across scatters
            - 2D array of shape (k, m) where k is number of possible depth sets
        photometry_array : np.ndarray
            Input photometry array of shape (m, n)
        N_scatters : int, default=5
            Number of scattered versions to generate for each input row
        depth_sigma : int, default=5
            Divisor to scale the depths
        return_errors : bool, default=False
            If True, return the errors as well
        phot_flux_units : str, default='AB'
            Units of the photometry fluxes. Used to convert depths to the same units.
            Can be "AB", "asinh" or any valid unyt flux density quantity.
        min_flux_pc_error : float, default=0.0
            Minimum percentage error to apply to the fluxes when scattering.

        Returns:
        --------
        np.ndarray or tuple
            Scattered photometry array of shape (m, n_scatters*n)
            If return_errors=True, also returns phot_errors array
        """
        # Get input array dimensions
        m, n = photometry_array.shape

        # Repeat the photometry array for each scatter
        photometry_repeated = np.repeat(photometry_array, N_scatters, axis=1)
        # Check if depths is 2D and handle accordingly
        if depths.ndim == 2:
            k, depth_cols = depths.shape

            # Verify dimensions match
            if depth_cols != m:
                raise ValueError(
                    f"""Mismatch in dimensions: photometry_array has {m}
                    rows but depths has {depth_cols} columns"""
                )

            # For each photometry row and each scatter, randomly select a depth set
            # Shape: (m, N_scatters) - indices of which depth set to use
            depth_indices = np.random.randint(0, k, size=(m, N_scatters))

            # Create depth array for scattering: shape (m, N_scatters)
            selected_depths = depths[depth_indices, np.arange(m)[:, np.newaxis]]

            # Convert to correct units and scale
            depths_std = selected_depths.to(photometry_array.units).value / depth_sigma

            # Expand depths_std to match output dimensions: (m, N_scatters * n)
            depths_std_expanded = np.repeat(depths_std, n, axis=1)

        elif depths.ndim == 1:
            # Original 1D case
            if len(depths) != m:
                raise ValueError(
                    f"""Mismatch in dimensions: photometry_array has {m}
                    rows but depths has {len(depths)} elements"""
                )

            # Convert depths to the correct units
            depths_std = depths.to(photometry_array.units).value / depth_sigma

            # Expand to match output dimensions: (m, N_scatters * n)
            depths_std_expanded = np.repeat(depths_std[:, np.newaxis], N_scatters * n, axis=1)
        elif depths.ndim == 0:
            # Single depth value for all rows
            depth_value = depths.to(photometry_array.units).value / depth_sigma
            depths_std_expanded = np.full((m, N_scatters * n), depth_value)
        else:
            raise ValueError("depths must be 1D or 2D array")

        if min_flux_pc_error > 0.0:
            logger.info(f"Applying minimum percentage error of {min_flux_pc_error}% to depths.")
            # Apply minimum percentage error to depths
            depths_std_expanded = np.maximum(
                depths_std_expanded,
                photometry_repeated.value * min_flux_pc_error / 100.0,
            )

        # Generate all random values at once for better performance
        random_values = (
            np.random.normal(loc=0, scale=depths_std_expanded, size=(m, N_scatters * n))
            * photometry_array.units
        )

        # Add the random values to the repeated photometry
        output_arr = photometry_repeated + random_values

        if return_errors:
            # Return the depth errors used for scattering
            phot_errors = depths_std_expanded * photometry_array.units
            return output_arr, phot_errors

        return output_arr

    def save_state(
        self,
        out_dir,
        name_append="",
        save_method="joblib",
        has_grid=True,
        **extras,
    ):
        r"""Save the state of the SBI fitter to a file.

        Parameters:
            out_dir: Directory to save the model parameters.
            name_append: String to append to the model name in the filename.
            save_method: Method to use for saving. Options are 'torch', 'pickle', '
                'joblib', 'hdf5', 'dill', or 'hickle'. Default is 'joblib'.
            has_grid: Whether the model has a grid. If True, feature and parameter
                arrays will be saved. Default is True.
            \**extras: Additional parameters to save in the parameter dictionary.
        """
        param_dict = {
            "feature_names": self.feature_names,
            "feature_units": self.feature_units,
            "fitted_parameter_units": self.fitted_parameter_units,
            "fitted_parameter_names": self.fitted_parameter_names,
            "timestamp": self._timestamp,
            "prior": self._prior,
            "library_path": self.library_path,
            "name": self.name,
            "has_simulator": self.has_simulator,
        }

        if len(name_append) > 0 and name_append[0] != "_":
            name_append = f"_{name_append}"

        save_path = f"{out_dir}/{self.name}{name_append}_params.pkl"

        param_dict.update(extras)

        if has_grid:
            param_dict["feature_array_flags"] = self.feature_array_flags
            param_dict["feature_array"] = self.feature_array
            param_dict["parameter_array"] = self.fitted_parameter_array

            if (
                "empirical_noise_models" in param_dict["feature_array_flags"]
                and param_dict["feature_array_flags"]["empirical_noise_models"]
            ):
                noise_models = param_dict["feature_array_flags"].pop("empirical_noise_models")
                noise_model_path = save_path.replace(".pkl", "_empirical_noise_models.h5")
                param_dict["empirical_noise_models"] = {}
                for key, model in noise_models.items():
                    save_unc_model_to_hdf5(model, noise_model_path, key, overwrite=True)
                    param_dict["empirical_noise_models"][key] = noise_model_path

        if "stats" in param_dict:
            stats_path = f"{out_dir}/{self.name}{name_append}_summary.json"
            # add some more model info to stats
            # copy any parameters in param_dict which are lists or strings to stats
            param_dict["stats"].append({})
            for key, value in param_dict.items():
                if isinstance(value, (list, str, float, int, bool)) and key not in ["stats"]:
                    param_dict["stats"][-1][key] = value
            try:
                with open(stats_path, "w") as f:
                    json.dump(param_dict["stats"], f, indent=4)
            except Exception as e:
                logger.info(param_dict["stats"])
                logger.error(f"Error saving stats to {stats_path}: {e}")
            param_dict["stats"].pop()

        # convert any torch tensors to numpy arrays on the cpu for compatibility
        param_dict = make_serializable(param_dict, allowed_types=[np.ndarray, UncertaintyModel])

        if save_method == "torch":
            with open(save_path, "wb") as f:
                torch.save(
                    param_dict,
                    save_path,
                )
        elif save_method == "pickle":
            with open(save_path, "wb") as f:
                pickle.dump(
                    param_dict,
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        elif save_method == "joblib":
            from joblib import dump

            dump(param_dict, save_path, compress=3)
        elif save_method == "hdf5":
            import h5py

            save_path = save_path.replace(".pkl", ".h5")

            with h5py.File(save_path, "w") as f:
                for key, value in param_dict.items():
                    if isinstance(value, np.ndarray):
                        f.create_dataset(key, data=value)
                    elif isinstance(value, torch.Tensor):
                        f.create_dataset(key, data=value.cpu().numpy())
                    else:
                        f.attrs[key] = value
        elif save_method == "hickle":
            save_path = save_path.replace(".pkl", ".h5")
            import hickle

            hickle.dump(param_dict, save_path, mode="w", compression="gzip")

        elif save_method == "dill":
            import dill

            with open(save_path, "wb") as f:
                dill.dump(param_dict, f)

        else:
            raise ValueError("Invalid save method. Use 'torch', 'pickle' or 'joblib'.")

        logger.info(f"Saved model parameters to {save_path}.")

    def _apply_empirical_noise_models(
        self,
        photometry_array: np.ndarray,
        phot_names: List[str],
        empirical_noise_models: Dict[str, EmpiricalUncertaintyModel],
        N_scatters: int = 5,
        min_flux_pc_error: float = 0.0,
        flux_units: str = "AB",
        return_errors: bool = False,
        normed_flux_units: str = "AB",
    ):
        """Apply empirical noise models to the photometry array.

        Parameters:
            photometry_array: The photometry array to apply the noise models to.
            empirical_noise_models: A dictionary of empirical noise models to use for
                scattering the fluxes.
                The keys should be the filter names, and
                the values should be the EmpiricalUncertaintyModel objects.
            phot_names: The names of the photometry filters to apply the noise models to.
            N_scatters: The number of times to scatter the fluxes.
            min_flux_pc_error: The minimum percentage error to apply to the fluxes.
                DOESN'T DO ANYTHING HERE YET.
            flux_units: The units of the fluxes in the photometry array.
            return_errors: Whether to return the errors as well.
            flux_units: The units of the fluxes after normalization.

        Returns:
            The photometry array with applied noise models.
        """
        if not isinstance(empirical_noise_models, dict):
            raise ValueError("empirical_noise_models must be a dictionary")

        # check all keys in empirical_noise_models are in phot_names
        for filter in phot_names:
            if filter not in empirical_noise_models:
                raise ValueError(
                    f"""No empirical noise model found for filter {filter}.
                    Please provide a valid model."""
                )

        for filter in empirical_noise_models.keys():
            if filter not in phot_names:
                raise ValueError(
                    f"Filter {filter} in empirical_noise_models is not in phot_names:\
                    {phot_names}. Please provide a valid filter name."
                )

        # Create a copy of the photometry array
        scattered_photometry_s = np.repeat(photometry_array, N_scatters, axis=1)
        errors_s = np.zeros(scattered_photometry_s.shape)

        def apply_noise_model(filter_name):
            if filter_name in empirical_noise_models.keys():
                noise_model = empirical_noise_models[filter_name]
                pos = list(phot_names).index(filter_name)

                # Apply the model to the photometry array
                flux = scattered_photometry_s[pos, :].copy()
                noisy_flux, sampled_sigma = noise_model.apply_noise(
                    flux,
                    true_flux_units=flux_units,
                    out_units=normed_flux_units,
                    # asinh_softening_parameters=asinh_softening_parameters,
                )
                # print(filter_name, noisy_flux.max())

                scattered_photometry_i = noisy_flux
                # Store the errors
                errors_i = sampled_sigma
            else:
                logger.warning(
                    f"No empirical noise model found for filter {filter_name}. Skipping."
                )

            return scattered_photometry_i, errors_i

        results = [apply_noise_model(filter_name) for filter_name in phot_names]
        # Stack the results correctly

        scattered_photometry = np.stack([result[0] for result in results], axis=0)
        errors = np.stack([result[1] for result in results], axis=0)

        assert np.shape(errors) == np.shape(errors_s), (
            f"Shape mismatch: errors {np.shape(errors)} vs errors_s {np.shape(errors_s)}"
        )

        if return_errors:
            return scattered_photometry, errors

        return scattered_photometry

    def detect_misspecification(self, x_obs, X_train=None, retrain=False):
        """Tests misspecification of the model using the MarginalTrainer from sbi.

        This function uses the MarginalTrainer from sbi to train a density
        estimator on the training data, and then calculates the misspecification
        log probability of the test data.

        Args:
            x_obs (np.ndarray): The observed data to test for misspecification.
            X_test (np.ndarray): The test data to check for misspecification.
            X_train (Optional[np.ndarray], optional): The training data to use
                for the misspecification check. If None, it will use the
                training data set in the class. Defaults to None.
            retrain (bool, optional): Whether to retrain the density estimator.
                If False, it will use the previously trained estimator if available.
                Defaults to False.

        Returns:
            np.ndarray:
                An array of misspecification log probabilities for the test data.

        .. note::
           This method is based on the following paper:

           .. code-block:: bibtex

              @inproceedings{schmitt2023detecting,
                  title={Detecting model misspecification in amortized Bayesian
                         inference with neural networks},
                  author={Schmitt, Marvin and Burkner, Paul-Christian and Kothe,
                          Ullrich and Radev, Stefan T},
                  booktitle={DAGM German Conference on Pattern Recognition},
                  pages={541--557},
                  year={2023},
                  organization={Springer}
              }
        """
        from packaging import version

        if version.parse(sbi.__version__) < version.parse("0.25.0"):
            raise ImportError(
                "sbi version >= 0.25.0 is required for misspecification detection. "
                f"Current version: {sbi.__version__}"
            )
        from sbi.diagnostics.misspecification import (
            calc_misspecification_logprob,
        )
        from sbi.inference.trainers.marginal import MarginalTrainer

        if X_train is None:
            if self._X_train is None:
                raise ValueError("No training data found. Please set the training data first.")
            X_train = self._X_train

        if not hasattr(self, "mispecification_trainer") or retrain:
            # Initialize the MarginalTrainer with the desired density estimator
            self.marginal_trainer = MarginalTrainer(density_estimator="NSF")
            self.marginal_trainer.append_samples(X_train)
            self.marginal_trainer_est = self.marginal_trainer.train()

        est = self.marginal_trainer_est

        p_value, reject_H0 = calc_misspecification_logprob(X_train, x_obs, est)

        plt.figure(figsize=(6, 4), dpi=80)
        plt.hist(
            est.log_prob(X_train).detach().numpy(),
            bins=50,
            alpha=0.5,
            label=r"log p($x_{train}$)",
        )
        plt.axvline(
            est.log_prob(x_obs).detach().item(),
            color="red",
            label=r"$\log p(x_{o_{mis}})$)",
        )
        plt.ylabel("Count")
        plt.xlabel(r"$\log p(x)$")
        plt.legend(title="p-value: {:.3f}, Reject H0: {}".format(p_value, reject_H0))
        plt.show()

    def lc2st(
        self,
        x_obs,
        posterior=None,
        X_test=None,
        y_test=None,
    ):
        """Perform the L-C2ST test for model coverage.

        This function uses the L-C2ST classifier to test the model coverage
        by comparing the posterior samples of the observed data with the
        posterior samples of the training data. It generates a plot showing
        the classifier probabilities on the observed data and the null hypothesis.

        Parameters:
            x_obs: The observed data to test.
            posterior: The posterior distribution to use for sampling.
                If None, it will use the posteriors set in the class.
            X_test: The test data to use for the L-C2ST test.
                If None, it will use the training data set in the class.
            y_test: The labels for the test data. If None, it will use
                the training labels set in the class.
        """
        from sbi.analysis.plot import pp_plot_lc2st
        from sbi.diagnostics.lc2st import LC2ST

        if X_test is None:
            if self._X_train is None:
                raise ValueError("No training data found. Please set the training data first.")
            X_test = self._X_train

        if y_test is None:
            if self._y_train is None:
                raise ValueError("No training labels found. Please set the training labels first.")
            y_test = self._y_train

        if posterior is None:
            if self.posteriors is None:
                raise ValueError("No posterior found. Please set the posterior first.")
            posterior = self.posteriors

        # Generate one posterior sample for every prior predictive.
        post_samples_cal = []
        for x in X_test:
            post_samples_cal.append(posterior.sample((1,), x=x)[0])
        post_samples_cal = torch.stack(post_samples_cal)

        # Train the L-C2ST classifier.
        lc2st = LC2ST(
            thetas=y_test,
            xs=X_test,
            posterior_samples=post_samples_cal,
            classifier="mlp",
            num_ensemble=1,
        )
        _ = lc2st.train_under_null_hypothesis()
        _ = lc2st.train_on_observed_data()

        # Note: x_o must have a batch-dimension.
        # I.e. `x_o.shape == (1, observation_shape)`.
        post_samples_star = posterior.sample((10_000,), x=x_obs)
        probs_data, _ = lc2st.get_scores(
            theta_o=post_samples_star,
            x_o=x_obs,
            return_probs=True,
            trained_clfs=lc2st.trained_clfs,
        )
        probs_null, _ = lc2st.get_statistics_under_null_hypothesis(
            theta_o=post_samples_star, x_o=x_obs, return_probs=True
        )

        pp_plot_lc2st(
            probs=[probs_data],
            probs_null=probs_null,
            conf_alpha=0.05,
            labels=["Classifier probabilities \n on observed data"],
            colors=["red"],
        )

    def create_feature_array(
        self,
        flux_units: str = "AB",
        extra_features: list = None,
        **kwargs,
    ):
        """Create a feature array from the raw observation library.

        A simpler wrapper for
        `create_feature_array_from_raw_photometry` with default values.
        This function will create a feature array from the raw observation library
        with no noise, and all photometry in mock catalogue used.
        """
        if self.observation_type == "photometry":
            return self.create_feature_array_from_raw_photometry(
                normed_flux_units=flux_units,
                extra_features=extra_features,
                **kwargs,
            )
        elif self.observation_type == "spectra":
            return self.create_feature_array_from_raw_spectra(
                normed_flux_units=flux_units,
                extra_features=extra_features,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Observation type {self.observation_type} not supported. "
                "Please use 'photometry' or 'spectra'."
            )

    def create_feature_array_from_raw_spectra(
        self,
        extra_features=["redshift"],
        crop_wavelength_range: Union[tuple, list] = None,
        normed_flux_units: str = "AB",
        parameters_to_remove: list = None,
        parameters_to_add: list = None,
        parameter_transformations: Dict[str, Callable] = None,
        resample_wavelengths: unyt_array = None,
        inst_resolution_wavelengths: unyt_array = None,
        inst_resolution_r: unyt_array = None,
        theory_r: float = np.inf,
        min_flux_value: float = -np.inf,
        max_flux_value: float = np.inf,
    ):
        """Create a feature array from the raw spectral library.

        Currently only support basic scaling and cropping

        Parameters
        ----------
            extra_features: list, optional
                Any extra features to be added from the parameter array.
                e.g. redshift
            crop_wavelength_range: tuple or list, optional
                A tuple or list of two values specifying the
                wavelength range to crop the spectra to. Should
                be given in microns (um). This will crop in the observed
                frame if 'redshift' is a feature.
            crop_wavelength_range: unyt_array (in wavelength units) of length 2: (min, max)
                wavelength range to crop the spectra to.
            normed_flux_units: str, optional
                The units of the flux to normalize to. E.g. 'AB', 'nJy', etc.
                If it starts with 'log10 ', it will be treated as a logarithmic
                normalization, e.g. 'log10 nJy'. This is not supported for AB magnitudes.
            parameters_to_remove: list, optional
                Parameters to remove from the parameter array.
            parameters_to_add: list, optional
                Parameters to add to the parameter array.
            parameter_transformations: dict, optional
                A dictionary of parameter transformations to apply to the
                parameters in the parameter array. The keys should be the parameter names,
                and the values should be functions that take a numpy array and return
                a transformed array.
            resample_wavelengths: unyt_array, optional
                If provided, the spectra will be resampled to these wavelengths
                after transforming to the observed frame. Should be in microns (um).
                Required if 'redshift' is a feature.
            inst_resolution_wavelengths: unyt_array, optional
                The wavelength array for the instrument resolution curve. Should be in microns (um).
                Required if 'redshift' is a feature.
            inst_resolution_r: unyt_array, optional
                The resolution (R = lambda / delta_lambda) of the instrument
                as a function of wavelength. Should be the same length as
                instrument_resolution_wave. Required if 'redshift' is a feature.
            theory_r: float, optional
                The resolution of the theoretical spectra. Used for convolution
                when transforming to the observed frame. Default is np.inf (no convolution).
            min_flux_value: float, optional
                Minimum flux value to clip the feature array to in units of normed_flux_units.
            max_flux_value: float, optional
                Maximum flux value to clip the feature array to in units of normed_flux_units.
        """
        if self.observation_type != "spectra":
            raise ValueError("This method is only for spectra models.")

        if self.raw_observation_grid is None:
            raise ValueError("Raw observation library is not set. Please provide a valid library.")

        if extra_features is None:
            extra_features = []

        if parameters_to_remove is None:
            parameters_to_remove = []

        if parameters_to_add is None:
            parameters_to_add = []

        # wavelength range (in um) will be self.raw_observation_names
        from .utils import transform_spectrum

        wavs = np.array(self.raw_observation_names, dtype=float) * um

        has_redshift = "redshift" in self.parameter_names

        grid = self.raw_observation_grid

        if has_redshift:
            logger.info("Redshift detected. Transforming spectra to observed frame.")
            # Assert that all necessary inputs for transformation are provided
            assert resample_wavelengths is not None, (
                "resample_wavelengths must be provided when transforming to observed frame."
            )
            assert inst_resolution_wavelengths is not None, (
                "inst_resolution_wavelengths must be provided for convolution."
            )
            assert inst_resolution_r is not None, (
                "inst_resolution_r must be provided for convolution."
            )

            observed_frame_grid = np.zeros(
                (len(resample_wavelengths), grid.shape[1]), dtype=np.float32
            )

            # Use the robust transformation function inside the loop
            for i in trange(grid.shape[1], desc="Transforming spectra"):
                z = self.parameter_array[i, list(self.fitted_parameter_names).index("redshift")]

                _, transformed_flux = transform_spectrum(
                    theory_wave=wavs.to("um").value,
                    theory_flux=grid[:, i],
                    z=z,
                    observed_wave=resample_wavelengths.to("um").value,
                    resolution_curve_wave=inst_resolution_wavelengths.to("um").value,
                    resolution_curve_r=inst_resolution_r,
                    theory_r=theory_r,
                )
                observed_frame_grid[:, i] = transformed_flux

            wavs = resample_wavelengths
            grid = observed_frame_grid
            mask = np.ones_like(resample_wavelengths, dtype=bool)
        elif crop_wavelength_range is not None:
            mask = (wavs >= crop_wavelength_range[0] * um) & (wavs <= crop_wavelength_range[1] * um)
        else:
            mask = np.ones(len(wavs), dtype=bool)

        if isinstance(mask, unyt_array):
            mask = mask.value
        output_array_size = len(wavs) + len(extra_features) - np.sum(~mask)

        self.feature_array = np.zeros(
            (output_array_size, grid.shape[1]),
        )

        temp_array = unyt_array(copy.deepcopy(grid), units=self.raw_observation_units)
        if normed_flux_units.startswith("log10 "):
            temp_array = np.log10(temp_array.to(normed_flux_units[6:]).value)
        elif normed_flux_units == "AB":
            temp_array = -2.5 * np.log10(temp_array.to(Jy).value) + 8.90
        else:
            temp_array = temp_array.to(normed_flux_units).value

        temp_array = temp_array[mask, :]
        wavs = wavs[mask]

        self.feature_array[: len(wavs), :] = temp_array
        del temp_array

        # Apply min/max flux value cuts
        np.clip(
            self.feature_array,
            a_min=min_flux_value,
            a_max=max_flux_value,
            out=self.feature_array,
        )

        # Add extra features from the parameter array
        feature_units = []
        if extra_features is not None:
            if not isinstance(extra_features, list):
                raise ValueError("extra_features must be a list of feature names.")
            for i, feature in enumerate(extra_features):
                if feature in self.fitted_parameter_names:
                    index = list(self.fitted_parameter_names).index(feature)
                    self.feature_array[len(wavs) + i, :] = self.parameter_array[:, index]
                    feature_units.append(self.fitted_parameter_units[index])
                else:
                    raise ValueError(f"Feature {feature} not found in parameter names.")

        self.feature_array = self.feature_array.T  # Transpose to have features as rows
        self.feature_names = ["spectra"] + extra_features
        self.feature_units = [normed_flux_units]
        self.has_features = True

        self.feature_array_flags = {
            "extra_features": extra_features,
            "crop_wavelength_range": crop_wavelength_range,
            "normed_flux_units": normed_flux_units,
            "parameters_to_remove": parameters_to_remove,
            "parameters_to_add": parameters_to_add,
            "parameter_transformations": parameter_transformations,
        }

        self.update_parameter_array(
            parameters_to_remove=parameters_to_remove,
            parameters_to_add=parameters_to_add,
            parameter_transformations=parameter_transformations,
        )

        logger.info(f"Spectra feature array created with shape {self.feature_array.shape}.")
        logger.info(f"Wavelength Range: {wavs.min().value:.3f} - {wavs.max().value:.3f} um")
        logger.info(f"Min flux: {np.nanmin(self.feature_array):.3f} {normed_flux_units}")
        logger.info(f"Max flux: {np.nanmax(self.feature_array):.3f} {normed_flux_units}")
        # count fraction nan
        nan_fraction = np.sum(~np.isfinite(self.feature_array)) / self.feature_array.size
        logger.info(f"Fraction of NaN/INF values in feature array: {nan_fraction * 100:.3f}%")
        if len(extra_features) > 0:
            logger.info("Extra features added:", extra_features)

    def create_feature_array_from_raw_photometry(
        self,
        normalize_method: Optional[str] = None,
        extra_features: Optional[list] = None,
        normed_flux_units: str = "AB",
        normalization_unit: str = "AB",
        verbose: bool = True,
        scatter_fluxes: Union[int, bool] = False,
        empirical_noise_models: Optional[Dict[str, EmpiricalUncertaintyModel]] = None,
        depths: Optional[unyt_array] = None,
        include_errors_in_feature_array: bool = False,
        min_flux_pc_error: float = 0.0,
        simulate_missing_fluxes: bool = False,
        missing_flux_value: float = 99.0,
        missing_flux_fraction: float = 0.0,
        missing_flux_options: Optional[list] = None,
        include_flags_in_feature_array: bool = False,
        override_phot_grid: Optional[np.ndarray] = None,
        override_phot_grid_units: Optional[str] = None,
        norm_mag_limit: float = 50.0,
        remove_nan_inf: bool = True,
        parameters_to_remove: Optional[list] = None,
        photometry_to_remove: Optional[list] = None,
        parameters_to_add: Optional[list] = None,
        drop_dropouts: bool = False,
        drop_dropout_fraction: float = 1.0,
        asinh_softening_parameters: Union[
            unyt_array,
            List[unyt_array],
            Dict[str, unyt_array],
            str,
            None,
        ] = None,
        max_rows: int = -1,
        parameter_transformations: Optional[Dict[str, Callable]] = None,
    ) -> np.ndarray:
        """Create a feature array from the raw observation library.

        Args:
            normalize_method (str, optional): Method to normalize photometry (e.g.,
                filter names, parameter names). Used to normalize fluxes.
                Defaults to None.
            extra_features (list, optional): Extra features to add. Can be
                functions of filter codes (e.g., ['F090W - F115W']) or
                parameters from the library (e.g., redshift). Defaults to None.
            normed_flux_units (str, optional): Target units for normalized flux
                (e.g., "AB", "asinh", "nJy"). Fluxes will be relative to the
                normalization filter in these units. Defaults to "AB".
            normalization_unit (str, optional): Units for the normalization
                factor (e.g., 'log10 nJy', 'nJy', 'AB'). Can differ from
                `normed_flux_units`. Defaults to "AB".
            verbose (bool, optional): If True, prints progress and summary
                information. Defaults to True.
            scatter_fluxes (Union[int, bool], optional): Whether to scatter
                fluxes with uncertainty. If False, no scatter. If an integer,
                applies that many scatters using noise models or depths.
                Defaults to False.
            empirical_noise_models (Dict[str, EmpiricalUncertaintyModel], optional):
                Dictionary mapping filter names to EmpiricalUncertaintyModel
                objects for flux scattering. Defaults to None.
            depths (unyt_array, optional): An array of depths, one per
                photometric filter. Defaults to None.
            include_errors_in_feature_array (bool, optional): If True,
                includes the RMS uncertainty as a separate feature in the
                input. Defaults to False.
            min_flux_pc_error (float, optional): Minimum percentage error to
                apply to fluxes when scattering. Defaults to 0.0.
            simulate_missing_fluxes (bool, optional): If True, simulates
                missing photometry for a fraction of data, marked with
                `missing_flux_value`. Defaults to False.
            missing_flux_value (float, optional): Value to use for missing
                fluxes. Defaults to 99.0.
            missing_flux_fraction (float, optional): Fraction of missing
                fluxes to simulate. Defaults to 0.0.
            missing_flux_options (list, optional): A list of predefined
                masks for missing fluxes, used if `simulate_missing_fluxes`
                is True. Defaults to None.
            include_flags_in_feature_array (bool, optional): If True, includes
                flags in the feature array. Defaults to False.
            override_phot_grid (np.ndarray, optional): A photometry grid to
                use instead of the raw observation grid. Defaults to None.
            override_phot_grid_units (str, optional): Units for the
                `override_phot_grid`. Defaults to None.
            norm_mag_limit (float, optional): Maximum magnitude limit for
                normalized fluxes. Defaults to 50.0.
            remove_nan_inf (bool, optional): If True, removes rows with
                NaN or Inf values from the feature array. Defaults to True.
            parameters_to_remove (list, optional): List of parameters to
                remove from the parameter array. Defaults to None.
            photometry_to_remove (list, optional): List of filters to remove.
                This is applied early, so other arguments (like `depths`)
                should not include these filters. Defaults to None.
            parameters_to_add (list, optional): List of supplementary
                parameters to add to the feature array. Defaults to None.
            drop_dropouts (bool, optional): If True, drops dropouts from
                the feature array. Defaults to False.
            drop_dropout_fraction (float, optional): The fraction of filters
                a galaxy can drop out in before being dropped. Defaults to 1.0.
            asinh_softening_parameters (Union[unyt_array, List, Dict, str], optional):
                Softening parameter for 'asinh' normalization. Can be a
                single quantity, list, dict, or string (e.g., 'SNR_10').
                Defaults to None.
            max_rows (int, optional): Maximum number of rows to return. If -1,
                all rows are returned. If > 0, rows are randomly sampled.
                Defaults to -1.
            parameter_transformations (Dict[str, Callable], optional):
                Dictionary of functions to apply to parameters (e.g.,
                {'age': np.log10} to infer log_age). Defaults to None.

        Returns:
            tuple:
                - **feature_array** (np.ndarray): The processed feature array.
                - **feature_names** (List[str]): The names of the features
                  (columns) in the feature array.
        """
        if self.observation_type != "photometry":
            raise ValueError("This method is only for photometric models.")

        if extra_features is None:
            extra_features = []

        if parameters_to_remove is None:
            parameters_to_remove = []
        if photometry_to_remove is None:
            photometry_to_remove = []
        if parameters_to_add is None:
            parameters_to_add = []
        if override_phot_grid is not None:
            phot_grid = override_phot_grid
            raw_observation_units = override_phot_grid_units
        else:
            phot_grid = copy.deepcopy(self.raw_observation_grid)
            raw_observation_units = self.raw_observation_units

        raw_observation_names = self.raw_observation_names

        assert isinstance(photometry_to_remove, list), (
            "photometry_to_remove must be a list of filter names to remove."
        )
        if len(photometry_to_remove) > 0:
            # Remove the photometry from the library
            photometry_to_remove = np.array(photometry_to_remove)
            remove_indices = [
                i
                for i, name in enumerate(self.raw_observation_names)
                if name in photometry_to_remove
            ]
            if len(remove_indices) > 0:
                logger.info(
                    f"""Removing {len(remove_indices)} photometry filters: {photometry_to_remove}."""  # noqa: E501
                )
                phot_grid = np.delete(phot_grid, remove_indices, axis=0)
                raw_observation_names = np.delete(self.raw_observation_names, remove_indices)
            else:
                raise ValueError(
                    f"""No matching photometry filters found in the \
                    raw photometry names: {photometry_to_remove}"""
                )

            if len(raw_observation_names) == 0:
                raise ValueError("No photometry filters left after removing the specified ones.")

        if normed_flux_units == "asinh":
            err_is_asinh = isinstance(empirical_noise_models, dict) and all(
                [
                    isinstance(model, AsinhEmpiricalUncertaintyModel)
                    for model in empirical_noise_models.values()
                ]
            )
            assert asinh_softening_parameters is not None or err_is_asinh, (
                "asinh_softening_parameters must be provided for asinh normalization."
            )
            if isinstance(asinh_softening_parameters, (list, np.ndarray)) and not isinstance(
                asinh_softening_parameters, unyt_array
            ):
                assert len(asinh_softening_parameters) == len(raw_observation_names), (
                    "asinh_softening_parameter must be a list of the same length as "
                    "the number of photometry filters."
                )
                asinh_softening_parameter = [
                    asinh_softening_parameters[i].to(Jy).value for i in raw_observation_names
                ] * Jy
            elif isinstance(asinh_softening_parameters, unyt_array):
                asinh_softening_parameter = asinh_softening_parameters.to(Jy)
            elif isinstance(asinh_softening_parameters, str):
                assert asinh_softening_parameters.startswith("SNR_"), (
                    "If a string, asinh_softening_parameters must start with 'SNR_'."
                )
                assert (scatter_fluxes) and (
                    depths is not None or empirical_noise_models is not None
                ), """If setting asinh_softening_parameters from noise models, \
                    depths or empirical_noise_models must be provided."""

                val = float(asinh_softening_parameters.split("_")[-1])
                asinh_softening_parameter = [val for name in raw_observation_names]
        else:
            asinh_softening_parameter = None

        phot = unyt_array(phot_grid, units=raw_observation_units)
        converted = False

        if scatter_fluxes:
            assert depths is not None or empirical_noise_models is not None, (
                "If scattering fluxes, depths or empirical noise models must be provided."
            )
            assert isinstance(phot, unyt_array)

            if depths is not None:
                logger.info(
                    f"Using depth-based noise models with {scatter_fluxes} scatters per row."
                )

                if isinstance(depths, dict):
                    # Map depths to the raw_observation_names
                    depths = unyt_array(
                        [depths[name] for name in raw_observation_names],
                        units=depths[list(depths.keys())[0]].units,
                    )
                assert isinstance(depths, unyt_array)
                self.phot_depths = depths
                self.min_flux_pc_error = min_flux_pc_error
                # Need to get units right here!
                phot, phot_errors = self._apply_depths(
                    depths,
                    phot,
                    scatter_fluxes,
                    return_errors=True,
                    min_flux_pc_error=min_flux_pc_error,
                )

                # Calculate correct asinh parameters if needed.
                if (
                    normed_flux_units == "asinh"
                    and isinstance(asinh_softening_parameters, str)
                    and asinh_softening_parameters.startswith("SNR_")
                ):
                    # Set the asinh softening parameter from the noise model
                    # if based on SNR. E.g. given depths unit, asinh
                    # softening parameter here would be e.g. 5 sigma depths
                    # if asinh_softening_parameters = "SNR_5".
                    asinh_softening_parameter = [
                        asinh_softening_parameter[i] * self.phot_depths[i].to("Jy").value / 5.0
                        for i in range(len(raw_observation_names))
                    ]
                    asinh_softening_parameter = unyt_array(
                        asinh_softening_parameter,
                        units=Jy,
                    )

                converted = False
            elif empirical_noise_models is not None:
                logger.info(f"Using empirical noise models with {scatter_fluxes} scatters per row.")
                self.empirical_noise_models = empirical_noise_models

                phot, phot_errors = self._apply_empirical_noise_models(
                    phot,
                    raw_observation_names,
                    empirical_noise_models,
                    N_scatters=scatter_fluxes,
                    min_flux_pc_error=min_flux_pc_error,
                    return_errors=True,
                    flux_units=phot.units,
                    normed_flux_units=normed_flux_units,
                )
                converted = True

        else:
            phot_errors = None

        if normed_flux_units == "AB":
            if phot_errors is not None and not converted:
                phot_errors = (
                    2.5 * phot_errors.to("uJy").value / (np.log(10) * phot.to("uJy").value)
                )

            if not converted:
                phot_mag = -2.5 * np.log10(phot.to("uJy").value) + 23.9
                mask = phot.to("uJy").value < 0
            else:
                phot_mag = phot
                mask = np.isnan(phot_mag) | np.isinf(phot_mag)
                logger.info(
                    f"number of NaN, Inf values in phot_mag: {np.sum(np.isnan(phot_mag))}, {np.sum(np.isinf(phot_mag))}"  # noqa: E501
                )
            # Any negative fluxes, just set to the limit
            phot_mag[mask] = norm_mag_limit
            phot = phot_mag
            norm_func = np.subtract

        elif normed_flux_units == "asinh":
            if not converted:
                if phot_errors is not None:
                    phot_errors = f_jy_err_to_asinh(
                        phot.to("Jy"),
                        phot_errors.to("Jy"),
                        f_b=asinh_softening_parameter,
                    )

                phot = f_jy_to_asinh(
                    phot.to("Jy"),
                    f_b=asinh_softening_parameter,
                )

            norm_func = np.subtract

        else:
            if not converted:
                norm_func = np.divide
                try:
                    phot.to(normed_flux_units)
                    convertible = True
                except Exception:
                    convertible = False

                if convertible:
                    phot = phot.to(normed_flux_units).value
                    if phot_errors is not None:
                        phot_errors = phot_errors.to(normed_flux_units).value
                else:
                    try:
                        scaling, unit = normed_flux_units.split(" ")
                    except ValueError:
                        raise ValueError(
                            "Don't understand normed_flux_units."
                            "If string, should be e.g. 'log10 nJy',Otherwise pass a Unit directly."
                        )
                    scales = {"log10": np.log10, "log": np.log, "": lambda x: x, "sqrt": np.sqrt}
                    if scaling not in scales:
                        raise ValueError(
                            f'Scaling "{scaling}" not recognized. Use "log10", "log","", or "sqrt".'
                        )

                    phot = phot.to(unit).value
                    if phot_errors is not None:
                        phot_errors = phot_errors.to(unit).value
                    if scaling in scales:
                        if phot_errors is not None:
                            # Need error propagation for log scaling
                            if scaling == "log10":
                                phot_errors = phot_errors / (phot * np.log(10))
                                norm_func = np.subtract
                            elif scaling == "log":
                                phot_errors = phot_errors / (phot)
                                norm_func = np.subtract
                            elif scaling == "sqrt":
                                phot_errors = phot_errors / (2 * np.sqrt(phot))
                        phot = scales[scaling](phot)
                    else:
                        raise ValueError(
                            f"Scaling {scaling} not recognized. Use 'log10', 'log', '', or 'sqrt'."
                        )

        delete_rows = []
        # Normalize the photometry grid
        if normalize_method is not None:
            if normalize_method in raw_observation_names:
                norm_index = list(raw_observation_names).index(normalize_method)

                normalization_factor = phot[norm_index, :]
                norm_factor_original = phot_grid[norm_index, :]

                # Create a copy of the raw photometry names for consistent reference
                raw_observation_names = np.array(raw_observation_names)
                raw_observation_names = np.delete(raw_observation_names, norm_index)
                phot = np.delete(phot, norm_index, axis=0)
                if scatter_fluxes:
                    phot_errors = np.delete(phot_errors, norm_index, axis=0)

                # Create normalized photometry while preserving the original shape
                normed_photometry = norm_func(phot, normalization_factor)

                # Convert the normalization factor to the desired unit
                if normalization_unit.startswith("log10"):
                    log = True
                    normalization_unit_cleaned = normalization_unit.split(" ")[1]
                else:
                    log = False
                    normalization_unit_cleaned = normalization_unit

                if normalization_unit_cleaned == "AB":
                    normalization_factor_converted = (
                        -2.5
                        * np.log10(
                            unyt_array(
                                norm_factor_original,
                                units=raw_observation_units,
                            )
                            .to("uJy")
                            .value
                        )
                        + 23.9
                    )
                else:
                    normalization_factor_converted = (
                        unyt_array(norm_factor_original, units=raw_observation_units)
                        .to(normalization_unit_cleaned)
                        .value
                    )

                if log:
                    normalization_factor_converted = np.log10(normalization_factor_converted)
                    normalization_factor_converted[normalization_factor_converted == -np.inf] = 0.0
                    normalization_factor_converted[normalization_factor_converted == np.inf] = 0.0

            elif normalize_method in self.supplementary_parameter_names:
                norm_index = list(self.supplementary_parameter_names).index(normalize_method)
                norm_unit = self.supplementary_parameter_units[norm_index]
                normalization_factor = self.supplementary_parameters[norm_index, :]
                # count where normalzation is 0

                assert normalization_factor.shape[0] == phot_grid.shape[1], (
                    "Normalization factor should have the same shape as the photometry grid."
                )
                assert norm_unit == raw_observation_units, (
                    "Normalization factor should have the same units as the photometry grid."
                )

                if normed_flux_units == "AB":
                    normalization_factor_use = (
                        -2.5
                        * np.log10(
                            unyt_array(normalization_factor, units=norm_unit).to("uJy").value
                        )
                        + 23.9
                    )
                elif normed_flux_units == "asinh":
                    # This may not work - so many edges cases.
                    normalization_factor_use = f_jy_to_asinh(
                        unyt_array(normalization_factor, units=norm_unit),
                        f_b=asinh_softening_parameter[norm_index],
                    )

                else:
                    normalization_factor_use = normalization_factor

                if scatter_fluxes:
                    normalization_factor_use = np.repeat(
                        normalization_factor_use, scatter_fluxes, axis=0
                    )
                    normalization_factor = np.repeat(normalization_factor, scatter_fluxes, axis=0)

                normed_photometry = norm_func(phot, normalization_factor_use)

                # Convert the normalization factor to the desired unit
                if normalization_unit.startswith("log10"):
                    log = True
                    normalization_unit_cleaned = normalization_unit.split(" ")[1]
                else:
                    log = False
                    normalization_unit_cleaned = normalization_unit

                if normalization_unit_cleaned == "AB":
                    normalization_factor_converted = (
                        -2.5
                        * np.log10(
                            unyt_array(normalization_factor, units=norm_unit).to("uJy").value
                        )
                        + 23.9
                    )
                else:
                    normalization_factor_converted = (
                        unyt_array(normalization_factor, units=norm_unit)
                        .to(normalization_unit_cleaned)
                        .value
                    )

                raw_observation_names = np.array(raw_observation_names)

            else:
                raise NotImplementedError(
                    """Normalization method not implemented.
                    Please use a filter name for normalization."""
                )
        else:
            normed_photometry = phot
            normalization_factor_converted = np.ones(normed_photometry.shape[1])
            normalization_factor = normalization_factor_converted
            raw_observation_names = np.array(raw_observation_names)

            # Convert the photometry to the desired units

        if phot_errors is not None:
            # Will have errors on the end of the photometry array. Add names (unc_*)
            # and units (uncertainty) to the end of the feature names and units.
            error_names = [f"unc_{name}" for name in raw_observation_names]
            error_units = [normed_flux_units] * len(error_names)
        else:
            error_names = []
            error_units = []

        if np.sum(normalization_factor == 0.0) > 0:
            # delete these indexes from the photometry
            logger.info(
                "Warning: Normalization factor is 0.0 for "
                f"{np.sum(normalization_factor == 0.0)} rows. These will be deleted."
            )
            delete_rows.extend(np.where(normalization_factor == 0.0)[0].tolist())

        if normed_flux_units == "AB":
            # very small fluxes can blow up here and go to infinity.
            # Set a maximum difference of some large amount - say 50
            # If not normalized, set minmum flux to 50 AB.
            # If normalized to some AB, set minimum normalized flux to 50.
            normed_photometry[normed_photometry > norm_mag_limit] = norm_mag_limit

        norm = 1 if normalize_method is not None else 0

        length = normed_photometry.shape[1]

        size = (
            len(raw_observation_names)
            + len(extra_features)
            + norm
            + include_errors_in_feature_array * len(error_names)
            + include_flags_in_feature_array * len(raw_observation_names)
        )
        # Create the feature array
        # Photometry + extra features + normalization factor
        feature_array = np.zeros((size, length))

        assert np.shape(feature_array[: len(raw_observation_names), :]) == np.shape(
            normed_photometry
        ), f"""Shape mismatch: {np.shape(feature_array[: len(raw_observation_names), :])}
            != {np.shape(normed_photometry)}"""
        # Fill the feature array with the normalized photometry
        feature_array[: len(raw_observation_names), :] = normed_photometry

        if phot_errors is not None and include_errors_in_feature_array:
            # Add the errors to the feature array
            # Work out the starting index given total length of feature array
            start_index = len(raw_observation_names)
            feature_array[start_index : start_index + len(error_names), :] = phot_errors

        flag_units = []
        flag_names = []

        if simulate_missing_fluxes:
            start_index = len(raw_observation_names) + len(error_names)
            if missing_flux_options is not None:
                # For each row, pick a mask randomly from the missing_flux_options
                mask = np.zeros((len(raw_observation_names), feature_array.shape[1]))
                for row in range(feature_array.shape[1]):
                    # Choose a random mask from the missing_flux_options
                    chosen_index = np.random.choice(
                        range(len(missing_flux_options)),
                        p=[1 / len(missing_flux_options)] * len(missing_flux_options),
                    )

                    mask[:, row] = missing_flux_options[chosen_index]

                mask = mask.astype(np.float32)

            else:
                # Flags will be after errors (if any)
                # generate a mask with missing_flux_fraction missing points
                # 1.0 is missing
                mask = np.random.choice(
                    [0.0, 1.0],
                    size=(len(raw_observation_names), feature_array.shape[1]),
                    p=[1 - missing_flux_fraction, missing_flux_fraction],
                )
            if include_flags_in_feature_array:
                flag_units = [None] * len(raw_observation_names)
                flag_names = [f"flag_{name}" for name in raw_observation_names]
                feature_array[start_index : start_index + len(raw_observation_names), :] = mask
            # Set the missing fluxes to the missing_flux_value
            feature_array[: len(raw_observation_names), :][mask == 1.0] = missing_flux_value

            if len(error_names) > 0:
                feature_array[len(raw_observation_names) : start_index, :][mask == 1.0] = (
                    missing_flux_value
                )

        if normalize_method is not None:
            # Add the normalization factor as the last column
            feature_array[-1, :] = normalization_factor_converted

        # Create the feature names
        nfeatures = feature_array.shape[0]
        feature_names = [""] * nfeatures

        # Add filter names
        for i in range(len(raw_observation_names)):
            feature_names[i] = raw_observation_names[i]

        if phot_errors is not None and include_errors_in_feature_array:
            # Add the error names
            for i in range(len(error_names)):
                feature_names[len(raw_observation_names) + i] = error_names[i]

        if include_flags_in_feature_array:
            # Add the flag names
            for i in range(len(flag_names)):
                feature_names[len(raw_observation_names) + len(error_names) + i] = flag_names[i]

        if normalize_method is not None:
            # Add the normalization factor name
            norm_name = f"norm_{normalize_method}_{normalization_unit}"
            feature_names[-1] = norm_name
        else:
            norm_name = None

        self.provided_feature_parameters = []
        # Process extra features if any.
        # Currently extra features don't have uncertainties.
        if len(extra_features) > 0:
            parser = FilterArithmeticParser()
            for i, feature in enumerate(extra_features):
                # pos is first '' in feature_names
                pos = np.where(np.array(feature_names) == "")[0][0]
                if feature in self.parameter_names:
                    if feature not in self.provided_feature_parameters:
                        self.provided_feature_parameters.append(feature)
                    index = list(self.parameter_names).index(feature)
                    arr = self.parameter_array[:, index]
                    if scatter_fluxes:
                        arr = np.repeat(arr, scatter_fluxes, axis=0)

                    feature_array[pos, :] = arr
                    feature_names[pos] = feature

                elif feature in self.supplementary_parameter_names:
                    if feature not in self.provided_feature_parameters:
                        self.provided_feature_parameters.append(feature)
                    index = list(self.supplementary_parameter_names).index(feature)
                    arr = self.supplementary_parameters[index, :]
                    if scatter_fluxes:
                        arr = np.repeat(arr, scatter_fluxes, axis=0)

                    feature_array[pos, :] = arr
                    feature_names[pos] = feature

                elif feature in self.simple_fitted_parameter_names:
                    if feature not in self.provided_feature_parameters:
                        self.provided_feature_parameters.append(feature)
                    index = list(self.simple_fitted_parameter_names).index(feature)
                    arr = self.simple_fitted_parameters[index, :]
                    if scatter_fluxes:
                        arr = np.repeat(arr, scatter_fluxes, axis=0)

                    feature_array[pos, :] = arr
                    feature_names[pos] = feature

                else:
                    # Parse the feature expression
                    tokens = parser.tokenize(feature)
                    names = [i.split(".")[-1] for i in raw_observation_names]
                    logger.info(f"Tokenizing feature: {feature}")
                    value = parser.evaluate(tokens, dict(zip(names, normed_photometry)))
                    feature_array[pos, :] = value

                    # Add the feature name
                    feature_names[pos] = feature

        if remove_nan_inf:
            # any row with a nan or inf in the feature array will be deleted
            mask = ~np.isfinite(feature_array)
            row_mask = np.sum(mask, axis=0) > 0
            delete_rows.extend(np.where(row_mask)[0].tolist())
            num = np.sum(row_mask)
            if num > 0:
                logger.warning(
                    f"""Warning: Deleting {num} rows with NaN or Inf
                    values in the feature array."""
                )

        if drop_dropouts:
            # Find all galaxies where more than a drop_fraction of
            # bands are at norm_mag_limit
            dropout_mask = (
                np.sum(
                    np.abs(feature_array[: len(raw_observation_names), :]) >= norm_mag_limit,
                    axis=0,
                )
                >= len(raw_observation_names) * drop_dropout_fraction
            )
            dropout_rows = np.where(dropout_mask)[0]
            if len(dropout_rows) > 0:
                logger.warning(
                    f"""Warning: Dropping {len(dropout_rows)} dropouts where more than
                    {drop_dropout_fraction * 100}% of bands are at the norm_mag_limit."""
                )
                delete_rows.extend(dropout_rows.tolist())

        if max_rows > 0 and (feature_array.shape[1] - len(delete_rows)) > max_rows:
            # select rows to delete to reduce the feature array to max_rows
            if verbose:
                logger.info(
                    f"""Reducing feature array from {feature_array.shape[1]} to \
                    {max_rows} rows."""
                )

            # Randomly sample max_rows rows from the feature array
            options = np.setdiff1d(
                np.arange(feature_array.shape[1]),
                delete_rows,
            )

            random_indices = np.random.choice(
                options,
                size=max_rows,
                replace=False,
            )

            # Add the random indices to the delete rows
            delete_rows.extend([i for i in options if i not in random_indices])

        # Remove any rows in delete_rows
        if len(delete_rows) > 0:
            feature_array = np.delete(feature_array, delete_rows, axis=1)

        # check if all rows got deleted
        if feature_array.shape[1] == 0:
            raise ValueError(
                "All rows in the feature array were deleted. Please check the input parameters."
            )

        assert "" not in feature_names, (
            "Feature names should not be empty. Please check the extra features."
        )

        self.feature_array = feature_array.astype(np.float32).T
        self.feature_names = feature_names
        self.feature_units = (
            [normed_flux_units] * len(raw_observation_names)
            + error_units
            + flag_units
            + [None] * len(extra_features)
            + [normalization_unit]
        )
        self.has_features = True

        if verbose:
            logger.info("---------------------------------------------")
            logger.info(
                f"""Features: {self.feature_array.shape[1]} features over \
{self.feature_array.shape[0]} samples"""
            )
            logger.info("---------------------------------------------")
            logger.info("Feature: Min - Max")
            logger.info("---------------------------------------------")

            for pos, feature_name in enumerate(feature_names):
                logger.info(
                    f"""{feature_name}: {np.min(feature_array[pos]):.6f} - \
{np.max(feature_array[pos]):.3f} {self.feature_units[pos]}"""
                )
            logger.info("---------------------------------------------")

        # Save all method inputs on self

        self.feature_array_flags = {
            "normalize_method": normalize_method,
            "extra_features": extra_features,
            "normed_flux_units": normed_flux_units,
            "normalization_unit": normalization_unit,
            "scatter_fluxes": scatter_fluxes,
            "empirical_noise_models": empirical_noise_models,
            "depths": depths,
            "include_errors_in_feature_array": include_errors_in_feature_array,
            "min_flux_pc_error": min_flux_pc_error,
            "simulate_missing_fluxes": simulate_missing_fluxes,
            "missing_flux_value": missing_flux_value,
            "missing_flux_fraction": missing_flux_fraction,
            "missing_flux_options": missing_flux_options,
            "include_flags_in_feature_array": include_flags_in_feature_array,
            "override_phot_grid": override_phot_grid,
            "override_phot_grid_units": override_phot_grid_units,
            "norm_mag_limit": norm_mag_limit,
            "remove_nan_inf": remove_nan_inf,
            "parameters_to_remove": parameters_to_remove,
            "photometry_to_remove": photometry_to_remove,
            "parameters_to_add": parameters_to_add,
            "drop_dropouts": drop_dropouts,
            "drop_dropout_fraction": drop_dropout_fraction,
            "raw_observation_names": raw_observation_names,
            "error_names": error_names,
            "flag_names": flag_names,
            "norm_name": norm_name,
            "asinh_softening_parameters": asinh_softening_parameters,
        }

        self.update_parameter_array(
            delete_rows=delete_rows,
            n_scatters=scatter_fluxes,
            parameters_to_remove=parameters_to_remove,
            parameters_to_add=parameters_to_add,
            parameter_transformations=parameter_transformations,
        )

        return feature_array, feature_names

    def bin_noisy_testing_data(
        self,
        X_test: np.ndarray = None,
        y_test: np.ndarray = None,
        snr_bins=[1, 5, 8, 10, np.inf],
        snr_feature_names: list = None,
        return_indices: bool = False,
    ):
        """Bin the testing data based on SNR.

        This method calculates the SNR for each feature in the testing data
        and bins the data based on the provided SNR bins. It returns the binned
        testing data and the corresponding labels.

        Parameters
        ----------
        X_test : np.ndarray, optional
            The testing data features. If None, uses the stored _X_test.
        y_test : np.ndarray, optional
            The testing data labels. If None, uses the stored _y_test.
        snr_bins : list, optional
            The SNR bins to use for binning the data. Default is [5, 8, 10].
        snr_feature_names : list, optional
            The feature names corresponding to the SNR features. If None,
            it will use the all the photomety feature names from the feature array.
        return_indices : bool, optional
            If True, return the indices of the binned data instead of the data itself.
        """
        if X_test is None:
            X_test = self._X_test
        if y_test is None:
            y_test = self._y_test

        if not self.has_features:
            raise RuntimeError(
                "The feature creation pipeline has not been initialized. Please run `create_feature_array_from_raw_photometry` first."  # noqa: E501
            )

        if (
            not self.feature_array_flags["scatter_fluxes"]
            or not self.feature_array_flags["include_errors_in_feature_array"]
        ):
            return X_test, y_test

        phot_units = self.feature_array_flags["normed_flux_units"]
        if self.feature_array_flags.get("empirical_noise_models", None) is not None:
            nm = self.feature_array_flags["empirical_noise_models"]
            nm = all([isinstance(nm, AsinhEmpiricalUncertaintyModel) for nm in nm.values()])
        else:
            nm = False

        snrs = []
        for feature_name in snr_feature_names or self.feature_names:
            if feature_name.startswith("unc_"):
                break
            elif nm:
                feature_index = self.feature_names.index(feature_name)
                err_index = self.feature_names.index(f"unc_{feature_name}")
                # Need to know SNR from asinh magnitudes.
                f_b = self.feature_array_flags["empirical_noise_models"][feature_name].b.to("Jy")
                snr = asinh_to_snr(
                    X_test[:, feature_index],
                    X_test[:, err_index],
                    f_b=f_b,
                )

            elif phot_units == "AB":
                feature_name = "unc_" + feature_name
                error_index = self.feature_names.index(feature_name)
                snr = 2.5 / (X_test[:, error_index] * np.log(10))
            elif isinstance(phot_units, unyt_quantity):
                error_index = self.feature_names.index(f"unc_{feature_name}")
                phot_index = self.feature_names.index(feature_name)
                snr = X_test[:, phot_index] / X_test[:, error_index]
            else:
                raise ValueError(
                    f"Unsupported photometric units: {phot_units}. "
                    "Only 'AB' or unyt_quantity are supported."
                )
            snrs.append(snr)

        # Calculate mean SNR acorss all features for each galaxy
        snrs = np.array(snrs)
        mean_snr = np.mean(snrs, axis=0)

        if return_indices:
            indices = []
            for i in range(len(snr_bins) - 1):
                lower_bound = snr_bins[i]
                upper_bound = snr_bins[i + 1]
                mask = (mean_snr >= lower_bound) & (mean_snr < upper_bound)
                indices.append(np.where(mask)[0])
            return indices
        # Bin the data based on the SNR bins
        binned_data = []
        binned_labels = []
        for i in range(len(snr_bins) - 1):
            lower_bound = snr_bins[i]
            upper_bound = snr_bins[i + 1]
            mask = (mean_snr >= lower_bound) & (mean_snr < upper_bound)
            binned_data.append(X_test[mask])
            binned_labels.append(y_test[mask])

        # Convert to numpy arrays
        binned_data = [np.array(data) for data in binned_data]
        binned_labels = [np.array(labels) for labels in binned_labels]

        return binned_data, binned_labels

    def plot_parameter_deviations(
        self,
        parameters="all",
        X_test: np.ndarray = None,
        y_test: np.ndarray = None,
        posteriors: np.ndarray = None,
        snr_bins=None,
        snr_feature_names: list = None,
        contours: bool = True,
        error_bars: bool = False,
    ):
        """Plot the deviation of parameters from the true values.

        E.g. Mrecoverd - Mtrue vs Mtrue. One row in grid per snr_bin if not None.
        """
        if X_test is None:
            X_test = self._X_test
        if y_test is None:
            y_test = self._y_test
        if not self.has_features:
            raise RuntimeError(
                "The feature creation pipeline has not been initialized. Please run `create_feature_array_from_raw_photometry` first."  # noqa: E501
            )

        if posteriors is None:
            if self.posteriors is None:
                raise ValueError(
                    "No posteriors provided and no posteriors stored in the SBI instance."
                )
            posteriors = self.posteriors

        if parameters == "all":
            parameters = self.fitted_parameter_names

        if snr_bins is not None:
            # Bin the testing data based on SNR
            binned_data, binned_labels = self.bin_noisy_testing_data(
                X_test=X_test,
                y_test=y_test,
                snr_bins=snr_bins,
                snr_feature_names=snr_feature_names,
            )
        else:
            binned_data = [X_test]
            binned_labels = [y_test]

        fig, axes = plt.subplots(
            nrows=len(binned_data),
            ncols=len(parameters),
            figsize=(3 * len(parameters), 3 * len(binned_data)),
            dpi=300,
        )

        for i, (data, labels) in enumerate(zip(binned_data, binned_labels)):
            y_test_samples = self.sample_posterior(
                posteriors=posteriors,
                X_test=data,
            )

            if len(self.fitted_parameter_names) == 1:
                labels = np.expand_dims(labels, axis=-1)

            # get quantiles of the posterior samples
            quantiles = np.quantile(y_test_samples, [0.16, 0.5, 0.84], axis=1)

            if len(self.fitted_parameter_names) == 1:
                labels = np.expand_dims(labels, axis=-1)

            for j, param in enumerate(parameters):
                if len(binned_data) > 1 and len(parameters) > 1:
                    ax = axes[i, j]
                elif len(binned_data) == 1 and len(parameters) > 1:
                    ax = axes[j]
                elif len(binned_data) > 1 and len(parameters) == 1:
                    ax = axes[i]
                else:
                    ax = axes
                # Get the true values for the parameter
                index = list(self.fitted_parameter_names).index(param)
                true_values = labels[:, index]
                # Get the recovered values from the posterior samples
                recovered_values = quantiles[1, :, index]  # median of the posterior samples
                # Calculate the deviation
                deviation = recovered_values - true_values

                mask = recovered_values == 0
                true_values = true_values[~mask]
                deviation = deviation[~mask]

                # Plot the deviation vs true values
                ax.scatter(true_values, deviation, alpha=0.5, s=1)
                ax.axhline(0, color="red", linestyle="--")
                if self.fitted_parameter_units is not None:
                    param_unit = f" ({self.fitted_parameter_units[index]})"
                else:
                    param_unit = ""
                ax.set_xlabel(f"True {param}{param_unit}", fontsize=10)
                ax.set_ylabel(rf"$\Delta$ {param}{param_unit}", fontsize=10)
                ax.set_title(
                    f"SNR bin: {snr_bins[i]} - {snr_bins[i + 1]}" if snr_bins else "All SNRs"
                )
                if contours:
                    import seaborn as sns

                    sns.kdeplot(
                        x=true_values,
                        y=deviation,
                        ax=ax,
                        fill=True,
                        levels=[0.68, 0.95, 0.997],
                        thresh=0,
                        alpha=0.5,
                    )
                if error_bars:
                    yerr = [
                        quantiles[1, :, index][~mask] - quantiles[0, :, index][~mask],
                        quantiles[2, :, index][~mask] - quantiles[1, :, index][~mask],
                    ]
                    ax.errorbar(
                        true_values,
                        deviation,
                        yerr=yerr,
                        fmt="o",
                        alpha=0.5,
                        capsize=0,
                        elinewidth=1,
                    )
                stdev = np.std(deviation)
                mean = np.mean(deviation)
                median = np.median(deviation)
                ax.text(
                    0.05,
                    0.95,
                    f"Mean: {mean:.2f}\nMedian: {median:.2f}\nStd: {stdev:.2f}",
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    horizontalalignment="left",
                )

        plt.tight_layout()
        return fig

    def create_features_from_observations(
        self,
        observations: Union[Table, pd.DataFrame],
        columns_to_feature_names: dict = None,
        flux_units: Union[str, unyt_quantity, None] = None,
        missing_data_flag: Any = -99,
        override_transformations: dict = {},
        ignore_missing=False,
    ):
        """Create a feature array from observational data.

        .. note::
           Transformations applied to an existing feature array are
           saved in `self.feature_array_flags`, but can be overridden
           with `override_transformations`. Available transformations are
           arguments to `create_feature_array_from_raw_photometry`,
           although some (like scattering fluxes or error modelling)
           are not applicable here and will have no effect.

        Args:
            observations (Union[np.ndarray, Table, pd.DataFrame]): The
                observational data to create the feature array from.
            flux_units (Union[str, unyt_quantity, None], optional):
                Required if the observations do not match 'normed_flux_units'
                in `self.feature_array_flags`. If None, units are taken
                from column metadata or assumed to be the same as the
                training data. Defaults to None.
            columns_to_feature_names (Dict[str, str], optional): A dictionary
                mapping column names in `observations` to feature names.
                If None, a direct mapping to
                `self.feature_array_flags['raw_observation_names']`
                is assumed. Defaults to None.
            missing_data_flag (Any, optional): Value in columns
                delineating missing data. Depending on setup, this may be
                flagged to the model or cause the entry to be ignored.
                Defaults to None.
            override_transformations (Dict[str, Any], optional): A dictionary
                of transformations to override defaults in
                `self.feature_array_flags` (e.g., normalization method).
                Defaults to None.
            ignore_missing (bool, optional): If True, will ignore missing
                data in the observations, keeping missing values unchanged.
                Primarily useful for SBI++ techniques. Defaults to False.

        Returns:
            np.ndarray:
                The processed feature array derived from the observations.
        """
        if len(self.feature_array_flags) == 0:
            raise ValueError("No feature array flags found. Please create the feature array first.")

        # if not getattr(self, "has_features", False):
        #    raise RuntimeError(
        #    )
        #        "The feature creation pipeline has not been initialized. Please run `create_feature_array_from_raw_photometry` first."  # noqa: E501

        feature_array_flags = self.feature_array_flags

        feature_array_flags.update(override_transformations)

        # Check if the observations are a Table or DataFrame
        if isinstance(observations, Table):
            observations = observations.to_pandas()

        elif not isinstance(observations, pd.DataFrame):
            raise TypeError("Observations must be a pandas DataFrame or an astropy Table.")

        # Check

        if columns_to_feature_names is None:
            # Assume a direct mapping between the
            # columns in the observations and feature names
            columns_to_feature_names = {col: col for col in observations.columns}

        feature_names_to_columns = {v: k for k, v in columns_to_feature_names.items()}

        # feature_names_to_columns should be e.g. {'HST.ACS_WFC.F606W': 'F606W'}

        if not self.feature_array_flags["simulate_missing_fluxes"]:
            # Should have a column for every photometry filter in the training data

            # Check all names are keys in columns_to_feature_names
            for name in feature_array_flags["raw_observation_names"]:
                if name not in feature_names_to_columns:
                    raise ValueError(
                        f"""Column '{name}' not found in observations. Please provide a mapping for all photometry filters."""  # noqa: E501
                    )

            if feature_array_flags["include_errors_in_feature_array"]:
                # Check all errors are keys in columns_to_feature_names
                for name in feature_array_flags["error_names"]:
                    if name not in feature_names_to_columns:
                        raise ValueError(
                            f"""Column '{name}' not found in observations. Please provide a mapping for all errors."""  # noqa: E501
                        )

        if feature_array_flags["include_flags_in_feature_array"]:
            # Check all flags are keys in columns_to_feature_names
            for name in feature_array_flags["flag_names"]:
                if name not in feature_names_to_columns:
                    if name in observations.columns:
                        # If the column is in the observations, but not in the mapping,
                        # we can assume it is a flag.
                        columns_to_feature_names[name] = name
                    '''raise ValueError(
                        f"""Column '{name}' not found in observations.
                        Please provide a mapping for all flags."""
                    )'''

        if (
            feature_array_flags["norm_name"] is not None
            and feature_array_flags["norm_name"] not in feature_names_to_columns
        ):
            raise ValueError(
                f"""Column '{feature_array_flags["norm_name"]}' not found in
                observations. Please provide a mapping for the normalization factor."""
            )

        for name in feature_array_flags["extra_features"]:
            if name not in feature_names_to_columns:
                raise ValueError(
                    f"""Column '{name}' not found in observations.
                    Please provide a mapping for all extra features."""
                )

        photometry_columns = [
            feature_names_to_columns[name] for name in feature_array_flags["raw_observation_names"]
        ]

        # Check if the flux units match the training data
        training_flux_units = feature_array_flags["normed_flux_units"]

        # Check for the exception
        # print(feature_array_flags["empirical_noise_models"])
        if (
            "empirical_noise_models" in feature_array_flags
            and feature_array_flags["empirical_noise_models"] is not None
        ):
            if all(
                isinstance(model, AsinhEmpiricalUncertaintyModel)
                for model in feature_array_flags["empirical_noise_models"]
            ):
                training_flux_units = "asinh"
        else:
            # if str, should match flux units.
            if isinstance(training_flux_units, str):
                assert flux_units == training_flux_units, f"""Flux units '{flux_units}' do not match
                    training data units '{training_flux_units}'."""
            elif isinstance(training_flux_units, unyt_quantity):
                if flux_units is None:
                    # Check .units attributes of columns are convertible
                    for col in photometry_columns:
                        if col not in observations.columns:
                            raise ValueError(f"Column '{col}' not found in observations.")
                        if (
                            not unyt_array(
                                observations[col].values,
                                units=observations[col].units,
                            )
                            .to(training_flux_units)
                            .check()
                        ):
                            raise ValueError(
                                f"""Column '{col}' units '{observations[col].units}' cannot
                            be converted to training flux units '{training_flux_units}'."""
                            )
                else:
                    # Check if the flux units are convertible to the training flux units
                    if not unyt_array(1.0, units=flux_units).to(training_flux_units).check():
                        raise ValueError(
                            f"""Flux units '{flux_units}' cannot be converted to
                            training flux units '{training_flux_units}'."""
                        )

            else:
                raise TypeError("Flux units must be a string or unyt_quantity.")

        # Create empty output feature array
        nrows = observations.shape[0]
        ncols = np.shape(self.feature_array)[1]  # Number of features in the training data

        feature_array = np.zeros((ncols, nrows), dtype=np.float32)

        # Loop afer columns, do conversion(s) and fill the feature array

        for i, col in enumerate(photometry_columns):
            # get index from self.feature_names
            if col not in observations.columns:
                raise ValueError(
                    f"""Column '{col}' not found in observations.
                    Please provide a mapping for all photometry filters."""
                )
            index = self.feature_names.index(columns_to_feature_names[col])
            # Convert the column to the training flux units
            if flux_units is not None and isinstance(flux_units, unyt_quantity):
                feature_array[index, :] = (
                    unyt_array(observations[col].values, units=observations[col].units)
                    .to(flux_units)
                    .value
                )
            else:
                feature_array[index, :] = observations[col].values

        # Fill in the errors if applicable
        for i, col in enumerate(feature_array_flags["error_names"]):
            observation_col = feature_names_to_columns[col]
            if observation_col not in observations.columns:
                raise ValueError(
                    f"""Column '{observation_col}' not found in observations.
                    Please provide a mapping for all errors."""
                )
            index = self.feature_names.index(col)
            # Convert the column to the training flux units
            if flux_units is not None and isinstance(flux_units, unyt_quantity):
                feature_array[index, :] = (
                    unyt_array(
                        observations[observation_col].values,
                        units=observations[observation_col].units,
                    )
                    .to(flux_units)
                    .value
                )
            else:
                feature_array[index, :] = observations[observation_col].values

        # Check for errors which are NAN when the corresponding fluxes aren't
        for i, col in enumerate(feature_array_flags["error_names"]):
            index = self.feature_names.index(col)
            flux_col = col.replace("unc_", "")
            flux_index = self.feature_names.index(flux_col)
            error_values = feature_array[index, :]
            flux_values = feature_array[flux_index, :]
            mask = np.isnan(error_values) & ~np.isnan(flux_values)
            if np.sum(mask) > 0:
                raise ValueError(
                    f"""Error column '{col}' contains NaN values where the
                    corresponding flux column '{flux_col}' does not."""
                    f"""{np.sum(mask)} NaN values found."""
                )
        # Fill in the flags if applicable
        for i, col in enumerate(feature_array_flags["flag_names"]):
            if col not in observations.columns:
                raise ValueError(
                    f"""Column '{col}' not found in observations.
                    Please provide a mapping for all flags."""
                )
            index = self.feature_names.index(col)
            # Convert the column to the training flux units
            feature_array[index, :] = observations[col].values

        # Fill in the normalization factor if applicable
        if feature_array_flags["norm_name"] is not None:
            if feature_array_flags["norm_name"] not in observations.columns:
                raise ValueError(
                    f"""Column '{feature_array_flags["norm_name"]}'
                    not found in observations.
                    Please provide a mapping for the normalization factor."""
                )
            index = self.feature_names.index(feature_array_flags["norm_name"])
            column_name = feature_names_to_columns[feature_array_flags["norm_name"]]
            # Convert the column to the training flux units
            if flux_units is not None:
                feature_array[index, :] = (
                    unyt_array(
                        observations[column_name].values,
                        units=observations[column_name].units,
                    )
                    .to(flux_units)
                    .value
                )
            else:
                feature_array[index, :] = observations[column_name].values

        # Fill in the extra features if applicable
        for i, col in enumerate(feature_array_flags["extra_features"]):
            if col not in observations.columns:
                raise ValueError(
                    f"""Column '{col}' not found in observations.
                    Please provide a mapping for all extra features."""
                )
            index = self.feature_names.index(col)
            # Convert the column to the training flux units
            feature_array[index, :] = observations[col].values

        # To Do:
        # Apply normalisation to flux columns if neccessary
        # replace NaN and Inf values with the missing_flux_value if applicable
        # apply min_flux percentage error if applicable
        # Somehow apply some things from EmpiricalNoise Models if applicable
        # Need to distinguish between NAN (dropout) and -99/blank - missing data
        # UPDATE; The below should apply some scalings from noise models if used.
        # Including SNR handling, upper/lower limits, min error etc.
        # Check if the flux units match the training data
        training_flux_units = feature_array_flags["normed_flux_units"]

        if feature_array_flags["empirical_noise_models"] is not None:
            for pos, (model_name, val) in enumerate(
                feature_array_flags["empirical_noise_models"].items()
            ):
                if isinstance(val, str):
                    if pos == 0:
                        logger.info("Loading noise models from HDF5.")
                    try:
                        empirical_model = load_unc_model_from_hdf5(
                            filepath=val, group_name=model_name
                        )
                    except (FileNotFoundError, KeyError, TypeError):
                        # If we've moved computer, the path will be wrong, but we may still
                        # be able to find the correct path.
                        filename = os.path.basename(val)
                        new_path = f"{code_path}/models/{self.name}/{filename}"
                        empirical_model = None
                        if os.path.exists(new_path):
                            try:
                                empirical_model = load_unc_model_from_hdf5(
                                    filepath=new_path, group_name=model_name
                                )
                            except Exception as e2:
                                empirical_model = None
                        if empirical_model is None:
                            logger.warning(
                                f"Could not load noise model '{model_name}' from {val} "
                                f"(fallback tried: {new_path}). Skipping. Error: {e}"
                            )
                            continue
                elif isinstance(val, UncertaintyModel):
                    empirical_model = val
                else:
                    raise TypeError(
                        f"Invalid empirical noise model type: {type(val)}. "
                        "Expected tuple or EmpiricalNoiseModel instance."
                    )

                if model_name in columns_to_feature_names:
                    model_name = columns_to_feature_names[model_name]

                if model_name in feature_names_to_columns:
                    # Apply the empirical noise model to the feature array
                    index = self.feature_names.index(model_name)
                    flux_column = feature_array[index, :]
                    # Resolve matching error column robustly
                    eindex = None
                    fname = model_name.split(".")[-1]
                    for ecol in feature_array_flags["error_names"]:
                        if ecol == f"unc_{model_name}" or (
                            ecol.startswith("unc_") and ecol.endswith(fname)
                        ):
                            eindex = self.feature_names.index(ecol)
                            break
                    if eindex is None:
                        logger.warning(
                            f"No matching error column for '{model_name}'. Skipping empirical scaling."
                        )
                        continue
                    error_column = feature_array[eindex, :]
                    new_flux, new_error = empirical_model.apply_scalings(
                        flux_column,
                        error_column,
                        true_flux_units=flux_units,
                        out_units=feature_array_flags["normed_flux_units"],
                    )
                    # print(np.sum(~np.isfinite(new_flux)), np.sum(~np.isfinite(new_error)))
                    feature_array[index, :] = new_flux
                    feature_array[eindex, :] = new_error
                    # Conversion handled by the noise model.

                else:
                    logger.warning(
                        f"Empirical noise model '{model_name}' not found in feature names."
                    )
            flux_units = feature_array_flags["normed_flux_units"]

        if feature_array_flags["normalize_method"] is not None:
            # Normalize the photometry columns
            norm_index = self.feature_names.index(feature_array_flags["norm_name"])
            normalization_factor = feature_array[norm_index, :]
            if feature_array_flags["normed_flux_units"] == "AB":
                normalization_factor = 10 ** ((23.9 - normalization_factor) / 2.5)
                method = np.subtract
            else:
                normalization_factor = (
                    unyt_array(
                        normalization_factor,
                        units=feature_array_flags["normalization_unit"],
                    )
                    .to(feature_array_flags["normed_flux_units"])
                    .value
                )
                method = np.divide

            # Normalize the photometry columns
            for i, col in enumerate(photometry_columns):
                index = self.feature_names.index(columns_to_feature_names[col])
                feature_array[index, :] = method(feature_array[index, :], normalization_factor)

        removed_data = np.zeros(len(feature_array[0]), dtype=bool)

        missing_mask = feature_array == missing_data_flag
        if np.isnan(missing_data_flag):
            missing_mask = np.isnan(feature_array)

        if feature_array_flags["simulate_missing_fluxes"] and not ignore_missing:
            num = np.sum(missing_mask)
            logger.info(
                f"""Replacing {num} NaN or Inf values with
                {feature_array_flags["missing_flux_value"]}."""
            )
            feature_array[missing_mask] = feature_array_flags["missing_flux_value"]
        elif ignore_missing:
            removed_data[:] = False  # keep all data
            pass
        else:
            nmiss = np.sum(missing_mask)
            if nmiss > 0:
                logger.info(f"Removing {nmiss} observations with missing data.")
            removed_data[missing_mask.any(axis=0)] = True

        feature_array = feature_array[:, ~removed_data]
        missing_mask = missing_mask[:, ~removed_data]

        # Replace NaN and Inf values with the missing_flux_value if applicable
        if feature_array_flags["remove_nan_inf"]:
            # Replace NaN and Inf values with the missing_flux_value
            mask = ~np.isfinite(feature_array)
            num = np.sum(mask)
            if num > 0 and ignore_missing:
                logger.warning(f"Replacing {num} NaN or Inf values with {missing_data_flag}.")

        mask = feature_array > feature_array_flags["norm_mag_limit"]
        # Exclude missing data points which hav been filled with missing_flux_value
        mask &= ~missing_mask
        feature_array[mask] = feature_array_flags["norm_mag_limit"]

        # if str, should match flux units.
        if isinstance(training_flux_units, str):
            assert flux_units == training_flux_units, (
                f"Flux units '{flux_units}' do not match \
                training data units '{training_flux_units}'."
            )
        elif isinstance(training_flux_units, unyt_quantity):
            if flux_units is None:
                # Check .units attributes of columns are convertible
                for col in photometry_columns:
                    if col not in observations.columns:
                        raise ValueError(f"Column '{col}' not found in observations.")
                    if (
                        not unyt_array(
                            observations[col].values,
                            units=observations[col].units,
                        )
                        .to(training_flux_units)
                        .check()
                    ):
                        raise ValueError(
                            f"""Column '{col}' units '{observations[col].units}' cannot
                        be converted to training flux units '{training_flux_units}'."""
                        )
            else:
                # Check if the flux units are convertible to the training flux units
                if not unyt_array(1.0, units=flux_units).to(training_flux_units).check():
                    raise ValueError(
                        f"""Flux units '{flux_units}' cannot be converted to
                        training flux units '{training_flux_units}'."""
                    )

        else:
            raise TypeError("Flux units must be a string or unyt_quantity.")

        logger.debug(f"Number of NANs in feature array: {np.sum(~np.isfinite(feature_array))}")

        # Cast negative or positive inf to NANs
        feature_array[~np.isfinite(feature_array)] = np.nan

        return feature_array.T, removed_data

    def fit_catalogue(
        self,
        observations: Union[Table, pd.DataFrame],
        columns_to_feature_names: dict = None,
        flux_units: Union[str, unyt_quantity, None] = None,
        missing_data_flag: Any = -99,
        quantiles: list = [0.16, 0.5, 0.84],
        sample_method: str = "direct",
        sample_kwargs: dict = {},
        num_samples: int = 1000,
        timeout_seconds_per_row: int = 5,
        override_transformations: dict = {},
        append_to_input: bool = True,
        return_feature_array: bool = False,
        recover_SEDs: bool = False,
        plot_SEDs: bool = False,
        check_out_of_distribution: bool = True,
        simulator: Optional[GalaxySimulator] = None,
        outlier_methods: list = [
            "iforest",
            "feature_bagging",
            "ecod",
            "knn",
            "lof",
            "gmm",
            "mcd",
            "kde",
        ],
        missing_data_mcmc: bool = False,
        missing_data_mcmc_params: dict = {
            "ini_chi": 5.0,
            "max_chi2": 50.0,
            "nmc": 100,
            "nposterior": 1000,
            "tmax_all": 10,
            "verbose": True,
        },
        return_full_samples: bool = False,
        log_times: bool = False,
        use_temp_samples: bool = False,
        **kwargs,
    ):
        r"""Infer posteriors for observational data.

        This method creates a feature array from the observations and then samples
        the posterior distributions of the fitted parameters based on the feature array.

        Parameters
        ----------

        observations : Union[Table, pd.DataFrame]
            The observational data to create the feature array from.
        columns_to_feature_names : dict
            A dictionary mapping the column names in the observations to feature names.
        flux_units : Union[str, unyt_quantity, None]
            The units of the flux in the observations. If None, the units will be taken
            from the column metadata if available and otherwise assumed to be the same as
            the training data.
        missing_data_flag : Any
            Value in columns delineating missing data. Depending on setup of features,
            this may be flagged to model or the galaxy will be ignored entirely.
        quantiles : list
            List of quantiles to compute from the posterior samples.
        sample_method : str
            The method to use for sampling the posterior distributions.
        sample_kwargs : dict
            Additional keyword arguments to pass to the sampling method.
        num_samples : int
            The number of samples to draw from the posterior distributions.
        timeout_seconds_per_row : int
            The timeout in seconds for sampling each row of the feature array.
        override_transformations : dict
            A dictionary of transformations to override the defaults in
            self.feature_array_flags. This can be used to change the normalization method,
            extra features, etc.
        append_to_input : bool
            If True, append the quantiles to the input observations DataFrame/Table.
        return_feature_array : bool
            If True, return the feature array and the mask instead of the observations.
        recover_seds : bool
            If True, recover the SEDs from the posterior samples. This will run
            self.recover_SED, which will use a simulator to recreate the Synthesizer
            model and recover the SEDs from the posterior samples.
        plot_SEDs : bool
            If True, plot the recovered SEDs. This will only work if recover_seds is True.
        check_out_of_distribution : bool
            If True, check if the feature array is in distribution using the
            robust Mahalanobis method. This will raise an error if any row is out of
            distribution. If False, no check is performed.
        simulator : Optional[GalaxySimulator]
            simulator: A GalaxySimulator object to use for generating the SED. Optional.
            Will attempt to create one from the library if not provided, or use the existing one.
        outlier_methods : list
            List of outlier detection methods to use from PyOD.
            See PyOD documentation for available methods.
        missing_data_mcmc : bool
            If True, use SBI++ to marginalize over missing data using KDE.
        return_full_samples : bool
            If True, return the full posterior samples as well as the quantiles.
        log_times : bool
            If True, log the time taken to sample each row of the feature array.
        use_temp_samples : bool
            If True, try to use the samples stored in self. This is useful for
            debugging and testing, or jumping straight to SED recovery.
        \**kwargs: Optional params - only used internally for simformer model.

        Returns:
        -------
        observations : Union[Table, pd.DataFrame]
            The original observations with additional columns for the quantiles of the
            fitted parameters, or a new Table with the quantiles if
            append_to_input is False.
        """
        feature_array, obs_mask = self.create_features_from_observations(
            observations,
            columns_to_feature_names=columns_to_feature_names,
            flux_units=flux_units,
            missing_data_flag=missing_data_flag,
            override_transformations=override_transformations,
            ignore_missing=missing_data_mcmc,
        )

        # Check length of feature array matches e.g. self._X_test

        if self._X_test is not None and feature_array.shape[1] != len(self.feature_names):
            raise ValueError(
                f"""Feature array samples have {feature_array.shape[1]} features,
                but the model was trained on {len(self.feature_names)} features."""
            )

        if check_out_of_distribution:
            """self.test_in_distribution(
                feature_array, direction="in", method="robust_mahalanobis", confidence=0.99
            )"""
            outliers = self.test_in_distribution_pyod(
                feature_array, direction="in", methods=outlier_methods, contamination=0.01
            )
            if np.any(outliers):
                logger.warning(f"{np.sum(outliers)} outlier(s) detected in the observational data.")

                temp = np.zeros(len(observations), dtype=bool)
                temp[~obs_mask] |= outliers
                temp[obs_mask] = True

                obs_mask = temp
        if return_feature_array:
            # If return_feature_array is True, return the feature array and the mask
            return feature_array, obs_mask

        reconstructed_photometry = np.zeros(
            (len(observations), len(self.feature_array_flags["raw_observation_names"]))
        )
        reconstructed_photometry[:] = np.nan

        mask_missing = np.zeros(len(observations), dtype=bool)
        if missing_data_mcmc:
            # Check for rows with missing data.
            if missing_data_flag is np.nan:
                mask_missing = np.any(np.isnan(feature_array), axis=1)
            else:
                mask_missing = np.any(feature_array == missing_data_flag, axis=1)
            if self.feature_array_flags["simulate_missing_fluxes"]:
                logger.warning(
                    "Using SBI++ to marginalze in a model trained with missing photometry may"
                    "not be the intended behaviour."
                )
            """
            output_samples = np.zeros(
                (len(self.simple_fitted_parameter_names), feature_array.shape[0], num_samples)
            )
            """

            if np.any(mask_missing):
                logger.info(
                    f"{np.sum(mask_missing)} rows with missing data found. Marginalizing over missing data using MCMC."  # noqa: E501
                )

                phot_handler = MissingPhotometryHandler.init_from_synference(
                    self, run_params=missing_data_mcmc_params
                )

                total_samples = (
                    missing_data_mcmc_params["nposterior"] * missing_data_mcmc_params["nmc"]
                )

                output_missing_samples = np.zeros(
                    (len(self.simple_fitted_parameter_names), np.sum(mask_missing), total_samples)
                )

                if self.feature_array_flags["include_errors_in_feature_array"]:
                    include_errors = True
                    error_names = self.feature_array_flags["error_names"]
                    err_indices = [self.feature_names.index(name) for name in list(error_names)]
                else:
                    include_errors = False

                observation_names = self.feature_array_flags["raw_observation_names"]

                observation_indices = [
                    self.feature_names.index(name) for name in list(observation_names)
                ]
                times = []
                for i, row in tqdm(
                    enumerate(np.where(mask_missing)[0]),
                    desc="Marginalizing over missing data...",
                    total=np.sum(mask_missing),
                ):
                    start = time.time()
                    if missing_data_flag is np.nan:
                        missing_data_idxs_i = np.where(np.isnan(feature_array[row, :]))[0]
                    else:
                        missing_data_idxs_i = np.where(feature_array[row, :] == missing_data_flag)[
                            0
                        ]
                    obs = feature_array[row, observation_indices]
                    if include_errors:
                        unc = feature_array[row, err_indices]
                    else:
                        unc = None

                    # Get other indices and make dictionary of extra features
                    extra = {}
                    for idx, name in enumerate(self.feature_names):
                        if idx not in observation_indices and idx not in err_indices:
                            extra[name] = feature_array[row, idx]

                    # 0 = present, 1=missing
                    missing_mask = [1 if i in missing_data_idxs_i else 0 for i in range(len(obs))]
                    missing_mask = np.array(missing_mask, dtype=bool)
                    obs = {
                        "mags_sbi": obs,
                        "mags_unc_sbi": unc,
                        "missing_mask": missing_mask,
                        "extra": extra,
                    }
                    output = phot_handler.process_observation(
                        obs,
                        true_flux_units=flux_units,
                        out_units=self.feature_array_flags["normed_flux_units"],
                    )  # noqa: E501

                    success = output["success"]
                    posterior_samples = output["posterior_samples"]
                    reconstructed_phot = output["reconstructed_photometry"]

                    # Reconstructed phot contains the provided phot as well. Keep only the
                    # reconstructed missing photometry.
                    reconstructed_phot[~missing_mask] = np.nan

                    if success:
                        # remove input from feature array, save indices, insert back
                        # after the rest are sampled
                        output_missing_samples[:, i, :] = posterior_samples.T

                        # Also save reconstructed photometry to output table.
                        reconstructed_photometry[row, :] = reconstructed_phot
                    else:
                        output_missing_samples[:, i, :] = np.nan
                        reconstructed_photometry[row, :] = np.nan
                        obs_mask[row] = True  # flag as outlier
                    end = time.time()
                    times.append(end - start)

                if log_times and len(times) > 0:
                    logger.info(
                        f"Median time per object: {np.median(times):.2f} seconds."
                        "16th and 84th percentiles of time per object:"
                        f" {np.percentile(times, 16):.2f}, {np.percentile(times, 84):.2f} seconds."
                    )
                    # TODO: Save generated photometry/errors to the output table.
                # Remove from feature array
                feature_array = feature_array[~mask_missing, :]

        skip = False
        if use_temp_samples and hasattr(self, "temp_samples"):
            if len(self.temp_samples[0]) != len(feature_array):
                raise ValueError(
                    f"Number of samples in self.temp_samples ({len(self.temp_samples)}) does not"
                    f" match number of valid observations ({len(feature_array)})."
                )
            skip = True

        if skip:
            samples_quant = self.temp_samples
        elif isinstance(self, Simformer_Fitter):
            samples = self.sample_posterior(
                X_test=feature_array,
                num_samples=num_samples,
                **kwargs,
            )
            samples_quant = samples

        else:
            samples = self.sample_posterior(
                X_test=feature_array,
                sample_method=sample_method,
                sample_kwargs=sample_kwargs,
                num_samples=num_samples,
                timeout_seconds_per_test=timeout_seconds_per_row,
                log_times=log_times,
            )

            samples_quant = samples.transpose(2, 0, 1)

        logger.info("Obtained posterior samples.")
        # Rearrange into correct shape for quantiles

        if append_to_input:
            # Append the quantiles to the input DataFrame
            table = observations.copy()
        else:
            table = Table()
            colnames = (
                observations.colnames
                if isinstance(observations, Table)
                else observations.columns.tolist()
            )  # noqa: E501
            table["ID"] = (
                observations["ID"] if "ID" in colnames else np.arange(len(observations)) + 1
            )  # noqa: E501
        # Do quantiles, get column names from self.simple_fitted_parameter_names
        # TODO: Check if different from priors!
        for i, param in enumerate(self.simple_fitted_parameter_names):
            samples_i = samples_quant[i, :, :]
            samples_q = np.quantile(samples_i, quantiles, axis=1)
            for j, quant in enumerate(samples_q):
                # need to expand quant with dummy values for masked obs if append_to_input
                # if append_to_input:
                full_quant = np.zeros(len(obs_mask)) + np.nan
                full_quant[~(obs_mask | mask_missing)] = quant
                quant = full_quant
                # else:

                table[f"{param}_{int(quantiles[j] * 100)}"] = quant
                table[f"{param}_{int(quantiles[j] * 100)}"][obs_mask] = np.nan

        if missing_data_mcmc and np.any(mask_missing):
            # Add a column to flag rows with missing data
            if append_to_input:
                table["has_missing_data"] = False
                table["has_missing_data"][mask_missing] = True
            else:
                table["has_missing_data"] = mask_missing
            # Now need to insert rows with missing data back into the output table.
            for i, param in enumerate(self.simple_fitted_parameter_names):
                samples_i = output_missing_samples[i, :, :]
                samples_q = np.quantile(samples_i, quantiles, axis=1)
                for j, quant in enumerate(samples_q):
                    colname = f"{param}_{int(quantiles[j] * 100)}"
                    table[colname][mask_missing] = quant

        # Add outlier flag if applicable
        if check_out_of_distribution:
            if append_to_input:
                table["is_outlier"] = False
                table["is_outlier"][obs_mask] = True
            else:
                table["is_outlier"] = obs_mask

        # Only for columns in reconstructed_phot which have missing data include them in the table
        if missing_data_mcmc and np.any(mask_missing):
            for i, name in enumerate(self.feature_array_flags["raw_observation_names"]):
                if np.any(reconstructed_photometry[:, i] != np.nan):
                    colname = f"predicted_{name}"
                    table[colname] = reconstructed_photometry[:, i]
                    table[colname][~mask_missing] = np.nan
                    # Set flux unit
                    table[colname].unit = self.feature_array_flags["normed_flux_units"]
                    if np.all(np.isnan(table[colname])):
                        table.remove_column(colname)

        if return_full_samples:
            # Make a dictionary of the full samples, with NaNs for masked observations
            full_samples = {}
            for row in range(len(observations)):
                if obs_mask[row] or (missing_data_mcmc and mask_missing[row]):
                    full_samples[row] = None
                else:
                    full_samples[row] = samples_quant[:, row - np.sum(obs_mask[:row]), :]
            return table, full_samples

        self.temp_samples = samples_quant

        output = {}
        if recover_SEDs:
            if not hasattr(self, "simulator"):
                self.recreate_simulator_from_library()

            samples_sed = np.transpose(samples_quant, (1, 2, 0))
            for pos, (obs_i, samples_i) in tqdm(
                enumerate(zip(feature_array, samples_sed)),
                total=len(feature_array),
                desc="Recovering SEDs from posterior samples...",
            ):
                extra_parameters = {}
                for param in self.feature_names:
                    if "." not in param:
                        extra_parameters[param] = obs_i[list(self.feature_names).index(param)]

                fnu_quantiles, wav, phot_fnu_draws, phot_wav, fig = self.recover_SED(
                    X_test=obs_i,
                    samples=samples_i,
                    num_samples=num_samples,
                    sample_method=sample_method,
                    sample_kwargs=sample_kwargs,
                    plot=plot_SEDs,
                    plot_name=f"{self.name}_SED_{pos}",
                    simulator=simulator,
                    extra_parameters=extra_parameters,
                )
                try:
                    id = table["ID"][pos]
                except KeyError:
                    id = pos

                output[id] = {
                    "wav": wav,
                    "fnu_quantiles": fnu_quantiles,
                    "phot_wav": phot_wav,
                    "phot_fnu_draws": phot_fnu_draws,
                    "fig": fig,
                }

                plt.show(block=False)
            return table, output

        return table

    def create_dataframe(self, data="all"):
        """Create a DataFrame from the training data.

        Parameters:
        -----------
        data : str
            'all' to include all data, 'photometry' for only photometry,
            'parameters' for only parameters

        """
        if data == "all":
            return pd.DataFrame(
                np.hstack(
                    (
                        copy.deepcopy(self.feature_array),
                        copy.deepcopy(self.fitted_parameter_array),
                    )
                ),
                columns=self.feature_names + self.simple_fitted_parameter_names,
            )
        elif data == "photometry":
            return pd.DataFrame(
                copy.deepcopy(self.fitted_parameter_array),
                columns=self.simple_fitted_parameter_names,
            )
        elif data == "parameters":
            return pd.DataFrame(copy.deepcopy(self.feature_array), columns=self.feature_names)
        else:
            raise ValueError("Invalid data type. Use 'all', 'photometry', or 'parameters'.")

    def split_dataset(
        self,
        train_fraction: float = 0.8,
        random_seed: int = None,
        verbose: bool = True,
    ) -> tuple:
        """Split the dataset into training and testing sets.

        Parameters:
            train_fraction: The fraction of the dataset to be used for training.
            random_seed: The random seed for reproducibility.

        Returns:
            A tuple containing the training and testing indices.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        if not self.has_features:
            raise ValueError("Feature array not created. Please create the feature array first.")

        num_samples = self.feature_array.shape[0]

        if verbose:
            logger.info(
                f"""Splitting dataset with {num_samples} samples into training"""
                f"""and testing sets with {train_fraction:.2f} train fraction."""
            )
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        train_size = int(num_samples * train_fraction)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        return train_indices, test_indices

    def create_priors(
        self,
        override_prior_ranges: dict = {},
        prior=CustomIndependentUniform,
        verbose: bool = True,
        debug_sample_acceptance: bool = False,
        extend_prior_range_pc: float = 0.0,
        set_self: bool = False,
    ):
        """Create parameter priors.

        Create the priors for the parameters.
        By default we will use the range of the parameters in the grid.
        If override_prior_ranges is provided for a given parameter,
        then that will be used instead.

        Parameters:
            override_prior_ranges: A dictionary containing the prior ranges
                for the parameters.
            prior: The prior distribution to be used.
            verbose: Whether to print the prior ranges.
            debug_sample_acceptance: Whether to print debug information
                about sample acceptance in the prior. Only used if prior is
                CustomIndependentUniform.
            extend_prior_range_pc: Percentage to extend the prior range for
                parameters. However will not be applied to extend parameters
                to negative values, or parameters close to unity.


        Returns:
            A prior object.

        """
        if not self.has_features and not self.has_simulator:
            raise ValueError(
                """Feature array not created and no simulator.
                Please create the feature array first."""
            )

        if self.fitted_parameter_array is None and not self.has_simulator:
            raise ValueError("Parameter grid not created. Please create the parameter grid first.")
        if self.fitted_parameter_names is None and not self.has_simulator:
            raise ValueError(
                "Parameter names not created. Please create the parameter names first."
            )

        if (
            len(override_prior_ranges) < len(self.fitted_parameter_names)
            and self.fitted_parameter_array is None
        ):
            raise ValueError(
                f"""Not enough prior ranges provided for online training.
                {len(override_prior_ranges)} provided,
                {len(self.fitted_parameter_names)} expected."""
            )

        # Create the priors for the parameters
        low = []
        high = []
        for i, param in enumerate(self.fitted_parameter_names):
            if param in override_prior_ranges:
                lo = override_prior_ranges[param][0]
                hi = override_prior_ranges[param][1]
            elif extend_prior_range_pc > 0.0:
                param_min = np.min(self.fitted_parameter_array[:, i])
                param_max = np.max(self.fitted_parameter_array[:, i])
                param_range = param_max - param_min
                extend = param_range * extend_prior_range_pc / 100.0
                lo = param_min - extend
                hi = param_max + extend
                # Don't extend to negative values
                if lo < 0 and param_min >= 0:
                    lo = 0.0
                # Don't extend parameters close to unity
                if np.isclose(param_max, 1.0, atol=0.05) and (hi > 1.0):
                    hi = 1.0
            else:
                lo = np.min(self.fitted_parameter_array[:, i])
                hi = np.max(self.fitted_parameter_array[:, i])

            if any(np.isnan([lo, hi])):
                raise ValueError(f"NAN value found in prior range for parameter '{param}'.")
            if lo == hi:
                raise ValueError(
                    f"Prior range for parameter '{param}' is zero ({lo} == {hi}). "
                    "Please provide a non-zero range."
                )

            low.append(lo)
            high.append(hi)

        low = np.array(low)
        high = np.array(high)

        # Print prior ranges
        if verbose:
            logger.info("---------------------------------------------")
            logger.info("Prior ranges:")
            logger.info("---------------------------------------------")
            for i, param in enumerate(self.fitted_parameter_names):
                if self.fitted_parameter_units is not None:
                    unit = f" [{self.fitted_parameter_units[i]}]"
                else:
                    unit = ""
                logger.info(f"{param}: {low[i]:.2f} - {high[i]:.2f}{unit}")
            logger.info("---------------------------------------------")

        low = torch.tensor(low, dtype=torch.float32, device=self.device)
        high = torch.tensor(high, dtype=torch.float32, device=self.device)

        extra_args = {}
        if issubclass(prior, CustomIndependentUniform):
            extra_args["name_list"] = list(self.fitted_parameter_names)
            extra_args["verbose"] = debug_sample_acceptance
        # Create the priors
        param_prior = prior(low=low, high=high, device=self.device, **extra_args)

        if isinstance(param_prior, CustomIndependentUniform):
            from sbi.utils import process_prior
            from torch.distributions import Independent

            param_prior = Independent(param_prior, 1)
            logger.info("Processing prior...")
            param_prior, _, _ = process_prior(param_prior)

        if set_self:
            self._prior = param_prior
        return param_prior

    def create_restricted_priors(
        self,
        prior: Optional[CustomIndependentUniform] = None,
        set_self=True,
    ):
        """Create restricted priors for the parameters."""
        assert self.has_features, (
            "Feature array not created. Please create the feature array first."
        )
        if prior is None:
            prior = self.create_priors()

        from sbi.utils import RestrictionEstimator

        X = self.feature_array
        y = self.fitted_parameter_array

        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device)

        restriction_estimator = RestrictionEstimator(prior=prior)
        restriction_estimator.append_simulations(y, X)
        classifier = restriction_estimator.train()

        if set_self:
            self.restricted_prior = restriction_estimator
            self.restricted_classifier = classifier

        return restriction_estimator, classifier

    def optimize_sbi(
        self,
        study_name: str,
        suggested_hyperparameters: dict = {
            "learning_rate": [1e-6, 1e-3],
            "hidden_features": [12, 200],
            "num_components": [2, 16],
            "training_batch_size": [32, 128],
            "num_transforms": [1, 4],
            "stop_after_epochs": [10, 30],
            "clip_max_norm": [0.1, 5.0],
            "validation_fraction": [0.1, 0.3],
        },
        fixed_hyperparameters: dict = {"n_nets": 1, "model_type": "mdn"},
        n_trials: int = 100,
        n_jobs: int = 1,
        random_seed: int = 42,
        verbose: bool = False,
        train_test_fraction=0.9,
        persistent_storage: bool = False,
        out_dir: str = f"{code_path}/models/",
        score_metrics: Union[str, List[str]] = "log_prob-pit",
        direction: str = "maximize",
        timeout_minutes_trial_sampling: float = 120.0,
        sql_db_path: Optional[str] = None,
    ) -> None:
        """Use Optuna to optimize the SBI model hyperparameters.

        Possible hyperparameters to optimize:
        - n_nets: Number of networks to use in the ensemble.
        - model_type: Type of model to use. Either 'mdn' or 'maf'.
        - hidden_features: Number of hidden features in the neural network.
        - num_components: Number of components in the mixture density network.
        - num_transforms: Number of transforms in the masked autoregressive flow.
        - training_batch_size: Batch size for training.
        - learning_rate: Learning rate for the optimizer.
        - validation_fraction: Fraction of the training set to use for validation.
        - stop_after_epochs: Number of epochs without improvement before stopping.
        - clip_max_norm: Maximum norm for gradient clipping.

        Parameters:
        ------------
        - study_name: Name of the Optuna study.
        - suggested_hyperparameters: Dictionary of hyperparameters to suggest.
            Keys are hyperparameter names and values are lists of possible values.
        - fixed_hyperparameters: Dictionary of hyperparameters to fix.
            Keys are hyperparameter names and values are fixed values.
        - n_trials: Number of trials to run in the optimization.
        - n_jobs: Number of parallel jobs to run.
            Note that Optuna uses the threading backend, not true parallelism,
            so the Python GIL can still be a bottleneck. See
            https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-parallelize-optimization
            for discussion.
        - random_seed: Random seed for reproducibility.
        - verbose: Whether to print progress and results.
        - persistent_storage: Whether to use persistent storage for the study.
        - out_dir: Directory to save the study results.
        - score_metrics: Metrics to use for scoring the trials. Either a string
            or a list of metrics.
        - direction: Direction of optimization, either 'minimize' or 'maximize',
            or a list of directions if using multi-objective optimization.
        - timeout_minutes_trial_sampling: Timeout in minutes for each trial sampling.
            e.g. if sampling gets stuck, will prune this trial.
        - sql_db_path: Optional path to an existing MySQL database for Optuna.
            If you don't have one set up, leave as None to create a SQLite database.

        """
        if not self.has_features and not self.has_simulator:
            raise ValueError(
                """Feature array not created and no simulator.
                Please create the feature array first."""
            )

        if self.fitted_parameter_array is None and not self.has_simulator:
            raise ValueError("Parameter grid not created. Please create the parameter grid first.")

        if self.fitted_parameter_names is None and not self.has_simulator:
            raise ValueError(
                "Parameter names not created. Please create the parameter names first."
            )

        out_dir = os.path.join(os.path.abspath(out_dir), self.name)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if isinstance(direction, (list, tuple)):
            directions = copy.deepcopy(direction)
            direction = None
        else:
            directions = None

        assert (direction is not None) or (directions is not None), (
            "Must provide optimization direction(s)."
        )

        if persistent_storage:
            if sql_db_path is None:
                sqlite_path = os.path.join(out_dir, f"{study_name}_optuna_study.db")
                url = create_sqlite_db(sqlite_path)
            else:
                url = create_database_universal(study_name, full_url=sql_db_path)

            study = optuna.create_study(
                study_name=study_name,
                storage=url,
                direction=direction,
                directions=directions,
                load_if_exists=True,
            )
        else:
            study = optuna.create_study(
                study_name=study_name,
                direction=direction,
                directions=directions,
            )

        if self.has_features:
            self._train_indices, self._test_indices = self.split_dataset(
                random_seed=random_seed,
                verbose=verbose,
                train_fraction=train_test_fraction,
            )
            train_indices = self._train_indices
            test_indices = self._test_indices

            X_test, y_test = None, None

        else:
            X_test, y_test = self.generate_pairs_from_simulator(5000)
            train_indices, test_indices = None, None

        def objective_func(trial):
            """Setup function here to use shared parameters."""
            return self._run_evaluate_sbi(
                trial=trial,
                train_indices=train_indices,
                test_indices=test_indices,
                X_test=X_test,
                y_test=y_test,
                suggested_hyperparameters=suggested_hyperparameters,
                verbose=verbose,
                score=score_metrics,
                timeout_minutes_trial_sampling=timeout_minutes_trial_sampling,
                **fixed_hyperparameters,
            )

        study.optimize(
            objective_func,
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=True,
            gc_after_trial=True,
        )

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        logger.info("Study statistics: ")
        logger.info(f"  Number of finished trials: {len(study.trials)}")
        logger.info(f"  Number of pruned trials: {len(pruned_trials)}")
        logger.info(f"  Number of complete trials: {len(complete_trials)}")

        logger.info("Best trial:")
        trial = study.best_trial

        logger.info(f"  Value: {trial.value}")

        logger.info("  Params: ")
        for key, value in trial.params.items():
            logger.info(f"    {key}: {value}")

        # Save the study to a file
        study_path = os.path.join(out_dir, f"{study_name}_optuna_study_{self._timestamp}.pkl")
        dump(study, study_path, compress=3)

    def test_in_distribution_pyod(
        self,
        X_test: np.ndarray,
        methods=["lof", "isolation_forest"],
        contamination=0.01,
        direction="in",
        combination_method="majority",
    ):
        """Test if X_test is in distribution of self.feature_array using pyod.

        If direction is "in", then we check if X_test is within the
        distribution of self.feature_array. If 'out' we check if self.feature_array is
        within the distribution of X_test.

        Parameters:
        ------------
        X_test : np.ndarray
            The test data to check for in-distribution or out-of-distribution.
        methods : list
            List of methods to use for outlier detection.
            See https://pyod.readthedocs.io/en/latest/pyod.models.html
        contamination : float
            Expected proportion of outliers (for applicable methods)
        direction : str
            Direction of the test, either 'in' or 'out'.
        combination_method : str
            Method to combine the results from different methods.
            Options are 'majority' (default), 'any', 'all', 'none'.
        """
        assert self.has_features, (
            "Feature array not created. Please create the feature array first."
        )

        if not isinstance(X_test, np.ndarray):
            raise TypeError("X_test must be a numpy array.")

        assert direction in ["in", "out"], "Direction must be either 'in' or 'out'."

        if direction == "in":
            dist1 = self.feature_array
            dist2 = X_test
        else:
            dist1 = X_test
            dist2 = self.feature_array

        results = detect_outliers_pyod(
            dist1,
            dist2,
            methods=methods,
            contamination=contamination,
            combination=combination_method,
            return_scores=False,
        )

        return results

    def test_in_distribution(
        self,
        X_test: np.ndarray,
        method="robust_mahalanobis",
        direction="in",
        contamination=0.1,
        n_neighbors=20,
        threshold=None,
        confidence=0.95,
        n_components=None,
        plot=True,
        feature_breakdown=False,
        **kwargs,
    ):
        """Test if X_test is in distribution of self.feature_array.

        If `direction` is "in", then we check if `X_test` is within the
        distribution of `self.feature_array`. If 'out', we check if
        `self.feature_array` is within the distribution of `X_test`.

        .. note::
           This function is a wrapper for `utils.detect_outliers`.

        Methods available:
            - 'robust_mahalanobis': Uses robust Mahalanobis distance.
            - 'mahalanobis': Uses standard Mahalanobis distance.
            - 'lof': Uses Local Outlier Factor.
            - 'isolation_forest': Uses Isolation Forest.
            - 'one_class_svm': Uses One-Class SVM.
            - 'pca': Uses PCA.
            - 'hotelling_t2': Uses Hotelling's T-squared test.
            - 'kde': Uses Kernel Density Estimation.

        Args:
            X_test (np.ndarray): The test data to check for in-distribution
                or out-of-distribution.
            method (str, optional): The method to use for outlier detection.
                Defaults to 'robust_mahalanobis'.
            direction (str, optional): Direction of the test. 'in' (is `X_test`
                in `self.feature_array`?) or 'out' (is `self.feature_array`
                in `X_test`?). Defaults to 'in'.
            contamination (float, optional): Expected proportion of outliers
                (for applicable methods). Defaults to 0.1.
            n_neighbors (int, optional): Number of neighbors for LOF.
                Defaults to 20.
            threshold (float, optional): Manual threshold for outlier
                detection. Defaults to None.
            confidence (float, optional): Confidence level for statistical
                tests. Defaults to 0.95.
            n_components (int, optional): Number of components for PCA
                (if None, uses all). Defaults to None.
            plot (bool, optional): Whether to plot the results of the
                outlier detection. Defaults to True.
            feature_breakdown (bool, optional): If True, will analyze
                feature contributions to outlier scores. Only works for
                'robust_mahalanobis', 'mahalanobis', 'euclidean'.
                Defaults to False.
            **kwargs (Any): Additional parameters passed to the underlying
                `detect_outliers` method.

        Returns:
            Dict[str, Any]:
                A dictionary containing the outlier detection results:

                - 'outlier_mask' (np.ndarray): Boolean array indicating outliers.
                - 'scores' (np.ndarray): Outlier scores for each observation.
                - 'threshold_used' (float): The threshold value used for detection.
                - 'method_info' (dict): Additional method-specific information.
        """
        assert self.has_features, (
            "Feature array not created. Please create the feature array first."
        )

        if not isinstance(X_test, np.ndarray):
            raise TypeError("X_test must be a numpy array.")

        assert direction in ["in", "out"], "Direction must be either 'in' or 'out'."

        if direction == "in":
            dist1 = self.feature_array
            dist2 = X_test
        else:
            dist1 = X_test
            dist2 = self.feature_array

        if not feature_breakdown:
            results = detect_outliers(
                dist1,
                dist2,
                method=method,
                contamination=contamination,
                n_neighbors=n_neighbors,
                threshold=threshold,
                confidence=confidence,
                n_components=n_components,
                plot=plot,
                **kwargs,
            )
        else:
            if method not in ["robust_mahalanobis", "mahalanobis", "euclidean", "all"]:
                raise ValueError(
                    "Feature breakdown only works with 'robust_mahalanobis', 'mahalanobis', or 'euclidean' methods."  # noqa E501
                )
            if method == "all":
                results = compare_methods_feature_importance(
                    dist1, dist2, feature_names=self.feature_names
                )
            else:
                results = analyze_feature_contributions(
                    dist1, dist2, feature_names=self.feature_names, method=method
                )

        return results

    def _run_evaluate_sbi(
        self,
        trial: optuna.Trial,
        suggested_hyperparameters: dict,
        verbose: bool = False,
        train_indices: np.ndarray = None,
        test_indices: np.ndarray = None,
        X_test: np.ndarray = None,
        y_test: np.ndarray = None,
        score: Union[str, List[str]] = "log_prob-pit",
        timeout_minutes_trial_sampling: float = 120.0,
        **fixed_hyperparameters,
    ) -> tuple:
        """Objective function to run the SBI training and evaluation."""
        parameters = {}

        for key, value in suggested_hyperparameters.items():
            if isinstance(value, list):
                assert len(value) == 2 or isinstance(
                    value[0], str
                ), f"""Value for {key} should be a list of length 2 or
                      list of strings. Got {value}"""
                if isinstance(value[0], str):
                    parameters[key] = trial.suggest_categorical(key, value)
                else:
                    if isinstance(value[0], int):
                        parameters[key] = trial.suggest_int(key, value[0], value[1])
                    elif isinstance(value[0], float):
                        # Check if small
                        if np.max(value) / np.min(value) > 100:
                            log = True
                        else:
                            log = False
                        parameters[key] = trial.suggest_float(key, value[0], value[1], log=log)

        parameters.update(fixed_hyperparameters)

        for key, param in fixed_hyperparameters.items():
            trial.set_user_attr(key, param)

        logger.info(f"Trial {trial.number}: Starting with {parameters=}")

        posterior, stats = self.run_single_sbi(
            train_indices=train_indices,
            test_indices=test_indices,
            set_self=False,
            save_model=False,
            plot=False,
            verbose=verbose,
            evaluate_model=False,
            load_existing_model=False,
            **parameters,
        )

        if X_test is None:
            X_test = self.feature_array[test_indices]
        if y_test is None:
            y_test = self.fitted_parameter_array[test_indices]

        timeout_seconds = timeout_minutes_trial_sampling * 60  # Convert minutes to seconds

        def timeout_handler(signum, frame):
            raise optuna.exceptions.TrialPruned(
                f"Execution exceeded timeout of {timeout_seconds} seconds"
            )

        # Set the signal handler
        try:
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_seconds))
        except ValueError:
            # If signal.alarm is not available (e.g., on Windows), we skip the timeout
            old_handler = None

        try:
            if isinstance(score, str):
                if score == "log_prob-pit":
                    # Do the evaluation
                    scoreval = np.mean(self.log_prob(X_test, y_test, posterior))

                    # Continue with the PIT calculation and adjust the score
                    pit = self.calculate_PIT(X_test, y_test, num_samples=5000, posteriors=posterior)
                    dpit_max = np.max(np.abs(pit - np.linspace(0, 1, len(pit))))
                    scoreval += -0.5 * np.log(dpit_max)
                elif score == "log_prob":
                    scoreval = np.mean(self.log_prob(X_test, y_test, posterior))
                elif score == "loss":
                    scoreval = np.mean(-self.log_prob(X_test, y_test, posterior))
                elif callable(score):
                    scoreval = score(posterior, X_test, y_test)
                else:
                    raise ValueError(f"Unknown score type: {score}")

                return score

            elif isinstance(score, list):
                scores = []
                for s in score:
                    if s == "log_prob":
                        scores.append(np.mean(self.log_prob(X_test, y_test, posterior)))
                    elif s == "tarp":
                        scores.append(self.calculate_TARP(X_test, y_test, posteriors=posterior))
                    else:
                        raise ValueError(f"Unknown score type: {s}")
                return scores
            else:
                raise ValueError(f"Score should be a string or a list of strings. Got {score}")
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            signal.alarm(0)  # Disable the alarm
            raise e
        finally:
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)

    def run_single_simformer(
        self,
        train_test_fraction: float = 0.8,
        random_seed: int = None,
        train_indices: np.ndarray = None,
        test_indices: np.ndarray = None,
        save_model: bool = True,
        verbose: bool = True,
        out_dir: str = f"{code_path}/models/name/",
        plot: bool = True,
        name_append: str = "timestamp",
        save_method: str = "dill",
        set_self: bool = True,
        override_prior_ranges: dict = {},
        load_existing_model: bool = True,
        use_existing_indices: bool = True,
        evaluate_model: bool = True,
        max_num_epochs: int = 100_000,
        sde_type: str = "ve",
        simformer_type="score",
        model_kwargs: dict = {},
        learning_rate: float = 0.0005,
        training_batch_size: int = 200,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        clip_max_norm: float = 5.0,
        training_args: dict = {},
    ) -> tuple:
        r"""Trains a single Simformer model using the SBI implementation.

        May need to be on this branch:
        https://github.com/sbi-dev/sbi/pull/1621/

        Parameters:
            train_test_fraction: Fraction of the dataset to be used for training.
            random_seed: Random seed for reproducibility.
            train_indices: Indices of the training set.
            test_indices: Indices of the test set. If None, no test set is used.
            save_model: Whether to save the trained model.
            verbose: Whether to print verbose output.
            out_dir: Directory to save the model.
            plot: Whether to plot the results.
            name_append: String to append to the model name.
            set_self: Whether to set the self attribute.
            override_prior_ranges: Dictionary to override prior ranges.
            load_existing_model: Whether to load an existing model.
            use_existing_indices: Whether to use existing indices.
            evaluate_model: Whether to evaluate the model.
            max_num_epochs: Maximum number of epochs to train the model.
            sde_type: Type of SDE to use ('ve','vp' or 'subvp'). Not used for flow matching.
            simformer_type: Type of Simformer to use ('score' or 'flow').
            model_kwargs: Additional keyword arguments to pass to the Simformer builder.
                Available kwargs and defaults are:
                For both 'score' and 'flow':
                - hidden_features: int = 100,
                - num_heads: int = 4,
                - num_layers: int = 4,
                - mlp_ratio: int = 2,
                - time_embedding_dim: int = 32,
                - embedding_net: nn.Module = nn.Identity(),
                - dim_val: int = 64,
                - dim_id: int = 32,
                - dim_cond: int = 16,
                - ada_time: bool = False,
                - \**kwargs: Any,
            learning_rate: Learning rate for the optimizer.
            training_batch_size: Batch size for training.
            validation_fraction: Fraction of the training set to use for validation.
            stop_after_epochs: Number of epochs without improvement before stopping.
            clip_max_norm: Maximum norm for gradient clipping.
            training_args: Additional arguments to pass to the training function.
        """
        from sbi.inference import FlowMatchingSimformer, Simformer

        assert self.has_features, (
            "Feature array not created. Please create the feature array first."
        )

        if self.fitted_parameter_array is None:
            raise ValueError("Parameter grid not created. Please create the parameter grid first.")

        if name_append == "timestamp":
            name_append = self._timestamp

        out_dir = os.path.join(os.path.abspath(out_dir), self.name)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        run = False
        if (
            os.path.exists(f"{out_dir}/{self.name}_{name_append}_params.pkl")
            and load_existing_model
        ):
            logger.info(
                f"Loading existing model from {out_dir}/{self.name}_{name_append}_params.pkl"  # noqa: E501
            )
            posterior, stats, params = self.load_model_from_pkl(
                f"{out_dir}/{self.name}_{name_append}_posterior.pkl",
                set_self=set_self,
            )
            # return posterior, stats
            run = True

            if params is not None:
                save_model = False  # Don't save the model again if we loaded it.
        elif os.path.exists(f"{out_dir}/{self.name}_{name_append}_simformer.pkl"):
            logger.info(
                "Model with same name already exists. \
                Please change the name of this model or delete the existing one."
            )
            return None

        if (
            use_existing_indices
            and (self._train_indices is not None)
            and (self._test_indices is not None)
        ):
            train_indices = self._train_indices
            test_indices = self._test_indices
            logger.info("Using existing train and test indices.")

        if (train_indices is None) or (test_indices is None):
            train_indices, test_indices = self.split_dataset(
                train_fraction=train_test_fraction,
                random_seed=random_seed,
                verbose=verbose,
            )

        if set_self:
            self._train_indices = train_indices
            self._test_indices = test_indices

        X_train = self.feature_array[train_indices]
        y_train = self.fitted_parameter_array[train_indices]
        X_test = self.feature_array[test_indices]
        y_test = self.fitted_parameter_array[test_indices]

        # Construct simformer unflattened features

        X_train = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        X_test = torch.tensor(X_test, dtype=torch.float32, device=self.device)
        y_test = torch.tensor(y_test, dtype=torch.float32, device=self.device)

        inputs = torch.cat([y_train, X_train], dim=1)

        prior = self.create_priors(
            override_prior_ranges=override_prior_ranges,
            verbose=verbose,
        )

        if not run:
            simformer = Simformer if simformer_type == "score" else FlowMatchingSimformer

            from torch.utils.tensorboard import SummaryWriter

            summary_writer = (
                SummaryWriter(log_dir=f"{out_dir}/logs/{self.name}_{name_append}")
                if verbose
                else None
            )

            # model_kwargs - passed to model_builder.
            inference = simformer(
                device=self.device,
                sde_type=sde_type,
                logging_level="INFO",
                show_progress_bars=verbose,
                summary_writer=summary_writer if verbose else None,
                **model_kwargs,
            )

            training_args_default = {
                "training_batch_size": training_batch_size,
                "learning_rate": learning_rate,
                "validation_fraction": validation_fraction,
                "stop_after_epochs": stop_after_epochs,
                "max_num_epochs": max_num_epochs,
                "clip_max_norm": clip_max_norm,
                "force_first_round_loss": False,
                "discard_prior_samples": False,
                "retrain_from_scratch": False,
                "show_train_summary": True,
                "calibration_kernel": None,
                "ema_loss_decay": 0.1,
                "validation_times": 10,
                "dataloader_kwargs": None,
            }
            training_args_default.update(training_args)

            inference.append_simulations(inputs, data_device=self.device)

            start_time = time.time()
            inference.train(**training_args_default)
            end_time = time.time()
            # use torch to save score_estimator

            stats = [inference._summary]

            try:
                torch.save(inference, f"{out_dir}/{self.name}_{name_append}_simformer.pkl")
            except Exception as e:
                logger.error(
                    f"Error saving simformer "
                    f"to {out_dir}/{self.name}_{name_append}_simformer.pkl: {e}"
                )

            inference.set_condition_indexes(
                new_posterior_latent_idx=list(range(len(self.fitted_parameter_names))),
                new_posterior_observed_idx=list(
                    range(len(self.fitted_parameter_names), inputs.shape[1])
                ),
            )
            posterior = inference.build_posterior(prior=prior)

            """ This would be nice, but DirectPosterior is not compatible with Simformer yet
            it seems (no support for .parameters() method)
            posterior = DirectPosterior(posterior_estimator=posterior, prior=prior)
            posterior = EnsemblePosterior(
                posteriors=[posterior],
                weights=torch.tensor([1.0], device=self.device),
                theta_transform=posterior.theta_transform,
            )
            """
            # Save the posterior in a compatible format.
            try:
                with open(f"{out_dir}/{self.name}_{name_append}_posterior.pkl", "wb") as f:
                    if save_method == "dill":
                        import dill

                        dill.dump(posterior, f)
                    elif save_method == "joblib":
                        dump(posterior, f, compress=3)
                    elif save_method == "pickle":
                        pickle.dump(posterior, f)
                    else:
                        torch.save(posterior, f"{out_dir}/{self.name}_{name_append}_posterior.pkl")
            except Exception as e:
                logger.error(
                    f"Error saving posterior {out_dir}/{self.name}_{name_append}_posterior.pkl: {e}"
                )

            if set_self:
                self.simformer = inference
                self.posteriors = posterior
                self._prior = prior
                self.stats = stats
                self._X_train = X_train.cpu().numpy()
                self._y_train = y_train.cpu().numpy()
                self._X_test = X_test.cpu().numpy()
                self._y_test = y_test.cpu().numpy()

            if save_model:
                param_dict = {
                    "train_indices": train_indices,
                    "test_indices": test_indices,
                    "sde_type": sde_type,
                    "simformer_type": simformer_type,
                    "model_kwargs": model_kwargs,
                    "random_seed": random_seed,
                    "training_time": end_time - start_time,
                    "training_args": training_args_default,
                    "prior": prior,
                    "stats": stats,
                }
                self.save_state(
                    out_dir=out_dir,
                    name_append=name_append,
                    save_method=save_method,
                    has_grid=True,
                    **param_dict,
                )

        if plot:
            if verbose:
                logger.info("Plotting training diagnostics...")
            self.plot_diagnostics(
                X_train=X_train.cpu().numpy(),
                y_train=y_train.cpu().numpy(),
                X_test=X_test.cpu().numpy(),
                y_test=y_test.cpu().numpy(),
                plots_dir=f"{out_dir}/plots/{name_append}/",
                stats=None,
                sample_method="direct",
                posteriors=posterior,
            )
        # Evaluate the model
        if evaluate_model:
            metrics_path = f"{out_dir}/{self.name}_{name_append}_summary.json"
            if verbose:
                logger.info("Evaluating the model...")
            stats = self.evaluate_model(
                posteriors=posterior,
                X_test=X_test,
                y_test=y_test,
            )

            try:
                with open(metrics_path, "w") as f:
                    json.dump(stats, f, indent=4)
            except Exception as e:
                logger.error(f"Error saving metrics to {metrics_path}: {e}")

            if set_self:
                self.stats = stats

        return inference

        # Build conditional just lets you set whichever parameters are missing.
        # Build_posterior and build_likelihood are just wrappers around build_conditional.

        # conditional = inference.build_conditional(condition_mask=[False, True])
        # conditional_samples = conditional.sample((10000,), x=x_o)

        # Must set indexes here to indicate which are observed and which are latent.

        # Set condition indexes properly from len(self.fitted_param_names)
        # and len(self.feature_names)

        # inference.set_condition_indexes(new_posterior_latent_idx=[0],
        # new_posterior_observed_idx=[1])

        #
        # posterior_samples = posterior.sample((10000,), x=x_o)

        # likelihood = inference.build_likelihood()
        # likelihood_samples = likelihood.sample((10000,), x=theta_o)

    def run_single_sbi(
        self,
        train_test_fraction: float = 0.8,
        random_seed: Optional[int] = None,
        backend: str = "sbi",
        engine: Union[str, List[str]] = "NPE",
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
        n_nets: int = 1,
        model_type: Union[str, List[str]] = "mdn",
        hidden_features: Union[int, List[int]] = 50,
        num_components: Union[int, List[int]] = 4,
        num_transforms: Union[int, List[int]] = 4,
        training_batch_size: int = 64,
        learning_rate: float = 1e-4,
        validation_fraction: float = 0.2,
        stop_after_epochs: int = 15,
        clip_max_norm: float = 5.0,
        additional_model_args: dict = {},
        save_model: bool = True,
        verbose: bool = True,
        prior_method: str = "ili",
        out_dir: str = f"{code_path}/models/",
        plot: bool = True,
        name_append: str = "timestamp",
        feature_scalar: Callable = StandardScaler,
        target_scalar: Callable = StandardScaler,
        set_self: bool = True,
        learning_type: str = "offline",
        simulator: Optional[Callable] = None,
        num_simulations: int = 1000,
        num_online_rounds: int = 5,
        initial_training_from_library: bool = False,
        override_prior_ranges: dict = {},
        online_training_xobs: Optional[np.ndarray] = None,
        load_existing_model: bool = True,
        use_existing_indices: bool = True,
        evaluate_model: bool = True,
        save_method: str = "joblib",
        num_posterior_draws_per_sample: int = 1000,
        embedding_net: Optional[torch.nn.Module] = torch.nn.Identity(),
        custom_config_yaml: Optional[str] = None,
        sql_db_path: Optional[str] = None,
    ) -> tuple:
        """Run a single SBI training instance.

        Args:
            train_test_fraction (float, optional): Fraction of the dataset to be
                used for training. Defaults to 0.8.
            random_seed (int, optional): Random seed for reproducibility.
                Defaults to None.
            backend (str, optional): Backend to use for training ('sbi', 'lampe',
                or 'pydelfi'). Pydelfi cannot be installed in the same
                environment. Defaults to "sbi".
            engine (Union[str, List[str]], optional): Engine to use ('NPE',
                'NLE', 'NRE' or sequential variants). Defaults to "NPE".
            train_indices (np.ndarray, optional): Indices of the training set.
                Defaults to None.
            test_indices (np.ndarray, optional): Indices of the test set. If None,
                no test set is used. Defaults to None.
            n_nets (int, optional): Number of networks to use in the ensemble.
                Defaults to 1.
            model_type (Union[str, List[str]], optional): Type of model (e.g.,
                'mdn', 'maf', 'nsf'). If a list, an ensemble is used.
                Defaults to "mdn".
            hidden_features (Union[int, List[int]], optional): Number of hidden
                features in the neural network. Defaults to 50.
            num_components (Union[int, List[int]], optional): Number of components
                in the mixture density network. Defaults to 4.
            num_transforms (Union[int, List[int]], optional): Number of transforms
                in the masked autoregressive flow. Defaults to 4.
            training_batch_size (int, optional): Batch size for training.
                Defaults to 64.
            learning_rate (float, optional): Learning rate for the optimizer.
                Defaults to 1e-4.
            validation_fraction (float, optional): Fraction of training set for
                validation to prevent over-fitting. Training stops if
                validation loss doesn't improve for `stop_after_epochs`.
                Defaults to 0.2.
            stop_after_epochs (int, optional): Epochs without improvement
                before stopping. Defaults to 15.
            clip_max_norm (float, optional): Maximum norm for gradient clipping.
                Defaults to 5.0.
            additional_model_args (dict, optional): Additional arguments for
                the model (e.g., num_layers, use_batch_norm). Defaults to {}.
            save_model (bool, optional): Whether to save the trained model.
                Defaults to True.
            verbose (bool, optional): Whether to print verbose output.
                Defaults to True.
            prior_method (str, optional): Method to create the prior
                ('manual' or 'ili'). Defaults to "ili".
            feature_scalar (Callable, optional): Scaler class for the features.
                Defaults to StandardScaler.
            target_scalar (Callable, optional): Scaler class for the targets.
                Defaults to StandardScaler.
            out_dir (str, optional): Directory to save the model. Defaults to
                f"{code_path}/models/".
            plot (bool, optional): Whether to plot the diagnostics.
                Defaults to True.
            name_append (str, optional): String to append to the model name.
                Defaults to "timestamp".
            set_self (bool, optional): Whether to set instance attributes with
                the trained model. Defaults to True.
            learning_type (str, optional): Type of learning ('offline' or
                'online'). If 'online', a simulator must be provided.
                Defaults to "offline".
            simulator (Callable, optional): Function to simulate data for
                'online' learning. Defaults to None.
            num_simulations (int, optional): Number of simulations to run in
                each call during 'online' learning. Defaults to 1000.
            num_online_rounds (int, optional): Number of rounds for 'online'
                learning. Defaults to 5.
            initial_training_from_library (bool, optional): Whether to use the
                initial training from the library in 'online' learning.
                WARNING: This is broken. Defaults to False.
            override_prior_ranges (dict, optional): Dictionary of prior ranges
                to override the defaults. Defaults to {}.
            online_training_xobs (np.ndarray, optional): A single input
                observation to condition on for 'online' training.
                Defaults to None.
            load_existing_model (bool, optional): Whether to load an existing
                model if it exists. Defaults to True.
            use_existing_indices (bool, optional): Whether to use existing
                train/test indices if they exist. Defaults to True.
            evaluate_model (bool, optional): Whether to evaluate the model
                after training (computes log prob, PIT). Defaults to True.
            save_method (str, optional): Method to save the model ('torch',
                'pickle', 'joblib', or 'h5py'). Defaults to "joblib".
            num_posterior_draws_per_sample (int, optional): Number of posterior
                draws for metrics and plots. Defaults to 1000.
            embedding_net (Optional[torch.nn.Module], optional): Optional
                embedding network for the simulator.
                Defaults to torch.nn.Identity().
            custom_config_yaml (Optional[str], optional): Path to a custom YAML
                config file to override settings. Defaults to None.
            sql_db_path (Optional[str], optional): Path to an SQL database for
                logging or results. Defaults to None.

        Returns:
            tuple: A tuple containing the trained posterior distribution
            and a dictionary of training statistics.
        """
        assert learning_type in [
            "offline",
            "online",
        ], "Learning type should be either 'offline' or 'online'."
        out_dir = os.path.join(os.path.abspath(out_dir), self.name)

        if name_append == "timestamp":
            name_append = f"{self._timestamp}"

        run = False
        if os.path.exists(f"{out_dir}/{self.name}_{name_append}_params.pkl") and save_model:
            if load_existing_model:
                logger.info(
                    f"Loading existing model from {out_dir}/{self.name}_{name_append}_params.pkl"  # noqa: E501
                )
                posteriors, stats, params = self.load_model_from_pkl(
                    f"{out_dir}/{self.name}_{name_append}_posterior.pkl",
                    set_self=set_self,
                )
                # return posterior, stats
                run = True

                if params is not None:
                    save_model = False  # Don't save the model again if we loaded it.
            else:
                logger.info(
                    "Model with same name already exists. \
                    Please change the name of this model or delete the existing one."
                )
                return None

        if simulator is not None:
            self.has_simulator = True

        start_time = datetime.now()

        if custom_config_yaml is not None:
            # Doing this here so it's defined for custom models we are loading.
            import yaml

            logger.info(f"Loading custom config from {custom_config_yaml}")

            with open(custom_config_yaml) as f:
                train_args = yaml.safe_load(f)["train_args"]
            if sql_db_path is not None:
                if "://" in sql_db_path:
                    storage = create_database_universal(
                        f"{self.name}_{name_append}", full_url=sql_db_path
                    )
                else:
                    storage = f"{out_dir}/{self.name}_{name_append}_optuna_study_storage.log"
            else:
                storage = create_sqlite_db(
                    f"{out_dir}/{self.name}_{name_append}_optuna_study_storage.db"
                )  # noqa: E501
            if not train_args.get("skip_optimization", False):
                train_args["optuna"]["study"]["study_name"] = f"{self.name}_{name_append}"
                train_args["optuna"]["study"]["storage"] = storage
                net_configs = [
                    {"model": m} for m in train_args["optuna"]["search_space"]["model_choice"]
                ]

            else:
                net_configs = [{"model": train_args["fixed_params"]["model_choice"]}]

        if not run:
            if learning_type == "offline" or initial_training_from_library:
                if not self.has_features:
                    raise ValueError(
                        "Feature array not created. Please create the feature array first."
                    )

                if self.fitted_parameter_array is None:
                    raise ValueError(
                        "Parameter grid not created. Please create the parameter grid first."
                    )

                if self.fitted_parameter_names is None:
                    raise ValueError(
                        """Parameter names not created.
                        Please create the parameter names first."""
                    )

                if train_indices is None:
                    if (
                        not hasattr(self, "_train_indices")
                        or not hasattr(self, "_test_indices")
                        or self._train_indices is None
                        or self._test_indices is None
                        or (
                            len(self._train_indices) + len(self._test_indices)
                            != self.feature_array.shape[0]
                        )
                        or not use_existing_indices
                    ):
                        train_indices, test_indices = self.split_dataset(
                            train_fraction=train_test_fraction,
                            random_seed=random_seed,
                            verbose=verbose,
                        )
                    else:
                        logger.info("Using existing train and test indices.")
                        train_indices = self._train_indices
                        test_indices = self._test_indices
                # Prepare data

                assert len(train_indices) > 0, "Training indices should not be empty."
                assert len(test_indices) > 0, "Test indices should not be empty."

                assert (
                    self.feature_array.shape[0] == self.fitted_parameter_array.shape[0]
                ), f"""Feature array and parameter array should have the same number of
                    samples, got {self.feature_array.shape[0]} and
                    {self.fitted_parameter_array.shape[0]}."""

                X_train = self.feature_array[train_indices]
                y_train = self.fitted_parameter_array[train_indices]

                X_test = self.feature_array[test_indices]
                y_test = self.fitted_parameter_array[test_indices]

                if set_self:
                    self._X_train = X_train
                    self._y_train = y_train
                    self._train_indices = train_indices
                    self._test_indices = test_indices
                    self._train_fraction = train_test_fraction
                    self._X_test = X_test
                    self._y_test = y_test

                if prior_method == "manual":
                    # Scale features and targets
                    self._feature_scalar = feature_scalar()
                    self._target_scalar = target_scalar()
                    X_scaler = self._create_feature_scaler(X_train)
                    y_scaler = self._create_target_scaler(y_train)

                    X_scaled = X_scaler.transform(X_train)
                    y_scaled = y_scaler.transform(y_train)

                    # Setup prior based on scaled targets
                    y_std = np.std(y_scaled, axis=0)
                    y_min = np.min(y_scaled, axis=0)
                    y_max = np.max(y_scaled, axis=0)

                    prior_low = torch.tensor(
                        y_min - 3 * y_std, dtype=torch.float32, device=self.device
                    )
                    prior_high = torch.tensor(
                        y_max + 3 * y_std, dtype=torch.float32, device=self.device
                    )
                    prior = CustomIndependentUniform(
                        low=prior_low,
                        high=prior_high,
                        names_list=self.fitted_parameter_names,
                        device=self.device,
                    )
                elif prior_method == "ili":
                    # Create the prior using the parameter array
                    prior = self.create_priors(
                        verbose=verbose,
                        override_prior_ranges=override_prior_ranges,
                    )

                    X_scaled = X_train
                    y_scaled = y_train
                else:
                    raise ValueError("Invalid prior method. Use 'manual' or 'ili'.")

                # Create data loader
                loader = NumpyLoader(X_scaled, y_scaled)
            else:
                prior = self.create_priors(
                    verbose=verbose, override_prior_ranges=override_prior_ranges
                )

            if learning_type == "online":
                assert engine in [
                    "SNPE",
                    "SNLE",
                    "SNRE",
                ], "Engine should be either 'SNPE', 'SNLE' or 'SNRE'. for online learning."

                # Do online learning
                if simulator is None:
                    simulator = self.simulator

                assert callable(simulator), (
                    "Simulator function must be provided for online learning."
                )
                assert num_simulations > 0, "Number of simulations must be greater than 0."

                if not os.path.exists(f"{out_dir}/online/"):
                    os.makedirs(f"{out_dir}/online/")

                if initial_training_from_library:
                    # Save already created data to .npy files
                    np.save(
                        f"{out_dir}/online/xobs.npy",
                        self.feature_array[train_indices],
                    )
                    np.save(
                        f"{out_dir}/online/theta.npy",
                        self.fitted_parameter_array[train_indices],
                    )
                else:
                    if online_training_xobs is not None:
                        xobs = np.squeeze(online_training_xobs)
                        logger.info(f"Using provided xobs for online training: {xobs.shape}")
                        np.save(f"{out_dir}/online/xobs.npy", xobs)

                    else:
                        logger.warning(
                            """Drawing random photometry from prior to conditon on.
                            Results probably won't generalize well."""
                        )
                        samples = prior.sample((1,))
                        phot = []
                        for i in range(len(samples)):
                            p = simulator(samples[i]).cpu().numpy()
                            phot.append(p)

                        phot = np.array(phot)
                        phot = np.squeeze(phot)
                        # shape phot to be (num_simulations, num_features)
                        # phot = phot.reshape(num_simulations, -1)#[np.newaxis, :]

                        theta = samples.cpu().numpy()  # [np.newaxis, :]

                        np.save(f"{out_dir}/online/xobs.npy", phot)
                        np.save(f"{out_dir}/online/thetafid.npy", theta)

                # Need to have option for intial training using saved data.
                if isinstance(simulator, GalaxySimulator):
                    # Create a simulator function
                    print("Wrapping GalaxySimulator for SBI...")

                    def run_simulator(params, return_type="tensor"):
                        if isinstance(params, torch.Tensor):
                            params = params.cpu().numpy()
                        if isinstance(params, dict):
                            params = {i: params[i] for i in self.fitted_parameter_names}
                        elif isinstance(params, (list, tuple, np.ndarray)):
                            params = np.squeeze(np.array(params))
                            params = {
                                self.fitted_parameter_names[i]: params[i]
                                for i in range(len(self.fitted_parameter_names))
                            }
                        phot = simulator(params)
                        if return_type == "tensor":  #
                            x = torch.tensor(phot[np.newaxis, :], dtype=torch.float32).to(
                                self.device
                            )
                            return x
                        else:
                            return phot

                    use_sim = run_simulator
                else:
                    use_sim = simulator
                loader = SBISimulator(
                    in_dir=f"{out_dir}/online/",
                    xobs_file="xobs.npy",
                    thetafid_file="thetafid.npy" if online_training_xobs is None else None,
                    num_simulations=num_simulations,
                    save_simulated=True,
                    x_file="x.npy",  # if initial_training_from_library else None,
                    theta_file="theta.npy",  # , if initial_training_from_library else None,
                    simulator=use_sim,
                )

            nets = []
            ensemble_model_types = []
            ensemble_model_args = []

            if custom_config_yaml is None:
                for i in range(n_nets):
                    # Configure model
                    model_args = {}
                    model_type = model_type if isinstance(model_type, str) else model_type[i]
                    eng = engine if isinstance(engine, str) else engine[i]

                    if model_type == "mdn":
                        model_args = {
                            "hidden_features": hidden_features[i]
                            if isinstance(hidden_features, list)
                            else hidden_features,
                            "num_components": num_components[i]
                            if isinstance(num_components, list)
                            else num_components,
                        }
                    elif model_type in [
                        "maf",
                        "nsf",
                        "made",
                        "ncsf",
                        "cnf",
                        "gf",
                        "sospf",
                        "naf",
                        "unaf",
                    ]:
                        model_args = {
                            "hidden_features": hidden_features[i]
                            if isinstance(hidden_features, list)
                            else hidden_features,
                            "num_transforms": num_transforms[i]
                            if isinstance(num_transforms, list)
                            else num_transforms,
                        }
                        # if model_type == "nsf":
                        #    model_args["num_bins"] = num_bins

                    elif model_type in ["linear"]:
                        model_args = {}
                    elif model_type in ["mlp", "resnet"]:
                        model_args = {
                            "hidden_features": hidden_features[i]
                            if isinstance(hidden_features, list)
                            else hidden_features,
                        }
                    else:
                        raise ValueError(
                            f"""Unknown model type: {model_type}.
                            Options include: sbi = mdn, maf, nsf, made, linear, mlp, resnet.
                            lampe = mdn, maf, nsf, ncsf, cnf, nice, sospf, gf, naf.
                            pydelfi: mdn, maf."""
                        )

                    model_args.update(additional_model_args)

                    # Create neural network
                    net = self._create_network(
                        model_type,
                        model_args,
                        engine=eng,
                        backend=backend,
                        verbose=verbose,
                        embedding_net=embedding_net,
                        device=self.device,
                    )

                    nets.append(net)
                ensemble_model_types.append(model_type)
                ensemble_model_args.append(model_args)

                # Setup trainer arguments
                train_args = {
                    "training_batch_size": training_batch_size,
                    "learning_rate": learning_rate,
                    "validation_fraction": validation_fraction,
                    "stop_after_epochs": stop_after_epochs,
                    "clip_max_norm": clip_max_norm,
                }

                if learning_type == "online":
                    train_args["num_round"] = num_online_rounds

                trainer = InferenceRunner.load(
                    backend=backend,
                    engine=engine,
                    prior=prior,
                    nets=nets,
                    train_args=train_args,
                    out_dir=out_dir if save_model else None,
                    name=f"{self.name}_{name_append}_",
                    device=self.device,
                )
                extra_args = {}
            else:
                from .custom_runner import SBICustomRunner

                trainer = SBICustomRunner(
                    engine=engine,
                    prior=prior,
                    train_args=train_args,
                    out_dir=out_dir if save_model else None,
                    name=f"{self.name}_{name_append}_",
                    device=self.device,
                    net_configs=net_configs,
                    embedding_net=embedding_net,
                )
                if learning_type == "online":
                    validation_loader = loader
                else:
                    validation_loader = NumpyLoader(
                        self.feature_array[test_indices],
                        self.fitted_parameter_array[test_indices],
                    )
                extra_args = {"validation_loader": validation_loader}

            logger.info(f"Training on {self.device}.")
            # Train the model
            try:
                if not verbose:
                    # Suppress output if not verbose
                    buffer = StringIO()
                    with redirect_stdout(buffer):
                        posteriors, stats = trainer(loader, **extra_args)
                else:
                    pass
                    # Train with normal output
                    posteriors, stats = trainer(loader, **extra_args)

                    if posteriors is None:
                        logger.warning("Exiting training as posteriors are None.")
            except Exception as e:
                raise RuntimeError(f"Error during SBI training: {str(e)}")

            if set_self:
                self._prior = prior
                self._loader = loader
                self.posteriors = posteriors
                self.stats = stats
                self._train_args = train_args
                self._ensemble_model_types = ensemble_model_types
                self._ensemble_model_args = ensemble_model_args
                if learning_type == "online" or initial_training_from_library:
                    self.simulator = simulator
                    self._num_simulations = num_simulations
                    self._num_online_rounds = num_online_rounds
                    self._initial_training_from_library = initial_training_from_library
                else:
                    self._train_indices = train_indices
                    self._test_indices = test_indices
                    self._train_fraction = train_test_fraction

            end_time = datetime.now()

            elapsed_time = end_time - start_time
            logger.info(f"Time to train model(s): {elapsed_time}")

        else:
            posteriors = self.posteriors
            stats = self.stats
            X_test = self._X_test
            y_test = self._y_test
            X_scaled = self._X_train
            y_scaled = self._y_train

        # Save the params with the model if needed
        if save_model:
            if custom_config_yaml is None:
                param_dict = {
                    "engine": engine,
                    "learning_type": learning_type,
                    "ensemble_model_types": ensemble_model_types,
                    "ensemble_model_args": ensemble_model_args,
                    "n_nets": n_nets,
                    "train_args": train_args,
                    "stats": stats,
                    "training_time": elapsed_time.total_seconds(),
                }
            else:
                # Option to serialize custom config yaml
                param_dict = {
                    "engine": engine,
                    "train_args": train_args,
                    "stats": stats,
                    "custom_config_yaml": custom_config_yaml,
                    "net_configs": net_configs,
                    "training_time": elapsed_time.total_seconds(),
                }

            if learning_type == "online" or initial_training_from_library:
                # param_dict["simulator"] = simulator # don't serailize this.
                param_dict["num_simulations"] = num_simulations
                param_dict["num_online_rounds"] = num_online_rounds
                param_dict["initial_training_from_library"] = initial_training_from_library
                param_dict["online_training_xobs"] = online_training_xobs

            if learning_type == "offline":
                param_dict["train_fraction"] = train_test_fraction
                param_dict["test_indices"] = test_indices
                param_dict["train_indices"] = train_indices

            self.save_state(
                out_dir=out_dir,
                name_append=name_append,
                save_method=save_method,
                has_grid=learning_type == "offline" or initial_training_from_library,
                **param_dict,
            )

        if plot:
            if engine in ["NLE"]:
                sample_method = "emcee"
            else:
                sample_method = "direct"
            if learning_type == "offline":
                # Deal with the sampling method.
                self.plot_diagnostics(
                    X_train=X_scaled,
                    y_train=y_scaled,
                    X_test=X_test,
                    y_test=y_test,
                    plots_dir=f"{out_dir}/plots/{name_append}/",
                    stats=stats,
                    sample_method=sample_method,
                    posteriors=posteriors,
                    num_samples=num_posterior_draws_per_sample,
                )
            else:
                self.plot_diagnostics(
                    plots_dir=f"{out_dir}/online/plots/{name_append}/",
                    stats=stats,
                    sample_method=sample_method,
                    posteriors=posteriors,
                    online=True,
                    num_samples=num_posterior_draws_per_sample,
                )

        if evaluate_model:
            metrics_path = f"{out_dir}/{self.name}_{name_append}_metrics.json"

            samples_path = f"{out_dir}/plots/{name_append}/posterior_samples.npy"
            samples = samples_path if os.path.isfile(samples_path) else None
            if not os.path.exists(metrics_path):
                logger.info("Evaluating model...")
                metrics = self.evaluate_model(
                    posteriors=posteriors,
                    X_test=X_test,
                    y_test=y_test,
                    samples=samples,
                    num_samples=num_posterior_draws_per_sample,
                )

                try:
                    with open(metrics_path, "w") as f:
                        json.dump(metrics, f, indent=4)
                except Exception as e:
                    logger.error(f"Error saving metrics to {metrics_path}: {e}")

        return posteriors, stats

    def _create_feature_scaler(self, X: np.ndarray) -> object:
        """Create and fit a scaler for the features.

        Args:
            X: Feature array.

        Returns:
            A fitted scaler for the features.
        """
        scaler = self._feature_scalar
        scaler.fit(X)
        return scaler

    def _create_target_scaler(self, y: np.ndarray) -> object:
        """Create and fit a scaler for the targets.

        Args:
            y: Target array.

        Returns:
            A fitted scaler for the targets.
        """
        scaler = self._target_scalar
        scaler.fit(y)
        return scaler

    def _create_network(
        self,
        model_type: str,
        model_args: dict,
        engine: str = "NPE",
        backend: str = "sbi",
        verbose: bool = False,
        embedding_net: torch.nn.Module = torch.nn.Identity,
        device: str = "cpu",
    ) -> nn.Module:
        """Create a neural network for the SBI model.

        Parameters:
            model_type: Type of model to use. Either 'mdn' or 'maf'.
            model_args: Arguments for the model.
            engine: Engine to use for training. Either 'NPE', 'NLE', 'NRE' o
                or the sequential variants (SNPE, SNLE, SNRE).
            backend: Backend to use for training. Either 'sbi', 'lampe', or 'pydelfi'.
            verbose: Whether to print verbose output.
            device: Device to use for training. Either 'cpu' or 'cuda'.

        Returns:
            A neural network.
        """
        # Import the necessary function from ili

        backend_args = {}
        if backend == "sbi":
            net = ili.utils.load_nde_sbi
        elif backend == "lampe":
            net = ili.utils.load_nde_lampe
            backend_args["device"] = device
        elif backend == "pydelfi":
            net = ili.utils.load_nde_pydelfi
        else:
            raise ValueError("Invalid backend. Use 'sbi', 'lampe' or 'pydelfi'.")

        if verbose:
            # print summary of network
            logger.info(
                f"""Creating {model_type} network with {engine} engine """
                f"""and {backend} backend."""
            )
            for key, value in model_args.items():
                logger.info(f"     {key}: {value}")
        return net(
            engine=engine,
            model=model_type,
            embedding_net=embedding_net,
            **backend_args,
            **model_args,
        )

    def fit_observation_using_sampler(
        self,
        observation: np.ndarray,
        override_prior_ranges: dict = {},
        sampler: str = "dynesty",
        truths: np.ndarray = None,
        min_flux_error: float = 0.0,
        min_flux_pc_error: float = 0.0,
        interpolate_grid: bool = False,
        time_loglikelihood: bool = False,
        override_prior_distributions: dict = {},
        out_dir: str = f"{code_path}/models/name/nested_logs/",
        sampler_kwargs: dict = dict(
            nlive=500, bound="multi", sample="rwalk", update_interval=0.6, walks=25
        ),
        remove_params: list = None,
        plot_name="",
    ) -> None:
        """Fit the observation using the Dynesty sampler.

        Args:
            observation: The observed data to fit.
            override_prior_ranges: Dictionary of prior ranges to
                override the default ranges.
            sampler: The sampler to use. Currently 'dynesty', 'nautilus ' or 'ultranest'.
            truths: The true parameter values for the observation, if known.
            min_flux_error: Minimum flux error added in quadrature to the observation errors.
            min_flux_pc_error: Minimum flux percentage error added in quadrature
                to the observation errors.
                Either min_flux_error or min_flux_pc_error can be used, not both.
            interpolate_grid: Whether to recreate the simulator using interpolation.
            override_prior_distributions: Dictionary of prior distributions to
                override the default prior distributions. Needs to transform the
                unit cube to the parameter space. If multiple parameters are to be
                overridden by the same function, provide a list of parameter names as the key.
            sampler_kwargs: Additional keyword arguments to pass to the sampler.
            out_dir: directory for outputs.
            time_loglikelihood: whether to print execution times for log likelhood calls.
            remove_params: List of parameters to remove from the fitting.
            plot_name: Name to use for the output plots.

        Returns:
            The result of the fitting.

        """
        if not self.has_simulator and not interpolate_grid:
            self.recreate_simulator_from_library(set_self=True)

            if not self.has_simulator:
                raise ValueError("Simulator must be set to fit the observation.")

        if not hasattr(self, "_prior"):
            raise ValueError("Prior must be set to fit the observation.")

        assert len(observation.shape) == 1, "Observation must be a 1D array."
        assert len(observation) == len(self.feature_names), (
            "Observation must have the same length as the number of features: "
            f"{len(self.feature_names)}"
        )

        assert min_flux_error == 0.0 or min_flux_pc_error == 0.0, (
            "Only one of min_flux_error or min_flux_pc_error can be non-zero."
        )

        if out_dir is not None:
            out_dir = out_dir.replace("name", self.name)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

        # Split observation into photometry and errors if errors are included
        # treat everything as numpy array for dynesty
        observation = np.array(observation)
        phot_obs = []
        phot_err = []
        for i, name in enumerate(self.feature_names):
            if name in self.feature_array_flags.get("error_names", []):
                phot_err.append(observation[i])
            if name in self.feature_array_flags.get("raw_observation_names", []):
                phot_obs.append(observation[i])

        phot_obs = np.array(phot_obs)
        phot_err = np.array(phot_err) if len(phot_err) > 0 else None

        # assume phot_units are self.feature_array_flags["normed_flux_units"]
        # convert to nJy if needed

        # Check asinh errors
        if self.feature_array_flags.get("empirical_noise_models", False) and all(
            [
                isinstance(nm, AsinhEmpiricalUncertaintyModel)
                for nm in self.feature_array_flags["empirical_noise_models"].values()
            ]
        ):  # noqa: E501
            logger.info("Converting asinh photometry and errors to f_nu.")
            filters = self.feature_array_flags.get("raw_observation_names", [])
            nms = self.feature_array_flags["empirical_noise_models"]
            if phot_err is not None:
                phot_err = [
                    asinh_err_to_f_jy(phot_obs[i], phot_err[i], nms[filters[i]].b).to("nJy").value
                    for i in range(len(phot_err))
                ]
                phot_err = np.array(phot_err)
            phot_obs = [
                asinh_to_f_jy(phot_obs[i], nms[filters[i]].b).to("nJy").value
                for i in range(len(phot_obs))
            ]
            phot_obs = np.array(phot_obs)

        elif self.feature_array_flags.get("normed_flux_units", "AB") == "AB":
            # Convert AB to nJy
            phot_obs = 3631e9 * 10 ** (-0.4 * phot_obs)
            if phot_err is not None:
                phot_err = (phot_obs * phot_err * np.log(10)) / 2.5
        else:
            # use unyt to convert to nJy
            phot_obs = unyt_array(
                phot_obs, self.feature_array_flags.get("normed_flux_units", "nJy")
            )
            phot_obs = phot_obs.to("nJy").value
            if phot_err is not None:
                phot_err = unyt_array(
                    phot_err, self.feature_array_flags.get("normed_flux_units", "nJy")
                )
                phot_err = phot_err.to("nJy").value

        if phot_err is not None and min_flux_error > 0.0:
            phot_err = np.sqrt(phot_err**2 + min_flux_error**2)
        elif phot_err is not None and min_flux_pc_error > 0.0:
            assert min_flux_pc_error < 1.0, (
                "min_flux_pc_error should be a fraction, e.g. 0.05 for 5%."
            )
            phot_err = np.sqrt(phot_err**2 + (min_flux_pc_error * phot_obs) ** 2)

        # Convert the tensor prior to a dynesty prior function
        # Define a function to transform from the unit cube `u` to our prior `p`.
        prior = self._prior
        if prior is None:
            prior = self.create_priors(verbose=True, override_prior_ranges=override_prior_ranges)

        low = prior.base_dist.low.cpu().numpy()
        high = prior.base_dist.high.cpu().numpy()

        if interpolate_grid:
            from scipy.spatial import cKDTree

            # Just need photometry grid and parameter grid.
            phot = self.raw_observation_grid[
                [self.feature_names.index(name) for name in self.raw_observation_names]
            ]  # noqa: E501
            param = self.fitted_parameter_array

            def simulator_func(theta):
                """Interpolate the photometry grid to the parameter grid."""
                tree = cKDTree(param)
                dist, idx = tree.query(theta, k=1)
                return phot[:, idx]

            self.simulator = simulator_func

        else:
            self.simulator.output_type = ["photo_fnu"]
            self.simulator.out_flux_unit = "nJy"
            self.simulator.ignore_scatter = True

            filters_to_remove = [
                f for f in self.raw_observation_names if f not in self.feature_names
            ]
            self.simulator.update_photo_filters(
                photometry_to_remove=filters_to_remove, photometry_to_add=None
            )

            assert len(phot_obs) == len(self.simulator.instrument.filters), (
                f"Observation length {len(phot_obs)} does not match number of filters in simulator {len(self.simulator.instrument.filters)}."  # noqa: E501
            )

            assert all(
                self.simulator.instrument.filters.filter_codes
                == self.feature_array_flags.get("raw_observation_names", [])
            ), (  # noqa: E501
                "Simulator filters do not match feature array raw observation names."
            )

        pass_in_observables = {}
        for feature in self.feature_names:
            if feature in self.parameter_names:
                feature_index = list(self.feature_names).index(feature)
                pass_in_observables[feature] = observation[feature_index]
                print(f"Passing in {feature} = {observation[feature_index]} as observable.")

        # print('Phot:', phot_obs)
        # print('Phot Err:', phot_err)

        # print('SNR:', phot_obs/phot_err)
        scales = {"log10": lambda x: 10**x, "sqrt": lambda x: x**2}
        for parameter in self.fitted_parameter_names:
            for scale, func in scales.items():
                if parameter.startswith(f"{scale}_"):
                    logger.info(f"Auto applying inverse {scale} transform for {parameter}.")
                    self.simulator.param_transforms[parameter] = (
                        parameter.replace(f"{scale}_", ""),
                        func,
                    )

        # Add a convenience check.
        for emitter in self.simulator.emitter_params.values():
            for param in emitter:
                if "tau_v" in param:
                    if (
                        "tau_v" not in self.fitted_parameter_names
                        and "tau_v" not in pass_in_observables
                    ):
                        logger.info("Adding Av to tau_v transform.")
                        self.simulator.param_transforms["Av"] = ("tau_v", lambda x: x / 1.086)

        ndim = len(self.fitted_parameter_names)

        # Drop dimensions which are't constrained.
        test_params = {i: j for i, j in zip(self.fitted_parameter_names, 0.5 * (low + high))}
        test_params.update(pass_in_observables)
        try:
            self.simulator(test_params)
        except Exception as e:
            raise RuntimeError(f"Error testing simulator with mid prior parameters: {e}")

        if remove_params is None:
            remove_params = []

        idx_to_drop = np.ones(len(self.fitted_parameter_names), dtype=bool)
        for param in self.fitted_parameter_names:
            if param in remove_params or (
                param in self.simulator.unused_params
                and param not in self.parameter_names
                and param not in self.simulator.param_transforms.keys()
            ):  # noqa: E501
                logger.warning(
                    f"Fitted parameter {param} does not affect the photometry and will be removed from the fit."  # noqa: E501
                )
                index = list(self.fitted_parameter_names).index(param)
                idx_to_drop[index] = False
                ndim -= 1

        if not np.all(idx_to_drop):
            low = low[idx_to_drop]
            high = high[idx_to_drop]

            fitted_parameter_names = self.fitted_parameter_names[idx_to_drop]
        else:
            fitted_parameter_names = self.fitted_parameter_names

        print("Fitting parameters:", fitted_parameter_names)

        def sampling_prior(u):
            """Transform from the unit cube to the prior."""
            output = low + (high - low) * u
            for key, dist_func in override_prior_distributions.items():
                if isinstance(key, str):
                    if key in fitted_parameter_names:
                        index = list(fitted_parameter_names).index(key)
                        if isinstance(dist_func, tuple):
                            dist_func, extra_args = dist_func
                            output[index] = dist_func(u[index], **extra_args)
                        else:
                            output[index] = dist_func(u[index])
                elif isinstance(key, (list, tuple)):
                    indexes = []
                    temp = []
                    for param in key:
                        index = list(fitted_parameter_names).index(param)
                        indexes.append(index)
                        temp.append(output[index])
                    if isinstance(dist_func, tuple):
                        dist_func, extra_args = dist_func
                        transformed = dist_func(temp, **extra_args)
                    else:
                        transformed = dist_func(temp)
                    for i, index in enumerate(indexes):
                        output[index] = transformed[i]
                else:
                    raise ValueError("Keys in override_prior_distributions must be str or list.")

            return output

        def log_likelihood(theta):
            """Calculate the log likelihood of the observation given the parameters."""
            if time_loglikelihood:
                start_time = datetime.now()
            if isinstance(theta, np.ndarray):
                theta = {
                    i: theta[j] for j, i in enumerate(self.fitted_parameter_names[idx_to_drop])
                }
            theta.update(pass_in_observables)
            sim = self.simulator(theta)

            if phot_err is not None and len(phot_err) == len(phot_obs):
                chi2 = np.sum(((phot_obs - sim) / phot_err) ** 2)
            else:
                chi2 = np.sum((phot_obs - sim) ** 2)

            loglike = -0.5 * chi2

            if not np.isfinite(loglike):
                raise ValueError("Non-finite log likelihood for theta: ", theta)

            if time_loglikelihood:
                end_time = datetime.now()
                elapsed_time = end_time - start_time
                logger.info(f"Time to compute log likelihood: {elapsed_time}")

            return loglike

        """
        import dill
        import hickle
        dynesty.utils.pickle_module = hickle

        with dynesty.pool.Pool(n_proc, loglike=log_likelihood,
                                prior_transform=dynesty_prior) as pool:
        """

        if sampler.lower() == "dynesty":
            import dynesty

            logger.info("Using Dynesty sampler.")
            run_kwargs = sampler_kwargs.pop("run_kwargs", {})
            sampler = dynesty.DynamicNestedSampler(
                log_likelihood, sampling_prior, ndim, **sampler_kwargs
            )
            sampler.run_nested(**run_kwargs)

            result = sampler.results

            logger.info("Dynesty fitting complete.")

            logger.info(result.summary())

            # use dynesty plotting functions to plot the results
            try:
                import dynesty.plotting as dyplot

                fig, axes = dyplot.cornerplot(
                    results=result,
                    truths=truths,
                    show_titles=True,
                    title_fmt=".2f",
                    quantiles=[0.16, 0.5, 0.84],
                    labels=self.fitted_parameter_names[idx_to_drop],
                )

                if out_dir is not None:
                    fig.savefig(f"{out_dir}/{name}_dynesty_corner.png", dpi=300)
                # print table of the results
                from dynesty import utils as dyfunc

                samples, weights = result.samples, result.importance_weights()
                mean, cov = dyfunc.mean_and_cov(samples, weights)
                logger.info("Parameter means and 1-sigma uncertainties:")
                for i, name in enumerate(self.parameter_names):
                    std = np.sqrt(cov[i, i])
                    logger.info(f"{name}: {mean[i]:.3f} ± {std:.3f}")
            except Exception as e:
                logger.error(f"Issue importing dynesty plotting module: {e}")
            return result
        elif sampler.lower() == "ultranest":
            try:
                import ultranest
            except ImportError:
                raise ImportError(
                    "Ultranest is not installed. Please install it to use this sampler."
                )
            logger.info("Using Ultranest sampler.")
            sampler = ultranest.ReactiveNestedSampler(
                self.fitted_parameter_names[idx_to_drop].tolist(),
                log_likelihood,
                transform=sampling_prior,
                log_dir=out_dir,
                resume=True,
            )
            sampler.run(**sampler_kwargs)
            logger.info("Ultranest fitting complete.")

            sampler.print_results()

            sampler.plot()
            sampler.plot_trace()

            return result
        elif sampler.lower() == "nautilus":
            try:
                import nautilus
            except ImportError:
                raise ImportError(
                    "Nautilus is not installed. Please install it to use this sampler."
                )
            logger.info("Using Nautilus sampler.")

            prior = nautilus.Prior()

            for lo, hi, name in zip(low, high, self.fitted_parameter_names[idx_to_drop]):
                prior.add_parameter(name, dist=(lo, hi))

            print(prior)

            sampler = nautilus.Sampler(prior, log_likelihood, **sampler_kwargs)
            sampler.run(verbose=True)

            points, log_w, log_l = sampler.posterior()
            logger.info("Nautilus fitting complete.")
            import corner

            fig = corner.corner(
                points,
                weights=np.exp(log_w),
                bins=20,
                labels=prior.keys,
                color="purple",
                plot_datapoints=False,
                range=np.repeat(0.999, len(prior.keys)),
                show_titles=True,
                title_fmt=".2f",
                quantiles=[0.16, 0.5, 0.84],
                title_kwargs={"fontsize": 12},
            )

            if out_dir is not None:
                fig.savefig(f"{out_dir}/nautilus_corner.png", dpi=300)

            return sampler
        elif sampler.lower() == "blackjax":
            raise NotImplementedError("Blackjax sampler not yet implemented.")

        else:
            raise ValueError("Sampler must be either 'dynesty', 'ultranest' or 'nautilus'.")

    def recreate_simulator_from_library(
        self,
        set_self=True,
        overwrite=False,
        override_library_path=None,
        override_grid_path=None,
        **kwargs,
    ):
        """Recreate the simulator from the HDF5 library.

        Simple libraries (single SFH, ZDist, one basis)
        with pre-existing simple EmissionModels
        can be recreated as a simulator, allowing
        for SED recovery without manual
        specification of the simulator function.

        Two possible use cases, either to generate
        samples for fitting (in which case we want
        the config to closely match the feature
        array) or to recover the SED, in
        which case we aren't concerned about
        depths, noise models, etc.

        Parameters:
            set_self: Whether to set the simulator attribute.
            overwrite: Whether to overwrite an existing simulator.
            override_library_path: Path to the library to use.
            override_grid_path: Path to the synthesizer grid directory to use.
            **kwargs: Additional keyword arguments to pass to the GalaxySimulator.

        Returns:
            The recreated GalaxySimulator object.
        """
        # grid path
        if override_library_path is not None:
            library_path = override_library_path
        else:
            library_path = self.library_path

        if self.has_simulator and not overwrite:
            return self.simulator

        default_kwargs = {
            "normalize_method": None,
            "include_phot_errors": False,
            "depths": None,
            "depth_sigma": None,
            "noise_models": None,
            "out_flux_unit": "AB",
        }

        if self.feature_array_flags != {}:
            flags = self.feature_array_flags

            flag_params = {
                "normalize_method": flags.get("normalize_method", None),
                # Make default behaviour not rescattering photometry.
                "include_phot_errors": False,  # flags["include_errors_in_feature_array"],
                "depths": flags.get("depths", None),
                "depth_sigma": 5,
                "noise_models": flags.get("empirical_noise_models", None),
                "photometry_to_remove": flags.get("photometry_to_remove", None),
                "out_flux_unit": flags.get("normed_flux_units", None),
            }

            default_kwargs.update(flag_params)
        else:
            flags = {}

        default_kwargs.update(kwargs)

        try:
            simulator = GalaxySimulator.from_library(
                library_path, override_synthesizer_grid_dir=override_grid_path, **default_kwargs
            )
        except ValueError as e:
            logger.error(
                "Could not recreate simulator from grid. This model"
                " may not be compatible. A GalaxySimulator object can"
                " be provided manually to recover the SED."
            )
            logger.error(f"Error message: {e}")
            return None

        removed_params = flags.get("parameters_to_remove", [])
        if removed_params:
            for param in removed_params:
                param_index = self.parameter_names.index(param)
                data = self.parameter_array[:, param_index]
                # check if all values are the same
                if np.all(data == data[0]):
                    simulator.fixed_params[param] = data[0]

        if set_self:
            self.simulator = simulator
            self.has_simulator = True
            logger.info(f"Simulator recreated from library at {library_path}.")

        scales = {"log10": lambda x: 10**x, "sqrt": lambda x: x**2}
        for parameter in self.fitted_parameter_names:
            for scale, func in scales.items():
                if parameter.startswith(f"{scale}_"):
                    logger.info(f"Auto applying inverse {scale} transform for {parameter}.")
                    self.simulator.param_transforms[parameter] = (
                        parameter.replace(f"{scale}_", ""),
                        func,
                    )

        # Add a convenience check.
        for emitter in self.simulator.emitter_params.values():
            for param in emitter:
                if "tau_v" in param:
                    if (
                        "tau_v" not in self.fitted_parameter_names
                        # and "tau_v" not in pass_in_observables
                    ):
                        logger.info("Adding Av to tau_v transform.")
                        self.simulator.param_transforms["Av"] = ("tau_v", lambda x: x / 1.086)

        return simulator

    def recover_SED(
        self,
        X_test: np.ndarray,
        samples: np.ndarray = None,
        num_samples=1000,
        sample_method: str = "direct",
        sample_kwargs: dict = {},
        posteriors=None,
        simulator: callable = None,
        prior: object = None,
        plot: bool = True,
        marginalized_parameters: Dict[str, callable] = None,
        extra_parameters: Dict[str, float] = None,
        phot_unit="AB",
        true_parameters=[],
        plot_name=None,
        plots_dir=f"{code_path}/models/name/plots/",
        sample_color="firebrick",
        param_labels=None,
        plot_closest_draw_to={},
        plot_sfh=True,
        plot_histograms=True,
        kde=True,
        save_plots=True,
        fig=None,
        ax=None,
        ax_sfh=None,
    ):
        """Recover the SED for a given observation, if a simulator is provided.

        Parameters:
            X_test: The input observation to recover the SED for.
            samples: Samples from the posterior distribution.
                if None, samples will be drawn from the posterior.
            num_samples: Number of samples to draw from the posterior.
            sample_method: Method to sample from the posterior if samples is None.
                Either 'direct' or 'emcee'.
            sample_kwargs: Additional keyword arguments for sampling.
            posteriors: The posterior distribution to use. If None, uses self.posteriors.
            simulator: A callable simulator function to generate the SED.
            prior: The prior distribution to use. If None, uses self._prior.
            plot: Whether to plot the results.
            marginalized_parameters: Dictionary of parameters to marginalize over.
            phot_unit: Photometric unit for the SED plot (default is 'AB').
            true_parameters: List of true parameters to plot on the SED.
            plot_name: Name for the plot file.
            plots_dir: Directory to save the plots.
            sample_color: Color for the sampled SED lines.
            param_labels: Labels for the parameters in the plot.
            plot_closest_draw_to: Dictionary of parameters and values
                to plot closest draws. E.g. if you want to plot the closest draw
                for a specific redshift, or mass, to see degeneracies in the posteriors.
            plot_sfh: Whether to plot the SFH in addition to the SED.
            plot_histograms: Whether to plot histograms of the parameters.
            kde: Whether to use KDE for the histograms.
            save_plots: Whether to save the plots.
            fig: Matplotlib figure to use for the SED plot.
            ax: Matplotlib axis to use for the SED plot.
            ax_sfh: Matplotlib axis to use for the SFH plot.

        Returns:
            fnu_quantiles: The quantiles of the recovered SED.

        """
        import astropy.units as u

        if posteriors is None:
            posteriors = self.posteriors

        if marginalized_parameters is None:
            marginalized_parameters = {}

        if extra_parameters is None:
            extra_parameters = {}

        if simulator is None:
            if not self.has_simulator:
                self.recreate_simulator_from_library(set_self=True)

            if not hasattr(self, "simulator"):
                raise ValueError("Simulator must be provided or set in the object.")
            simulator = self.simulator

        if prior is None:
            if not hasattr(self, "_prior"):
                raise ValueError("Prior must be provided or set in the object.")
            prior = self._prior

        plots_dir = plots_dir.replace("name", self.name)

        # Draw samples, run through simulator,
        if samples is None:
            samples = self.sample_posterior(
                X_test=[X_test],
                sample_method=sample_method,
                num_samples=num_samples,
                posteriors=posteriors,
                sample_kwargs=sample_kwargs,
            )

        # Run through simulator
        # Check if simulator is callable
        if not callable(simulator):
            raise ValueError("Simulator must be a callable function.")

        fnu_draws = []
        phot_fnu_draws = []
        wav_draws = []
        sfh = []
        counter = 0
        if isinstance(simulator, GalaxySimulator) or True:  # This is broken somehow??
            simulator.output_type = ["photo_fnu", "fnu", "sfh"]
            simulator.out_flux_unit = "nJy"
            for i in trange(num_samples, desc="Running simulator on samples"):
                if isinstance(samples, dict):
                    params = {key: samples[key][i] for key in samples.keys()}
                else:
                    params = {
                        self.simple_fitted_parameter_names[j]: samples[i, j]
                        for j in range(len(self.fitted_parameter_names))
                    }
                for parameter in marginalized_parameters.keys():
                    params[parameter] = marginalized_parameters[parameter](params)

                pass_in_observables = {}
                for feature in self.feature_names:
                    if feature in self.parameter_names:
                        feature_index = list(self.feature_names).index(feature)
                        pass_in_observables[feature] = X_test[feature_index]

                params.update(extra_parameters)
                params.update(pass_in_observables)
                output = simulator(params)
                if i == 0:
                    phot_wav = output["photo_wav"]
                    filters = output["filters"]
                wav_draws.append(output["fnu_wav"])
                fnu_draws.append(output["fnu"])
                if np.sum(np.isnan(output["fnu"])) > 0:
                    logger.warning(f"Warning! NaN values found in fnu draw {i}.", output["fnu"])
                phot_fnu_draws.append(output["photo_fnu"])
                sfh.append(output["sfh"])
                if np.isnan(output["sfh"]).any():
                    counter += 1
        else:
            raise ValueError("Simulator must be a GalaxySimulator object.")

        fnu_draws = np.array(fnu_draws)
        phot_fnu_draws = np.array(phot_fnu_draws)
        wav_draws = np.array(wav_draws)
        sfh = np.array(sfh)
        if counter > 0:
            logger.warning(f"Number of NaN SFH: {counter}")

        extra_indexes = []
        extra_labels = []
        extra_lines = []
        if len(plot_closest_draw_to) > 0:
            for key, value in plot_closest_draw_to.items():
                if key in self.fitted_parameter_names or key in self.simple_fitted_parameter_names:
                    if key in self.simple_fitted_parameter_names:
                        index = list(self.simple_fitted_parameter_names).index(key)
                    else:
                        index = list(self.fitted_parameter_names).index(key)

                    closest_index = np.argmin(np.abs(samples[:, index] - value))
                    logger.info(
                        f"""Closest draw to {key}={value} is
                        {samples[closest_index, index]} at index {closest_index}"""
                    )
                    logger.info(f"Full draw: {samples[closest_index]}")
                    extra_indexes.append(closest_index)
                    extra_labels.append(f"{key}={value}")

        fnu_quantiles = np.nanquantile(fnu_draws, [0.16, 0.5, 0.84], axis=0)
        phot_fnu_quantiles = np.nanquantile(phot_fnu_draws, [0.16, 0.5, 0.84], axis=0)
        sfh_quantiles = np.nanquantile(sfh, [0.16, 0.5, 0.84], axis=0)
        wav = np.nanmean(wav_draws, axis=0)

        # Convert fnu_quantiles
        if phot_unit == "AB":
            # Convert to AB mag
            fnu_quantiles = -2.5 * np.log10(fnu_quantiles) + 31.4
            phot_fnu_quantiles = -2.5 * np.log10(phot_fnu_quantiles) + 31.4
            fnu_draws = -2.5 * np.log10(fnu_draws) + 31.4
        else:
            if not isinstance(fnu_quantiles, unyt_quantity):
                fnu_quantiles = unyt_array(fnu_quantiles, self.simulator.out_flux_unit).to_astropy()
            fnu_quantiles = fnu_quantiles.to_value(
                phot_unit, equivalencies=u.spectral_density(wav * u.AA)
            )
            if not isinstance(phot_fnu_quantiles, unyt_quantity):
                phot_fnu_quantiles = unyt_array(
                    phot_fnu_quantiles, self.simulator.out_flux_unit
                ).to_astropy()
            phot_fnu_quantiles = phot_fnu_quantiles.to_value(
                phot_unit, equivalencies=u.spectral_density(phot_wav * u.AA)
            )
            if not isinstance(fnu_draws, unyt_quantity):
                fnu_draws = unyt_array(fnu_draws, self.simulator.out_flux_unit).to_astropy()
            fnu_draws = fnu_draws.to_value(phot_unit, equivalencies=u.spectral_density(wav * u.AA))

        if plot:
            if fig is None and ax is None:
                fig = plt.Figure(figsize=(8, 6), dpi=200, constrained_layout=True)
            else:
                fig = None
            if plot_histograms:
                ngrid = len(self.fitted_parameter_names) // 6 + 1
            else:
                ngrid = 0
            if fig is not None:
                gridspec = fig.add_gridspec(1 + ngrid, 6, height_ratios=[1] + [0.4] * ngrid)
            if ax is None:
                ax = fig.add_subplot(gridspec[0, :])

            if plot_sfh and (fig is not None or ax_sfh is not None):
                # inset axes for SFH inside ax
                if ax_sfh is None:
                    inset_ax = fig.add_axes([0.78, 0.65, 0.18, 0.18])
                else:
                    inset_ax = ax_sfh
                # plot the SFH
                inset_ax.plot(
                    output["sfh_time"],
                    sfh_quantiles[1],
                    label="Median SFH",
                    color=sample_color,
                )
                inset_ax.fill_between(
                    output["sfh_time"],
                    sfh_quantiles[0],
                    sfh_quantiles[2],
                    alpha=0.5,
                    color=sample_color,
                )
                # Don't let the time go beyond 0
                new_lim = optimize_sfh_xlimit(inset_ax)
                inset_ax.set_xlim(0, new_lim)
                inset_ax.set_xlabel("Lookback Time (Myr)", fontsize=10)
                inset_ax.set_ylabel(r"M$_{\odot} \rm \ yr^{-1}$", fontsize=10)
                # Reduced xlabel size
                inset_ax.tick_params(axis="both", which="major", labelsize=8)

            ax.plot(
                wav,
                fnu_quantiles[1],
                label="Median SED",
                color=sample_color,
                zorder=7,
                lw=2,
            )
            ax.fill_between(
                wav,
                fnu_quantiles[0],
                fnu_quantiles[2],
                alpha=0.5,
                color=sample_color,
                zorder=7,
            )

            for f, lam in zip(fnu_draws, wav_draws):
                ax.plot(lam, f, color=sample_color, alpha=0.01, lw=0.14, zorder=5)

            if len(extra_indexes) > 0:
                for i, index in enumerate(extra_indexes):
                    if np.sum(fnu_draws[index]) == 0:
                        logger.warning(f"The draw at index {index} has all zeros. Skipping.")
                    line = ax.plot(
                        wav,
                        fnu_draws[index],
                        label=extra_labels[i],
                        lw=1,
                        zorder=10,
                    )
                    extra_lines.append(line)

            if fig is not None:
                ax.set_xlabel("Wavelength (Angstrom)")
                phottxt = phot_unit.to_latex() if isinstance(phot_unit, u.UnitBase) else phot_unit
                ax.set_ylabel(f"Flux Density ({phottxt})")

            # Try and match filters names to feature names
            # and plot the photometry we have been given
            # filter_codes = [i.split("/")[-1] for i in filters.filter_codes]
            if true_parameters is not None and len(true_parameters) > 0:
                # make dict
                true_parameters_dict = {
                    self.simple_fitted_parameter_names[j]: true_parameters[j]
                    for j in range(len(self.fitted_parameter_names))
                }
                for parameter in marginalized_parameters.keys():
                    true_parameters_dict[parameter] = marginalized_parameters[parameter](
                        true_parameters_dict
                    )

                true_parameters_dict.update(extra_parameters)
                true_parameters_dict.update(pass_in_observables)

                true_sed_output = simulator(true_parameters_dict)
                true_sed = true_sed_output["fnu"]
                if phot_unit == "AB":
                    true_sed = -2.5 * np.log10(true_sed) + 31.4
                else:
                    from astropy import units as u

                    if not isinstance(true_sed, u.Quantity):
                        true_sed = true_sed * u.nJy
                    if isinstance(true_sed_output["fnu_wav"], u.Quantity):
                        wav = wav.to(u.AA).value
                    wav *= u.AA
                    true_sed = true_sed.to(phot_unit, equivalencies=u.spectral_density(wav)).value
                ax.plot(
                    wav,
                    true_sed,
                    label="True SED",
                    color="#191970",
                    lw=1,
                    zorder=11,
                )
                if plot_sfh and (fig is not None or ax_sfh is not None):
                    true_sfh = true_sed_output["sfh"]
                    true_sfh_time = true_sed_output["sfh_time"]
                    inset_ax.plot(true_sfh_time, true_sfh, label="True SFH", color="#191970")

            # Match indexes of the filters to the feature names
            # Now plot the photometry we have been given (X_test).
            # TODO; Deal with the fact that this could be normalized
            labelled = False
            phots = []

            for pos, filter in enumerate(filters.filter_codes):
                index = self.feature_names.index(filter)
                phot = X_test[index]
                phots.append(phot)

            phots = np.array(phots)
            median_phot = np.nanmedian(phots)
            max_phot_diff = 4
            show_lims = phots > (median_phot + max_phot_diff)
            phots[show_lims] = median_phot + max_phot_diff

            min_x, max_x = filters.get_non_zero_lam_lims()

            phots = np.array(phots)
            max_phot = 1.03 * np.nanmax(phots[phots < 35])

            ax.set_xlim(min_x, max_x)
            fnu_lam_mask = (wav > min_x) & (wav < max_x)
            max_f = min(1.03 * np.nanmax(fnu_quantiles[2][fnu_lam_mask]), max_phot)
            min_f = 0.97 * np.nanmin(fnu_quantiles[0][fnu_lam_mask])

            for pos, filter in enumerate(filters.filter_codes):
                if filter in self.feature_names:
                    if f"unc_{filter}" in self.feature_names:
                        # Get the index of the filter
                        index = self.feature_names.index(f"unc_{filter}")
                        phot_unc = X_test[index]
                    else:
                        phot_unc = 0
                    index = self.feature_names.index(filter)
                    phot = X_test[index]
                    # Phot needs a unit conversion if not AB
                    unit = self.feature_array_flags.get("normed_flux_units", "AB")

                    noise_models = self.feature_array_flags.get("empirical_noise_models", None)
                    if noise_models is not None:
                        if isinstance(noise_models[filter], AsinhEmpiricalUncertaintyModel):
                            unit = "asinh"
                    if phot_unit != unit:
                        from astropy import units as u
                        from astropy.units import spectral_density

                        if unit == "asinh":
                            phot_unc = asinh_err_to_f_jy(phot, phot_unc, noise_models[filter].b)
                            phot = asinh_to_f_jy(phot, noise_models[filter].b)
                            unit = u.Jy

                        if unit == "AB":
                            unit = u.ABmag
                        phot_with_unit = phot * u.Unit(unit)

                        if phot_unit == "AB":
                            pu = u.ABmag
                        else:
                            pu = u.Unit(phot_unit)

                        phot = (
                            phot_with_unit.to(
                                pu, equivalencies=spectral_density(phot_wav[pos] * u.AA)
                            )
                        ).value
                        if unit == "AB":
                            # Convert uncertainty from mag to flux
                            if phot_unc > 0:
                                phot_unc = (phot * phot_unc * np.log(10)) / 2.5

                        phot_unc = (phot_unc * u.Unit(unit)).to(
                            phot_unit, equivalencies=spectral_density(phot_wav[pos] * u.AA)
                        )  # noqa: E501
                        phot_unc = phot_unc.value

                    if phot_unit == "AB":
                        snr = 2.5 / (phot_unc * np.log(10) / phot)
                    else:
                        snr = phot / phot_unc

                    if show_lims[pos] or snr < 3:
                        # phot = median_phot + max_phot_diff
                        # Plot a downward arrow patch
                        if show_lims[pos]:
                            snr_limit = median_phot + max_phot_diff

                        else:
                            snr_limit = 3 * phot_unc

                        """scale = 0 #
                        # 0.07 * (max_f - min_f)

                        ax.add_patch(
                            FancyArrowPatch(
                                (phot_wav[pos], phot),
                                (phot_wav[pos], phot-scale),
                                arrowstyle="->",
                                mutation_scale=10,
                                color="black",
                                lw=1,
                                zorder=12,
                            )"""

                        ax.scatter(
                            phot_wav[pos],
                            snr_limit,
                            label="",
                            marker="v",
                            color="#191970",
                            zorder=15,
                            s=5,
                        )
                    else:
                        labelled = True
                        if phot_unc == 0:
                            ax.scatter(
                                phot_wav[pos],
                                phot,
                                label="Input Phot." if not labelled else "",
                                marker="o",
                                color="#191970",
                                edgecolor="black",
                                markeredgewidth=0.5,
                                zorder=25,
                                s=10,
                            )
                        else:
                            ax.errorbar(
                                phot_wav[pos],
                                phot,
                                yerr=phot_unc,
                                label="Input Phot." if not labelled else "",
                                marker="o",
                                color="none",
                                markeredgecolor="#191970",
                                markeredgewidth=0.5,
                                ecolor="#191970",
                                elinewidth=1,
                                zorder=25,
                                markersize=5,
                                linestyle="None",
                            )

                # Plot the photometry we have drawn from the posterior

                phot = phot_fnu_quantiles[1][pos]

                phot_lower = phot_fnu_quantiles[0][pos] - phot
                phot_upper = phot - phot_fnu_quantiles[2][pos]

                if phot_unit == "AB":
                    err = [[phot_upper], [phot_lower]]
                else:
                    err = [[phot_lower], [phot_upper]]
                err = np.abs(err)
                ax.errorbar(
                    phot_wav[pos],
                    phot,
                    yerr=err,
                    label="",  # "Posterior Phot." if pos == 0 else "",
                    marker="s",
                    color=sample_color,
                    zorder=14,
                    markersize=5,
                    linestyle="None",
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                )

            # Set min and max of the x axis based on observed photomery

            ax.set_ylim(min_f, max_f)
            ax.legend(fontsize=8, loc="upper left")

            # If AB mag, flip y axis
            if phot_unit == "AB":
                ax.invert_yaxis()
                ax.set_ylabel("Flux Density (AB mag)")

            if param_labels is None:
                param_labels = self.simple_fitted_parameter_names
            if plot_histograms and fig is not None:
                # Add a row of axis underneath and plot histograms of parameters
                for i, param in enumerate(param_labels):
                    if i > 5:
                        jax = (1 + i // 6) % 5
                        iax = i % 6
                    else:
                        jax = 1
                        iax = i
                    ax = fig.add_subplot(gridspec[jax, iax])
                    if not kde:
                        ax.hist(
                            samples[:, i],
                            bins=50,
                            density=False,
                            alpha=0.9,
                            color=sample_color,
                        )
                    else:
                        try:
                            from scipy.stats import gaussian_kde

                            kde_skl = gaussian_kde(samples[:, i])
                            x_min = np.percentile(samples[:, i], 0.1)
                            x_max = np.percentile(samples[:, i], 99.9)
                            x_range = np.linspace(x_min, x_max, 100)
                            ax.plot(
                                x_range,
                                kde_skl(x_range) * num_samples * (x_max - x_min) / 50,
                                color=sample_color,
                            )
                            ax.fill_between(
                                x_range,
                                0,
                                kde_skl(x_range) * num_samples * (x_max - x_min) / 50,
                                alpha=0.5,
                                color=sample_color,
                            )
                        except Exception as e:
                            logger.warning(f"Could not make KDE plot for {param}: {e}")

                    unit = self.fitted_parameter_units[
                        list(self.fitted_parameter_names).index(param)
                    ]
                    if param in self.fitted_parameter_names:
                        param = (
                            param.replace("_", " ")
                            .replace("log10 ", "")
                            .replace("log10", "")
                            .replace("log", "")
                            .replace("floor", "")
                            .title()
                        )
                    param_labels_unit = (
                        f"{param} ({unit})" if str(unit) != "dimensionless" else param
                    )

                    param_labels_unit = (
                        param_labels_unit.replace("Myr", "Myr")
                        .replace("_Msun", r"M$_\odot$")
                        .replace("Msun", r"M$_\odot$")
                        .replace("Zsun", r"Z$_\odot$")
                        .replace("Lsun", r"L$_\odot$")
                        .replace("Angstrom", r"\AA")
                    )  # noqa: E501
                    param_labels_unit = (
                        param_labels_unit.replace("log10_", r"$\log_{10}$")
                        .replace("log10 ", r"$\log_{10}$")
                        .replace("log10", r"$\log_{10}$")
                        .replace("log ", r"$\log~$")
                    )
                    param_labels_unit = (
                        param_labels_unit.replace("Sfr 10", "SFR$_{10}$")
                        .replace("Sfr", "SFR")
                        .replace("Zgas", "Z$_{gas}$")
                        .replace("Zstar", "Z$_{star}$")
                        .replace("Tauv", r"$\tau_V$")
                    )
                    param_labels_unit = (
                        param_labels_unit.replace("Av", r"$A_V$")
                        .replace("Muv", r"$M_{UV}$")
                        .replace("Ha", r"H$\alpha$")
                        .replace("Hb", r"H$\beta$")
                        .replace("Metallicity", r"Z$_\star$")
                        .replace("Zstar", r"Z$_\star$")
                    )  # noqa: E501
                    param_labels_unit = (
                        param_labels_unit.replace("Sfh", "SFH")
                        .replace("Mass Weighted Age", "Age$_{MW}$")
                        .replace("floor", "")
                        .replace("Surviving Mass", "M$_{\\star, \\rm surv}$")
                    )  # noqa: E501
                    ax.set_xlabel(param_labels_unit, fontsize=10)
                    ax.set_yticks([])
                    if len(true_parameters) > 0:
                        ax.axvline(
                            true_parameters[i],
                            color="black",
                            linestyle="--",
                            label="True",
                        )

                    if len(extra_indexes) > 0:
                        for j, index in enumerate(extra_indexes):
                            line = extra_lines[j][0]
                            ax.axvline(
                                samples[index, i],
                                color=line.get_color(),
                                linestyle="--",
                                label=extra_labels[j],
                            )
                    percentiles = np.nanpercentile(samples[:, i], [16, 50, 84])
                    upper = percentiles[2] - percentiles[1]
                    lower = percentiles[1] - percentiles[0]
                    ax.text(
                        0.04,
                        0.96,
                        f"${percentiles[1]:.2f}^{{+{upper:.2f}}}_{{-{lower:.2f}}}$",
                        transform=ax.transAxes,
                        fontsize=8,
                        verticalalignment="top",
                        horizontalalignment="left",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                    )

            if plot_name is None:
                plot_name = f"{self.name}_SED_{self._timestamp}.png"
            # plt.show(block=False)
            if fig is not None and save_plots:
                logger.info(f"Saving SED plot to {plots_dir}/{plot_name}")
                if not os.path.exists(plots_dir):
                    os.makedirs(plots_dir)
                fig.savefig(os.path.join(plots_dir, plot_name), dpi=200)
            if not save_plots:
                plt.show(block=False)

        else:
            fig = None

        return fnu_quantiles, wav, phot_fnu_draws, phot_wav, fig

    def sample_posterior(
        self,
        X_test: np.ndarray = None,
        sample_method: str = "direct",
        sample_kwargs: dict = {},
        posteriors: object = None,
        num_samples: int = 1000,
        timeout_seconds_per_test=30,
        log_times=False,
        **kwargs,
    ) -> np.ndarray:
        """Sample from the posterior distribution.

        Parameters:
            X_test: Test feature array. If None, will use the stored test data.
            sample_method: Method to use for sampling. Options are 'direct', 'emcee',
                'pyro', or 'vi'.
            sample_kwargs: Additional keyword arguments for the sampler.
            posteriors: List of posterior distributions. If None, will use the stored
                posteriors.
            num_samples: Number of samples to draw from the posterior.
            timeout_seconds_per_test: Timeout in seconds for each test sample.

        Returns:
            A numpy array of samples drawn from the posterior distribution.
            Shape (num_objects, num_samples, num_parameters).
        """
        if posteriors is None:
            posteriors = self.posteriors

        if X_test is None:
            if hasattr(self, "_X_test") and getattr(self, "_X_test") is not None:
                X_test = self._X_test
            else:
                raise ValueError("X_test must be provided or set in the object.")

        if sample_method == "direct":
            sampler = DirectSampler
        elif sample_method in ["emcee", "mcmc"]:
            sampler = EmceeSampler
        elif sample_method == "pyro":
            sampler = PyroSampler
        elif sample_method == "vi":
            sampler = VISampler
        else:
            raise ValueError("Invalid sample method. Use 'direct', 'emcee'/'mcmc', 'pyro' or 'vi'.")

        sampler = sampler(posteriors, **sample_kwargs)

        # Properly check dimensionality and shape
        X_test_array = np.asarray(X_test)  # Ensure it's a numpy array
        X_test_array = torch.as_tensor(X_test_array, device=self.device)
        # Handle single sample case
        single = False
        if X_test_array.ndim == 1 or (X_test_array.ndim == 2 and X_test_array.shape[0] == 1):
            # Just ensure shape is correct
            X_test_array = torch.unsqueeze(X_test_array, 0)
            single = True
            # print("Single sample inference.")
            # return sample_with_timeout(sampler, num_samples, X_test_array,
            #                                   timeout_seconds_per_test)

        # ISSUE! sample_batched seems to be much slower than sampling one by one!
        """
        if hasattr(posteriors, "sample_batched"):
            X_test_array = torch.squeeze(X_test_array)
            samples = (
                posteriors.sample_batched((1,), x=X_test_array, show_progress_bars=True)
                .detach()
                .cpu()
                .numpy()
            )  # noqa E501
            samples = np.transpose(samples, (1, 0, 2))
            return samples
        """

        # Handle multiple samples case
        shape = len(self.fitted_parameter_names)
        # test_sample = sample_with_timeout(
        #    sampler, num_samples, X_test_array[0], timeout_seconds_per_test
        # )
        # shape = test_sample.shape

        if log_times:
            times = []
        # Draw samples from the posterior
        samples = np.zeros((len(X_test_array), num_samples, shape))
        # samples[0] = test_sample  # First sample is already drawn
        for i in trange(0, len(X_test_array), desc="Sampling from posterior"):
            start_time = time.time()
            try:
                # print(X_test_array[i])
                samples[i] = sampler.sample(nsteps=num_samples, x=X_test_array[i], progress=False)
                """samples[i] = sample_with_timeout(
                    sampler,
                    num_samples,
                    X_test_array[i],
                    timeout_seconds_per_test,
                )"""
            except TimeoutException:
                logger.error(
                    f"""Timeout exceeded for sample {i}.
                    Returning empty array for this sample."""
                )
                samples[i] = np.nan
            except KeyboardInterrupt:
                logger.warning("Sampling interrupted by user. Returning samples collected so far.")
                break
            except Exception as e:
                logger.error(f"Error occurred while sampling for sample {i}: {e}")
                samples[i] = np.nan
            if log_times:
                elapsed = time.time() - start_time
                times.append(elapsed)

        if log_times and len(times) > 0:
            logger.info(
                f"Median time per sample: {np.median(times):.5f} seconds."
                f"16th-84th: {np.percentile(times, 16):.5f}-{np.percentile(times, 84):.5f}s."
            )

        if single:
            samples = np.squeeze(samples, 0)

        return samples

    @property
    def likelihood_func(self):
        if hasattr(self, "posteriors") and self.posteriors is not None:
            likelihood = self.posteriors.potential_fn.potential_fns[0].likelihood_estimator
            return likelihood
        return None

    def evaluate_model(
        self,
        posteriors: list = None,
        X_test: np.ndarray = None,
        y_test: np.ndarray = None,
        num_samples: int = 1000,
        verbose: bool = True,
        samples=None,
        independent_metrics: bool = True,
        return_samples: bool = False,
    ) -> dict:
        """Evaluate the trained model on test data.

        .. note::
            The following metrics are calculated:

            - **mse**: Mean Squared Error (Lower is better).
              Average of squared differences between predicted and actual values.
            - **rmse**: Root Mean Squared Error (Lower is better).
              Square root of MSE, in the same units as the target.
            - **mae**: Mean Absolute Error (Lower is better).
              Average of absolute differences, less sensitive to outliers.
            - **median_ae**: Median Absolute Error (Lower is better).
              Median of absolute differences, robust to outliers.
            - **r_squared**: Coefficient of Determination (1 is perfect).
              Proportion of variance in the target explained by the model.
            - **rmse_norm**: Normalized RMSE (Lower is better).
              RMSE divided by the range of the target variable.
            - **mae_norm**: Normalized MAE (Lower is better).
              MAE divided by the range of the target variable.
            - **tarp**: Tests of Accuracy with Random Points (Lower is better).
              Probability of predictions falling within a range of true values.
            - **log_dpit_max**: (Lower is better).
              Log of the max distance between predicted and true values.
            - **mean_log_prob**: Mean log probability of the predictions given
              the true values.

        Args:
            X_test (np.ndarray): Test feature array.
            y_test (np.ndarray): Test target array.
            X_scaler (StandardScaler, optional): Scaler for the features (if used).
                Defaults to None.
            y_scaler (StandardScaler, optional): Scaler for the targets (if used).
                Defaults to None.
            num_samples (int, optional): Number of samples to draw from the
                posterior. Defaults to 1000.
            verbose (bool, optional): Whether to print verbose output.
                Defaults to True.
            posteriors (List[Any], optional): List of posterior distributions.
                If None, will use the stored posteriors. Defaults to None.
            samples (np.ndarray, optional): Precomputed samples from the
                posterior. If None, will draw new samples. Defaults to None.
            independent_metrics (bool, optional): If True, calculate metrics
                independently for each parameter. Defaults to True.
            return_samples (bool, optional): If True, return the samples
                used for evaluation. Defaults to False.

        Raises:
            ValueError: If posteriors or test data are not provided.

        Returns:
            Union[Dict[str, Any], Tuple[Dict[str, Any], np.ndarray]]:
                A dictionary of evaluation metrics. If `return_samples` is True,
                returns a tuple of (metrics_dict, samples_array).
        """
        if posteriors is None:
            if hasattr(self, "posteriors"):
                posteriors = self.posteriors
            else:
                raise ValueError("Posteriors must be provided or set in the object.")

        if X_test is None or y_test is None:
            if hasattr(self, "_X_test") and hasattr(self, "_y_test"):
                X_test = self._X_test
                y_test = self._y_test
            else:
                raise ValueError("X_test and y_test must be provided or set in the object.")

        if isinstance(samples, str):
            # Load samples from a file
            if samples.endswith(".npy"):
                samples = np.load(samples)
            elif samples.endswith(".pt") or samples.endswith(".pth"):
                samples = torch.load(samples).numpy()
            else:
                raise ValueError("Unsupported file format for samples. Use .npy or .pt/.pth.")

        elif samples is None:
            # Draw samples from the posterior
            samples = self.sample_posterior(X_test, num_samples=num_samples, posteriors=posteriors)

        assert samples.ndim == 3, (
            "Samples must have shape (num_objects, num_samples, num_parameters)."
        )

        if samples.shape[0] != len(X_test):
            if samples.ndim == 3 and samples.shape[1] == len(X_test):
                # Transpose samples if the shape is (num_samples, num_objects, num_parameters)
                samples = np.transpose(samples, (1, 0, 2))
                logger.warning(
                    "Transposing samples to match shape (num_objects, num_samples, num_parameters)."
                )

        assert samples.shape[0] == len(X_test), (
            f"Samples must match the number of test samples, {samples.shape[0]} != {len(X_test)}."
            f"Samples shape: {samples.shape}"
        )  # noqa E501
        if samples.shape[1] != num_samples:
            raise ValueError(
                f"Samples must have {num_samples} samples per test sample, but got {samples.shape[1]}."  # noqa E501
            )

        # Calculate basic metrics
        mean_pred = np.mean(samples, axis=1)
        median_pred = np.median(samples, axis=1)

        if independent_metrics:
            axis = 0  # Calculate metrics independently for each parameter
        else:
            axis = None  # Calculate metrics across all parameters
        # R-squared
        ss_res = np.sum((y_test - mean_pred) ** 2, axis=axis)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2, axis=axis)

        # Normalized metrics
        rmse_normalized = np.sqrt(np.mean((y_test - mean_pred) ** 2, axis=axis)) / np.std(y_test)
        mae_normalized = np.mean(np.abs(y_test - mean_pred), axis=axis) / np.std(y_test)

        # PIT
        pit = self.calculate_PIT(X_test, y_test, samples=samples, posteriors=posteriors)

        dpit_max = np.max(np.abs(pit - np.linspace(0, 1, len(pit))))
        log_dpit_max = -0.5 * np.log(dpit_max)

        # has shape (200, 1000, 6)
        # need num_samples, n_sims, n_dims).
        tarp_samples = np.transpose(samples, (1, 0, 2))

        # Calculate metrics
        metrics = {
            "MSE": np.mean((y_test - mean_pred) ** 2, axis=axis),
            "RMSE": np.sqrt(np.mean((y_test - mean_pred) ** 2, axis=axis)),
            "mean_ae": np.mean(np.abs(y_test - mean_pred), axis=axis),
            "median_ae": np.median(np.abs(y_test - median_pred), axis=axis),
            "R_squared": 1 - (ss_res / ss_tot),
            "RMSE_norm": rmse_normalized,
            "mean_ae_norm": mae_normalized,
            "tarp": np.array(
                [
                    self.calculate_TARP(X_test, y_test, samples=tarp_samples, posteriors=posteriors)
                    .cpu()
                    .numpy()
                ]
            ),
            "log_dpit_max": log_dpit_max,
        }

        try:
            metrics["mean_log_prob"] = np.mean(self.log_prob(X_test, y_test, posteriors))
        except Exception:
            pass

        metrics = make_serializable(metrics)

        if verbose:
            # print a nicely formatted table of the metrics.
            # For metrics which are per parameter, print as a table
            # with parameter names as columns.
            param_metrics = []
            full_metrics = []

            for metric in metrics:
                # print(metric, type(metrics[metric]))
                if (
                    isinstance(metrics[metric], (np.ndarray, list, torch.Tensor, jnp.ndarray))
                    and hasattr(metrics[metric], "__len__")
                    and len(metrics[metric]) == len(self.fitted_parameter_names)
                ):
                    try:
                        metric = metric.tolist()
                        metrics[metric] = metrics[metric].tolist()
                    except AttributeError:
                        pass

                    param_metrics.append(metric)
                else:
                    full_metrics.append(metric)

            # Print full metrics
            logger.info("=" * 60)
            logger.info("MODEL PERFORMANCE METRICS")
            logger.info("=" * 60)

            if len(full_metrics) > 0:
                logger.info("Full Model Metrics:")
                logger.info("-" * 40)
                for metric in full_metrics:
                    metric_name = metric.replace("_", " ").upper()
                    num = metrics[metric]
                    if isinstance(num, np.ndarray):
                        num = num.item()
                    elif isinstance(num, list) and len(num) == 1:
                        num = num[0]
                    try:
                        logger.info(f"{metric_name:.<25} {num:.6f}")
                    except Exception:
                        logger.info(f"{metric_name:.<25} {num}{type(num)}")

            # Print parameter-specific metrics
            if len(param_metrics) > 0:
                logger.info("Parameter-Specific Metrics:")
                logger.info("-" * 40)

                # Calculate column widths
                metric_col_width = max(
                    len(metric.replace("_", " ").upper()) for metric in param_metrics
                )
                metric_col_width = max(metric_col_width, len("Metric"))  # Ensure header fits

                param_col_widths = []
                for param_name in self.fitted_parameter_names:
                    # Calculate width needed for parameter name and values
                    max_value_width = 0
                    for metric in param_metrics:
                        value_str = f"{metrics[metric][list(self.fitted_parameter_names).index(param_name)]:.6f}"  # noqa E501
                        max_value_width = max(max_value_width, len(value_str))

                    col_width = max(len(param_name), max_value_width) + 2  # Add padding
                    param_col_widths.append(col_width)

                # Create header
                header = f"{'Metric':<{metric_col_width}}"
                for i, param_name in enumerate(self.fitted_parameter_names):
                    header += f"{param_name:>{param_col_widths[i]}}"
                logger.info(header)
                logger.info("-" * len(header))

                # Print each metric row
                for metric in param_metrics:
                    metric_name = metric.replace("_", " ").upper()
                    row = f"{metric_name:<{metric_col_width}}"
                    for i, param_name in enumerate(self.fitted_parameter_names):
                        row += f"{metrics[metric][i]:>{param_col_widths[i]}.6f}"
                    logger.info(row)

            logger.info("=" * 60)
        # convert numpy arrays to lists for JSON serialization

        if return_samples:
            return metrics, samples
        else:
            return metrics

    def plot_diagnostics(
        self,
        X_train: np.ndarray = None,
        y_train: np.ndarray = None,
        X_test: np.ndarray = None,
        y_test: np.ndarray = None,
        stats: list = None,
        plots_dir: str = f"{code_path}/models/name/plots/",
        sample_method: str = "direct",
        posteriors: object = None,
        online: bool = False,
        num_samples: int = 1000,
    ) -> None:
        """Plot the diagnostics of the SBI model.

        Args:
            X_train (np.ndarray): Training feature array.
            y_train (np.ndarray): Training target array.
            X_test (np.ndarray): Test feature array.
            y_test (np.ndarray): Test target array.
            stats (Optional[List[str]], optional): List of statistics to plot.
                If None, will use `self.stats`. Defaults to None.
            plots_dir (str, optional): Directory to save the plots.
            sample_method (str, optional): Method to use for sampling.
                Options are 'direct', 'emcee', 'pyro', or 'vi'.
                Defaults to 'direct'.
            posteriors (Optional[List[Any]], optional): List of posterior distributions.
                If None, will use `self.posteriors`. Defaults to None.
            online (bool, optional): If True, will not plot the posterior
                and coverage. Defaults to False.
            num_samples (int, optional): Number of samples to draw from the
                posterior for plotting for each test sample. Samples will
                be saved in the plots_dir. Defaults to 1000.
        """
        plots_dir = plots_dir.replace("name", self.name)
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        if stats is None:
            if hasattr(self, "stats"):
                stats = self.stats

        if X_train is None or y_train is None and not online:
            if hasattr(self, "_X_train") and hasattr(self, "_y_train"):
                X_train = self._X_train
                y_train = self._y_train
            else:
                raise ValueError("X_train and y_train must be provided or set in the object.")
        if X_test is None or y_test is None and not online:
            if hasattr(self, "_X_test") and hasattr(self, "_y_test"):
                X_test = self._X_test
                y_test = self._y_test
            else:
                raise ValueError("X_test and y_test must be provided or set in the object.")

        # Plot the loss
        if stats is not None:
            self.plot_loss(stats, plots_dir=plots_dir)

        # DEBUG
        # posteriors._modules['posteriors'][0].prior = self.create_priors()#prior=Uniform)

        if not online:
            # Plot the posterior
            self.plot_posterior(
                X=X_test,
                y=y_test,
                plots_dir=plots_dir,
                sample_method=sample_method,
                posteriors=posteriors,
                num_samples=num_samples,
            )

        # Plot the coverage
        self.plot_coverage(
            X=X_test,
            y=y_test,
            plots_dir=plots_dir,
            sample_method=sample_method,
            posteriors=posteriors,
            num_samples=num_samples,
        )

    def plot_loss(
        self,
        summaries: list = "self",
        plots_dir: str = f"{code_path}/models/name/plots/",
        overwrite: bool = False,
    ) -> None:
        """Plot the loss of the SBI model."""
        if summaries == "self":
            if hasattr(self, "stats"):
                summaries = self.stats
            else:
                raise ValueError("No summaries found. Please provide the summaries.")

        plots_dir = plots_dir.replace("name", self.name)

        if os.path.exists(f"{plots_dir}/loss.png") and not overwrite:
            return

        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        # plot train/validation loss
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        for i, m in enumerate(summaries):
            tlp = m.get("training_log_probs", None)
            if tlp is None:
                if "training_loss" not in m:
                    continue
                tlp = -1.0 * np.array(m["training_loss"])

            vlp = m.get("validation_log_probs", None)
            if vlp is None:
                vlp = -1.0 * np.array(m["validation_loss"])

            ax.plot(tlp, ls="-", label=f"{i}_train")
            ax.plot(vlp, ls="--", label=f"{i}_val")
        ax.set_xlim(0)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Log probability")
        ax.legend()

        fig.savefig(f"{plots_dir}/loss.png", dpi=300)

    def plot_histogram_parameter_array(
        self,
        bins="knuth",
        plots_dir: str = f"{code_path}/models/name/plots/",
        seperate_test_train=False,
        max_bins_row=6,
    ):
        """Plot histogram of each parameter using astropy.visualization.hist."""
        nrow = int(np.ceil(len(self.fitted_parameter_names) / max_bins_row))
        ncol = min(len(self.fitted_parameter_names), max_bins_row)
        fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3 * nrow), squeeze=False)
        axes = axes.flatten()  # Flatten the axes array for easier indexing

        for i, param in enumerate(self.simple_fitted_parameter_names):
            axes[i].set_title(param)
            unit = (
                f" ({self.fitted_parameter_units[i]})"
                if i < len(self.fitted_parameter_units)
                else ""
            )
            axes[i].set_xlabel(f"Value{unit}")
            axes[i].set_ylabel("Count")
            if seperate_test_train:
                hist(self._y_train[:, i], ax=axes[i], bins=bins, label="Train")
                hist(self._y_test[:, i], ax=axes[i], bins=bins, label="Test")
                axes[i].legend()
            else:
                hist(self.fitted_parameter_array[:, i], ax=axes[i], bins=bins)

        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()

        plots_dir = plots_dir.replace("name", self.name)

        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        logger.info(f"saving {plots_dir}/param_histogram.png")
        fig.savefig(f"{plots_dir}/param_histogram.png", dpi=300)

        return fig

    def plot_histogram_feature_array(
        self,
        bins="knuth",
        plots_dir: str = f"{code_path}/models/name/plots/",
        seperate_test_train=False,
        max_bins_row=6,
        comparison_array: np.ndarray = None,
        density: bool = False,
        log: bool = False,
    ):
        """Plot histogram of each feature using astropy.visualization.hist."""
        if self.observation_type == "spectra":
            logger.warning("Plotting histograms for spectra is not implemented yet.")
            return None

        nrow = int(np.ceil(len(self.feature_names) / max_bins_row))
        ncol = min(len(self.feature_names), max_bins_row)
        fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3 * nrow), squeeze=False)
        axes = axes.flatten()  # Flatten the axes array for easier indexing

        for i, feature in enumerate(self.feature_names):
            axes[i].set_title(feature)
            if self.feature_units is None:
                unit = ""
            else:
                unit = f" ({self.feature_units[i]})" if i < len(self.feature_units) else ""
            axes[i].set_xlabel(f"Value {unit}")

            if seperate_test_train:
                hist(
                    self._X_train[:, i],
                    ax=axes[i],
                    bins=bins,
                    label="Train",
                    density=density,
                )
                hist(
                    self._X_test[:, i],
                    ax=axes[i],
                    bins=bins,
                    label="Test",
                    density=density,
                )
                axes[i].legend()
            else:
                hist(self.feature_array[:, i], ax=axes[i], bins=bins)
            if comparison_array is not None:
                hist(
                    comparison_array[:, i],
                    ax=axes[i],
                    bins=bins,
                    alpha=0.5,
                    density=density,
                )

            if log:
                axes[i].set_yscale("log")

        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()

        plots_dir = plots_dir.replace("name", self.name)

        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        logger.info(f"saving {plots_dir}/feature_histogram.png")
        fig.savefig(f"{plots_dir}/feature_histogram.png", dpi=300)

        return fig

    def plot_posterior(
        self,
        ind: int = "random",
        X: np.ndarray = None,
        y: np.ndarray = None,
        seed: int = None,
        num_samples: int = 1000,
        sample_method: str = "direct",
        sample_kwargs: dict = {},
        plots_dir: str = f"{code_path}/models/name/plots/",
        plot_kwargs=dict(fill=True),
        posteriors: object = None,
        **kwargs: dict,
    ) -> None:
        """Plot the posterior of the SBI model."""
        plots_dir = plots_dir.replace("name", self.name)

        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        if X is None or y is None:
            if hasattr(self, "_X_test") and hasattr(self, "_y_test"):
                X = self._X_test
                y = self._y_test
                logger.info(
                    "Defaulting to _X_test and _y_test. "
                    "If this is not what you want, please provide X and y explicitly."
                )  # noqa E501
            else:
                raise ValueError("X and y must be provided or set in the object.")
        # Drop leading dimensions and then ensure still 2D

        X = np.squeeze(X)
        y = np.squeeze(y)
        X = np.atleast_2d(X)
        if len(self.fitted_parameter_names) > 1:
            y = np.atleast_2d(y)

        if ind == "random":
            if seed is not None:
                np.random.seed(seed)
            ind = np.random.randint(0, len(X))

        draw = y[ind]

        logger.info(y[ind])

        # draw = np.atleast_2d(draw) # doesn't work for mutli dimension y

        # use ltu-ili's built-in validation metrics to plot the posterior for this point
        metric = PlotSinglePosterior(
            num_samples=num_samples,
            sample_method=sample_method,
            labels=self.simple_fitted_parameter_names,
            out_dir=plots_dir,
            **sample_kwargs,
        )

        if posteriors is None:
            posteriors = self.posteriors

        # give xobs a batch dimension
        x_obs = torch.tensor(np.array([X[ind]]), dtype=torch.float32, device=self.device)

        fig = metric(
            posterior=posteriors,
            x_obs=x_obs,
            theta_fid=draw,
            plot_kws=plot_kwargs,
            signature=f"{self.name}_{ind}_",
            **kwargs,
        )
        if self.observation_type == "photometry":
            text = "\n".join(
                [
                    f"{self.feature_names[i]}: {X[ind][i]:.3f} {self.feature_units[i]}"
                    for i in range(len(X[ind]))
                ]
            )

            fig.fig.text(
                0.95,
                0.95,
                text,
                fontsize=23,
                ha="right",
                va="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
            )

        fig.fig.savefig(
            os.path.join(plots_dir, f"{self.name}_{ind}_plot_single_posterior.jpg"),
            dpi=200,
        )

        return fig

    '''
    def plot_posterior_samples(self):
        """Plot the posterior samples of the SBI model."""
        pass

    def plot_posterior_predictions(self):
        """Plot the posterior predictions of the SBI model."""
        pass
    '''

    def calculate_TARP(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_samples: int = 1000,
        posteriors: object = None,
        num_bootstrap=200,
        samples: np.ndarray = None,
    ) -> np.ndarray:
        """Calculate Tests of Accuracy with Random Points (TARP) for the samples.

        Parameters
        ----------
        X : 2-dimensional array of shape (num_samples, n_features)
            Feature array.
        y : 1-dimensional array of shape (num_samples,)
            Target variable.

        Returns:
        -------
        tarp : 1-dimensional array
            The TARP values.
        """
        if samples is None:
            samples = self.sample_posterior(X, num_samples=num_samples, posteriors=posteriors)

        ecp, _ = tarp.get_tarp_coverage(
            samples,
            y,
            norm=True,
            bootstrap=True,
            num_bootstrap=num_bootstrap,
        )

        tarp_val = torch.mean(torch.from_numpy(ecp[:, ecp.shape[1] // 2])).to(self.device)

        return abs(tarp_val - 0.5)

    def calculate_PIT(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_samples: int = 1000,
        posteriors: object = None,
        samples: np.ndarray = None,
    ) -> np.ndarray:
        """Calculate the probability integral transform (PIT) for the samples.

        Parameters
        ----------
        X : 2-dimensional array of shape (num_samples, n_features)
            Feature array.
        y : 1-dimensional array of shape (num_samples,)
            Target variable.

        Returns:
        -------
        pit : 1-dimensional array
            The PIT values.
        """
        if samples is None:
            samples = self.sample_posterior(X, num_samples=num_samples, posteriors=posteriors)

        pit = np.empty(len(y))
        for i in range(len(y)):
            pit[i] = np.mean(samples[i] < y[i])

        pit = np.sort(pit)
        pit /= pit[-1]

        return pit

    def log_prob(
        self,
        X: np.ndarray,
        y: np.ndarray,
        posteriors: object = None,
        verbose=True,
    ) -> np.ndarray:
        """Calculate the log-probability of the posterior for the given samples.

        Parameters
        ----------
        X : 2-dimensional array of shape (num_samples, n_features)
            Feature array.
        y : 2-dimensional array of shape (num_samples, n_parameters)
            Test parameter values.
        posteriors : object, optional
            Posteriors to use for the log-probability calculation.
            If None, will use the posteriors stored in the object.
        verbose : bool, optional
            Whether to print progress information. Default is True.

        Returns:
        -------
        lp : 1-dimensional array
            The log-probability values for each sample.
        """
        lp = np.empty((len(X)))

        if posteriors is None:
            posteriors = self.posteriors

        for i in tqdm(range(len(X)), disable=not verbose, desc="Log prob"):
            x = torch.tensor(np.array([X[i]]), dtype=torch.float32, device=self.device)
            theta = torch.tensor(np.array([y[i]]), dtype=torch.float32, device=self.device)
            lp[i] = posteriors.log_prob(x=x, theta=theta)  # norm_posterior=True)

        return lp

    def plot_latent_residual(self):
        """Plot the latent residual of the SBI model."""
        pass

    def calculate_MAP(self):
        """Calculate the maximum a posteriori (MAP) estimate of the SBI model."""
        # DirectPosterior has a MAP method - could use that
        pass

    def plot_coverage(
        self,
        X: np.ndarray = None,
        y: np.ndarray = None,
        posterior_plot_type: Union[str, int] = "total",
        num_samples: int = 1000,
        sample_method: str = "direct",
        sample_kwargs: dict = {},
        plot_list=["predictions", "histogram", "logprob", "coverage", "tarp"],
        plots_dir: str = f"{code_path}/models/name/plots/",
        n_test_draws: int = 1000,
        posteriors: object = None,
        overwrite: bool = False,
    ) -> None:
        """Plot the coverage of the SBI model.

        Parameters
        ----------

        X : 2-dimensional array of shape (num_samples, n_features), optional
            Feature array. If None, will use the test set stored in the object.
        y : 2-d array of true target values, optional
            Target variable. If None, will use the test set stored in the object.
        posterior_plot_type : str or int, optional
            Type of posterior plot to use. Can be 'total', 'seperate',
            or an index of the posterior.
        num_samples : int, optional
            Number of samples to draw from the posterior. Default is 1000.
        sample_method : str, optional
            Method to use for sampling from the posterior. Default is 'direct'.
        sample_kwargs : dict, optional
            Additional keyword arguments for the sampling method.
        plot_list : list, optional
            List of plots to include in the coverage plot. Default is
            ['predictions', 'histogram', 'logprob', 'coverage', 'tarp'].
        plots_dir : str, optional
            Directory to save the plots. Default is f"{code_path}/models/name/plots/".
        n_test_draws : int, optional
            Number of test draws to generate if X and y are not provided.
        posteriors : object, optional
            Posteriors to use for the coverage plot. If None, will use the
            posteriors stored in the object.

        Returns:
        -------
        fig : matplotlib Figure
            The figure containing the coverage plots.
        """
        if X is None or y is None:
            if (
                hasattr(self, "_X_test")
                and hasattr(self, "_y_test")
                and self._X_test is not None
                and self._y_test is not None
            ):
                X = self._X_test
                y = self._y_test
            else:
                X, y = self.generate_pairs_from_simulator(n_test_draws)

        plots_dir = plots_dir.replace("name", self.name)

        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        remove = []
        for metric in plot_list:
            m_name = {
                "tarp": "plot_TARP",
                "histogram": "ranks_histogram",
                "logprob": "plot_true_logprobs",
                "coverage": "plot_coverage",
                "predictions": "plot_predictions",
            }
            m = m_name.get(metric, metric)
            if os.path.exists(f"{plots_dir}/{m}.jpg") and not overwrite:
                remove.append(metric)

        for metric in remove:
            plot_list.remove(metric)

        if len(plot_list) == 0:
            return

        # Drop leading dimensions and then ensure still 2D
        X = np.squeeze(X)
        y = np.squeeze(y)
        if X.ndim == 1:
            raise ValueError(
                "X must be a 2D array. Can't assess coverage "
                "for a single sample. Please provide a 2D array."
            )

        if len(self.fitted_parameter_names) == 1:
            y = np.expand_dims(y, axis=-1)
        logger.info(f"shapes: X:{X.shape}, y:{y.shape}")

        metric = PosteriorCoverage(
            num_samples=num_samples,
            sample_method=sample_method,
            labels=self.simple_fitted_parameter_names,
            plot_list=plot_list,
            out_dir=plots_dir,
            save_samples=True,
            **sample_kwargs,
        )

        # samples dir is
        # samples_path = f"{plots_dir}/single_samples.npy"
        # move file to validation_samples.npy
        # if os.path.exists(samples_path):
        #    os.rename(samples_path, f"{plots_dir}/validation_samples.npy")

        if posteriors is None:
            posteriors = self.posteriors

        fig = None
        if posterior_plot_type == "seperate":
            for i in range(len(posteriors)):
                fig = metric(
                    posterior=posteriors[i],
                    x=X,
                    theta=y,
                    fig=fig,
                )
        else:
            fig = metric(
                posterior=(
                    posteriors
                    if posterior_plot_type == "total"
                    else posteriors[posterior_plot_type]
                ),
                x=X,
                theta=y,  # X and y are the feature and target arrays
            )

        return fig

    def run_validation_from_file(
        self,
        validation_file: str,
        plots_dir: str = f"{code_path}/models/name/plots/",
        metrics: list = [
            "coverage",
            "histogram",
            "predictions",
            "tarp",
            "logprob",
        ],
    ) -> None:
        """Run the validation from a file."""
        posterior = ValidationRunner.load_posterior_sbi(validation_file)

        plots_dir = plots_dir.replace("name", self.name)
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        runner = ValidationRunner(
            posterior=posterior,
            metrics=metrics,
            out_dir=plots_dir,
            name=f"{self.name}_validation_",
        )

        runner(self._loader)

    @property
    def validation_log_probs(self):
        """Validation set log-probability of each epoch for each net.

        Returns:
        ---------
        list of 1-dimensional arrays
        """
        if self.stats is None:
            raise RuntimeError("The regressor has not been fitted yet.")

        return [stat["validation_log_probs"] for stat in self.stats]

    @property
    def training_log_probs(self):
        """Training set log-probability of each epoch for each net.

        Returns:
        ---------
        list of 1-dimensional array
        """
        if self.stats is None:
            raise RuntimeError("The regressor has not been fitted yet.")

        return [stat["training_log_probs"] for stat in self.stats]

    def load_model_from_pkl(
        self,
        model_file: str,
        set_self: bool = True,
        load_arrays: bool = True,
    ) -> Tuple[list, dict, dict]:
        """Load the model from a pickle file.

        Parameters
        -----------
        model_file : str
            Path to the model file. Can be a directory or a file.
        set_self : bool, optional
            If True, set the attributes of the class to the loaded values.
            Default is True.
        load_arrays : bool, optional
            If True, load the feature and parameter arrays from the model file.
            Default is True.

        Returns:
        ---------
        posteriors : list
            List of posteriors loaded from the model file.
        stats : dict
            Dictionary of statistics loaded from the model file.
        params : dict
            Dictionary of parameters loaded from the model file.
        """
        if not os.path.exists(model_file):
            if os.path.exists(f"{code_path}/models/{model_file}"):
                model_file = f"{code_path}/models/{model_file}"
            else:
                raise ValueError(f"Model file {model_file} does not exist.")

        if os.path.isdir(model_file):
            files = glob.glob(os.path.join(model_file, "*_posterior.pkl"))
            if len(files) == 0:
                raise ValueError(f"No parameter files found in {model_file}.")
            elif len(files) > 1:
                raise ValueError(
                    f"""Multiple parameter files found in {model_file}.
                    Please specify a single file."""
                )
            model_file = files[0]

        try:
            with open(model_file, "rb") as f:
                posteriors = load(f)
        except (RuntimeError, pickle.UnpicklingError):
            # probably because we ran this model on
            # with a GPU which is not available now.
            # with open(model_file, "rb") as f:
            try:
                posteriors = torch.load(f, map_location=torch.device(self.device))
            except (ValueError, RuntimeError, pickle.UnpicklingError):
                from .utils import CPU_Unpickler

                with open(model_file, "rb") as f:
                    posteriors = CPU_Unpickler(f).load()
        except ModuleNotFoundError:
            from typing import Any, Mapping

            class RenamingUnpickler(pickle.Unpickler):
                """A custom Unpickler that can rename modules when loading a pickle."""

                def __init__(self, *args, renames: Mapping[str, str], **kwargs):
                    self.renames = renames
                    super().__init__(*args, **kwargs)

                def find_class(self, module: str, name: str) -> Any:
                    renamed_module = module
                    # Iterate through the rename mapping
                    for old_prefix, new_prefix in self.renames.items():
                        # Check for a direct match or a prefix match (e.g., 'a.c' starts with 'a.')
                        if module == old_prefix or module.startswith(old_prefix + "."):
                            # Replace the old prefix with the new one
                            renamed_module = new_prefix + module[len(old_prefix) :]
                            break  # Stop after the first successful replacement
                    return super().find_class(renamed_module, name)

            renames = {"sbifitter": "synference"}
            with open(model_file, "rb") as f:
                posteriors = RenamingUnpickler(f, renames=renames).load()

        stats = model_file.replace("posterior.pkl", "summary.json")

        if os.path.exists(stats):
            with open(stats, "r", encoding="utf-8") as f:
                stats = json.load(f)

            if set_self:
                self.stats = stats

        else:
            stats = None
            logger.info(f"Warning: No summary file found for {model_file}.")

        logger.info(f"Loaded model from {model_file}.")
        logger.info(f"Device: {self.device}")

        posteriors = move_to_device(posteriors, self.device)

        if set_self:
            self.posteriors = posteriors

        params = model_file.replace("posterior.pkl", "params.pkl")
        params_alt = model_file.replace("posterior.pkl", "params.h5")

        arrays = model_file.replace("posterior.pkl", "arrays.pkl")
        arrays_alt = model_file.replace("posterior.pkl", "arrays.h5")

        if os.path.exists(params) or os.path.exists(params_alt):
            if os.path.exists(params):
                try:
                    with open(params, "rb") as f:
                        params = load(f)
                except (RuntimeError, pickle.UnpicklingError) as e:
                    logger.error(e)
                    try:
                        params = torch.load(params, map_location=torch.device(self.device))
                    except RuntimeError as e:
                        logger.error(e)
                        from .utils import CPU_Unpickler

                        with open(params, "rb") as f:
                            params = CPU_Unpickler(f).load()
            elif os.path.exists(params_alt):
                # read with hickle
                import hickle as hkl

                with open(params_alt, "rb") as f:
                    params = hkl.load(f)

            if load_arrays and (os.path.exists(arrays) or os.path.exists(arrays_alt)):
                if os.path.exists(arrays):
                    try:
                        with open(arrays, "rb") as f:
                            arrays = load(f)
                    except (RuntimeError, pickle.UnpicklingError) as e:
                        print(e)
                        try:
                            arrays = torch.load(arrays, map_location=torch.device(self.device))
                        except RuntimeError as e:
                            print(e)
                            from .utils import CPU_Unpickler

                            with open(arrays, "rb") as f:
                                arrays = CPU_Unpickler(f).load()
                elif os.path.exists(arrays_alt):
                    # read with hickle
                    import hickle as hkl

                    with open(arrays_alt, "rb") as f:
                        arrays = hkl.load(f)

                if isinstance(arrays, dict):
                    params.update(arrays)

            if set_self:
                # Set attributes of class again.
                self.fitted_parameter_names = params["fitted_parameter_names"]
                self.simple_fitted_parameter_names = [
                    i.split("/")[-1] for i in self.fitted_parameter_names
                ]
                self.fitted_parameter_units = params.get(
                    "fitted_parameter_units", self.parameter_units
                )
                learning_type = params.get("learning_type", "offline")
                self.feature_names = params["feature_names"]

                # print(params.keys())
                if "feature_array_flags" in params:
                    self.feature_array_flags = params["feature_array_flags"]
                    self.has_features = True

                if "empirical_noise_models" in params:
                    if "empirical_noise_models" not in self.feature_array_flags:
                        self.feature_array_flags["empirical_noise_models"] = params[
                            "empirical_noise_models"
                        ]

                if learning_type == "offline":
                    self.feature_names = params["feature_names"]
                    self.feature_units = params.get("feature_units", None)
                    if load_arrays:
                        self.fitted_parameter_array = params["parameter_array"]
                        self.feature_array = params["feature_array"]
                        if self.feature_array is not None:
                            self.has_features = True

                        if "parameter_array" in params:
                            self.parameter_array = params["parameter_array"]
                        if "train_indices" in params:
                            self._train_indices = params["train_indices"]
                        if "test_indices" in params:
                            self._test_indices = params["test_indices"]
                        if "train_fraction" in params:
                            self._train_fraction = params["train_fraction"]
                        if hasattr(self, "_train_indices") and hasattr(self, "_test_indices"):
                            try:
                                self._X_test = self.feature_array[self._test_indices]
                                self._y_test = self.fitted_parameter_array[self._test_indices]
                                self._X_train = self.feature_array[self._train_indices]
                                self._y_train = self.fitted_parameter_array[self._train_indices]
                            except IndexError:
                                logger.warning("IndexError when trying to set train/test arrays. ")
                            self._X_test = np.squeeze(self._X_test)
                            self._y_test = np.squeeze(self._y_test)
                            self._X_train = np.squeeze(self._X_train)
                            self._y_train = np.squeeze(self._y_train)
                else:
                    self._X_test = None
                    self._y_test = None
                    self._X_train = None
                    self._y_train = None
                    self._train_indices = None
                    self._test_indices = None

                    self.simulator = params["simulator"]

                    self._num_simulations = params["num_simulations"]
                    self._num_online_rounds = params["num_online_rounds"]
                    self._initial_training_from_library = params["initial_training_from_library"]

                self._train_args = params["train_args"]
                self._prior = params["prior"]

                try:
                    self._ensemble_model_types = params["ensemble_model_types"]
                    self._ensemble_model_args = params["ensemble_model_args"]
                except KeyError:
                    pass

        else:
            logger.warning(f"No parameter file found for {model_file}.")
            params = None

        return posteriors, stats, params

    @property
    def _timestamp(self):
        """Get the current date and time as a string."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def generate_pairs_from_simulator(self, num_samples=1000):
        """Generate pairs of data from the simulator."""
        if not self.has_simulator:
            raise ValueError("No simulator found. Please provide a simulator.")

        if self.simulator is None:
            raise ValueError("No simulator found. Please provide a simulator.")

        if self._prior is None:
            raise ValueError("No prior found. Please provide a prior.")

        samples = self._prior.sample_n(num_samples)

        phot = []
        for i in range(len(samples)):
            p = self.simulator(samples[i])
            if isinstance(p, torch.Tensor):
                p = p.cpu().numpy()

            phot.append(p)
        phot = np.array(phot)
        phot = np.squeeze(phot)

        if isinstance(samples, torch.Tensor):
            samples = samples.cpu().numpy()

        # shape phot to be (num_simulations, num_features)
        # X, y
        return phot, samples


class MissingPhotometryHandler:
    """Handles missing photometry in SEDs using an sbi++ approach.

    This class separates the process into two main steps:
    1. Imputation: Generating a distribution of plausible, complete photometry vectors.
    2. Sampling: Drawing posterior samples from the SBI model for a given set of vectors.
    """

    def __init__(
        self,
        training_photometry: np.ndarray,
        posterior_estimator: SBI_Fitter,
        run_params: Optional[Dict[str, Any]] = None,
        photometry_units: Optional[str] = "AB",
        uncertainty_models: Optional[Dict[str, UncertaintyModel]] = None,
        band_names: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None,
        device: str = "cpu",
    ) -> None:
        """Initializes the MissingPhotometryHandler.

        Args:
            training_photometry: Array of shape (n_samples, n_bands) with complete photometry.
            posterior_estimator: An SBI_Fitter instance for posterior sampling.
            run_params: Optional dictionary of parameters for imputation.
            photometry_units: Units of the photometry (default: "AB").
            uncertainty_models: Optional dictionary of uncertainty models for each band.
            band_names: Optional list of band names corresponding to the photometry.
            feature_names: Optional list of feature names for the photometry.
            device: Device to run computations on (default: "cpu").
        """
        self.y_train = training_photometry
        self.posterior_estimator = posterior_estimator
        self.device = device
        self.uncertainty_models = uncertainty_models
        self.band_names = band_names
        self.photometry_units = photometry_units
        self.feature_names = feature_names

        if self.uncertainty_models:
            if self.band_names is None:
                raise ValueError("`band_names` must be provided if `uncertainty_models` are used.")
            n_bands = self.y_train.shape[1]
            if len(self.band_names) != n_bands:
                raise ValueError(
                    f"Length of `band_names` ({len(self.band_names)}) must match "
                    f"the number of bands in training_photometry ({n_bands})."
                )

        self.run_params: Dict[str, Any] = {
            "ini_chi2": 5.0,
            "max_chi2": 50.0,
            "nmc": 100,
            "nposterior": 1000,
            "tmax_all": 10,
            "verbose": False,
        }
        if run_params:
            self.run_params.update(run_params)

    def _get_neighbor_kdes(self, obs: Dict[str, np.ndarray]) -> List[stats.gaussian_kde]:
        """Finds nearest neighbors and creates KDEs for missing bands."""
        y_obs, sig_obs, invalid_mask = obs["mags_sbi"], obs["mags_unc_sbi"], obs["missing_mask"]
        # print('check', obs["mags_sbi"], obs["mags_unc_sbi"], obs["missing_mask"])
        invalid_mask = np.array(invalid_mask, dtype=bool)
        valid_idx, not_valid_idx = np.where(~invalid_mask)[0], np.where(invalid_mask)[0]

        chi2_nei = self._chi2dof(self.y_train[:, valid_idx], y_obs[valid_idx], sig_obs[valid_idx])

        # print(np.sort(chi2_nei)[:10])
        # print the 10 nearest observetions in chi2
        # for i in np.argsort(chi2_nei)[:10]:
        #    print(self.y_train[i], chi2_nei[i])

        _chi2_thres = self.run_params["ini_chi2"]
        while _chi2_thres <= self.run_params["max_chi2"]:
            idx_chi2_selected = np.where(chi2_nei <= _chi2_thres)[0]
            if len(idx_chi2_selected) >= 30:
                break
            _chi2_thres += 5
        else:
            if self.run_params["verbose"]:
                logger.warning(f"Failed to find 30 neighbors. Using {len(idx_chi2_selected)}.")
            if len(idx_chi2_selected) == 0:
                idx_chi2_selected = np.argsort(chi2_nei)[:100]

        success = len(idx_chi2_selected) >= 30

        # print(valid_idx, not_valid_idx, idx_chi2_selected, success)

        # print([neighbors_missing[:, i] for i in range(neighbors_missing.shape[1])])
        if success:
            neighbors_valid = self.y_train[:, valid_idx][idx_chi2_selected]
            neighbors_missing = self.y_train[:, not_valid_idx][idx_chi2_selected]

            dists = np.linalg.norm(y_obs[valid_idx] - neighbors_valid, axis=1)
            dists[dists == 0] = 1e-10
            weights = 1.0 / dists

            # print("Using", len(idx_chi2_selected), "neighbors with chi2 <=", _chi2_thres)
            # print([neighbors_missing[:, i] for i in range(neighbors_missing.shape[1])])
            kdes = [
                stats.gaussian_kde(neighbors_missing[:, i], bw_method=0.2, weights=weights)
                for i in range(neighbors_missing.shape[1])
            ]
            return kdes

        else:
            return None

    def _chi2dof(
        self, mags: np.ndarray, obsphot: np.ndarray, obsphot_unc: np.ndarray
    ) -> np.ndarray:
        """Calculates reduced chi-square."""
        chi2 = np.nansum(((mags - obsphot) / obsphot_unc) ** 2, axis=1)
        return chi2 / np.sum(np.isfinite(obsphot))

    ## --------------------------------------------------------------------
    ## Step 1: Generate Imputations for Missing Data
    ## --------------------------------------------------------------------
    def generate_imputations(
        self,
        obs: Dict[str, np.ndarray],
        true_flux_units: Optional[str] = None,
        out_units: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generates a set of plausible, complete observation vectors for an SED with missing data.

        Args:
            obs: Dictionary with observed data ('mags_sbi', 'mags_unc_sbi', 'missing_mask').
            true_flux_units: Units of the input flux for the noise model.
            out_units: Desired output units for the noise model.

        Returns:
            A tuple containing:
            - An array of imputed observation vectors, shape (nmc, n_features).
            - A dictionary with metadata ('success', 'timeout', 'count').
        """
        if self.run_params["verbose"]:
            logger.info(f"Generating {self.run_params['nmc']} imputations for missing bands.")

        st = time.time()
        kdes = self._get_neighbor_kdes(obs)

        if kdes is None:
            return np.nan, {"success": False, "timeout": False, "count": 0}

        success = True

        nbands = len(obs["mags_sbi"])
        not_valid_idx = np.where(obs["missing_mask"])[0]

        imputed_vectors = []
        timeout_flag = False

        for i in range(self.run_params["nmc"]):
            if (time.time() - st) / 60 > self.run_params["tmax_all"]:
                timeout_flag = True
                success = False
                break

            if self.uncertainty_models is None:
                # Mode 1: Flux-only
                x = np.copy(obs["mags_sbi"])
                for i, idx in enumerate(not_valid_idx):
                    x[idx] = kdes[i].resample(size=1)[0]
            else:
                # Mode 2: Flux + Uncertainty
                x = np.full(2 * nbands, np.nan)
                x[:nbands] = obs["mags_sbi"]
                x[nbands:] = obs["mags_unc_sbi"]
                for i, idx in enumerate(not_valid_idx):
                    band_name = self.band_names[idx]
                    model = self.uncertainty_models[band_name]
                    true_flux = kdes[i].resample(size=1)[0]
                    if isinstance(model, AsinhEmpiricalUncertaintyModel):
                        if self.photometry_units == "AB":
                            # Convert to Jy
                            true_flux = 10 ** ((true_flux - 8.90) / -2.5)
                        elif self.photometry_units == "asinh":
                            true_flux = asinh_to_f_jy(true_flux, model.b)
                            true_flux_units = None
                            out_units = None
                    scat_flux, flux_err = model.apply_noise(
                        true_flux, true_flux_units=true_flux_units, out_units=out_units
                    )
                    # print(true_flux, scat_flux, flux_err)
                    x[idx] = scat_flux
                    x[idx + nbands] = flux_err
            imputed_vectors.append(x)

        metadata = {"success": success, "timeout": timeout_flag, "count": len(imputed_vectors)}
        return np.array(imputed_vectors), metadata

    ## --------------------------------------------------------------------
    ## Step 2: Sample Posteriors from Completed Data
    ## --------------------------------------------------------------------
    def sample_posterior(self, observation_vectors: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Draws posterior samples from the SBI model for one or more observation vectors.

        Args:
            observation_vectors: A single vector (1D array) or a batch of vectors (2D array)
                                 representing one or more complete observations.

        Returns:
            An array of posterior samples, shape (n_vectors * nposterior, n_params).
        """
        if self.run_params["verbose"]:
            n_vecs = observation_vectors.shape[0] if observation_vectors.ndim > 1 else 1
            logger.info(f"Sampling posterior for {n_vecs} observation vector(s).")

        if observation_vectors.ndim == 1:  # Ensure input is 2D for batch processing
            observation_vectors = observation_vectors[np.newaxis, :]

        all_posts = self.posterior_estimator(
            X_test=observation_vectors, n_samples=self.run_params["nposterior"]
        )

        return np.concatenate(all_posts, axis=0)

    ## --------------------------------------------------------------------
    ## Wrapper for End-to-End Processing
    ## --------------------------------------------------------------------
    def process_observation(
        self,
        obs: Dict[str, np.ndarray],
        true_flux_units: Optional[str] = None,
        out_units: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Processes a single observation, handling missing bands and returning posterior samples.

        This is a convenience wrapper that calls `generate_imputations`
        followed by `sample_posterior`.
        """
        if not np.any(obs["missing_mask"]):
            print(f"No missing data: {obs}")
            # --- Case: Complete Data ---
            if self.uncertainty_models is None:
                x = obs["mags_sbi"]
            else:
                x = np.concatenate([obs["mags_sbi"], obs["mags_unc_sbi"]])

            posterior_samples = self.sample_posterior(x)
            return {"posterior_samples": posterior_samples, "success": True}

        # --- Case: Missing Data ---
        imputed_vectors, metadata = self.generate_imputations(obs, true_flux_units, out_units)

        if "extra" in obs and metadata["count"] > 0:
            full_imputed_vectors = np.full(
                (imputed_vectors.shape[0], len(self.feature_names)), np.nan
            )
            full_imputed_vectors[:, : len(imputed_vectors[0])] = imputed_vectors
            for feature, value in obs["extra"].items():
                index = list(self.feature_names).index(feature)
                full_imputed_vectors[:, index] = value
            imputed_vectors = full_imputed_vectors

        if metadata["count"] > 0 and len(imputed_vectors[0]) != len(self.feature_names):
            raise ValueError(
                f"Imputed vector length {len(imputed_vectors[0])} does not match "
                f"model feature length {len(self.feature_names)}."
            )

        # if np.any(~np.isfinite(imputed_vectors)):
        #    print(obs)
        #    print(imputed_vectors)
        ##   print(metadata)

        if np.all(metadata["count"] > 0):
            posterior_samples = self.sample_posterior(imputed_vectors)
            # Extract reconstructed photometry from the mean of the imputations
            reconstructed_phot = np.mean(
                [vec[: len(obs["mags_sbi"])] for vec in imputed_vectors], axis=0
            )

        else:
            reconstructed_phot = np.full_like(obs["mags_sbi"], np.nan)
            posterior_samples = np.array([])

        return {
            "posterior_samples": posterior_samples,
            "reconstructed_photometry": reconstructed_phot,
            "imputed_vectors": imputed_vectors,
            **metadata,
        }

    @classmethod
    def init_from_synference(cls, synference, **run_params):
        """Initialize from a fitted SBI model.

        Parameters:
        -----------
        synference : object
            Fitted SBI model object

        """
        feature_array_flags = getattr(synference, "feature_array_flags", None)
        uncertainty_models = None
        if feature_array_flags is not None:
            scatter_fluxes = feature_array_flags.get("scatter_fluxes", None)
            noise_models = feature_array_flags.get("empirical_noise_models", None)
            if scatter_fluxes and noise_models:
                uncertainty_models = noise_models

        raw_phot_names = feature_array_flags.get("raw_observation_names", None)
        idxs = np.where(np.isin(synference.feature_names, raw_phot_names))[0]
        training_photometry = synference.feature_array[:, idxs]

        units = feature_array_flags.get("normed_flux_units", "AB")

        # Swap units to asinh if all uncertainty models are asinh
        if uncertainty_models is not None:
            if all(
                isinstance(m, AsinhEmpiricalUncertaintyModel) for m in uncertainty_models.values()
            ):
                units = "asinh"

        # Need to figure out unit conversions here.
        return cls(
            training_photometry=training_photometry,
            posterior_estimator=synference.sample_posterior,
            band_names=raw_phot_names,
            uncertainty_models=uncertainty_models,
            run_params=run_params,
            device=synference.device,
            photometry_units=units,
            feature_names=synference.feature_names,
        )


class ModelComparison:
    """Class for model comparison using e.g. Evidence Networks or Harmonic Evidence."""

    def __init__(
        self,
        model1: SBI_Fitter,
        model2: SBI_Fitter,
        data: np.ndarray,
    ):
        """Initialize the model comparison class.

        Args:
            model1: First model to compare.
            model2: Second model to compare.
            data: Data to use for comparison.
        """
        self.model1 = model1
        self.model2 = model2
        self.data = data


class Simformer_Fitter(SBI_Fitter):
    """Simformer Fitter for SBI models.

    This class implements the Simformer architecture for SBI tasks.

    To Do:
    - Ensure more inherited methods actually work with Simformer.
    - Implement TARP and other metrics which are native to LtU-ILI.
    - Ensure methods to act on real observations and recover photometry work.

    """

    def __init__(self, name: str = "simformer_fitter", **kwargs):
        """Initialize the Simformer Fitter."""
        super().__init__(name=name, **kwargs)

        self.simformer_task = None

    @classmethod
    def init_from_hdf5(
        cls, model_name, hdf5_path: str = None, return_output: bool = False, **kwargs
    ):
        """Initialize the Simformer Fitter."""
        return super().init_from_hdf5(
            model_name,
            hdf5_path,
            return_output,
            **kwargs,
        )

    @classmethod
    def load_saved_model(cls, model_name: str, library_path: str, model_file: str, **kwargs):
        """Load a saved Simformer model from a file."""
        model = cls.init_from_hdf5(
            model_name=model_name,
            hdf5_path=library_path,
            return_output=False,
            **kwargs,
        )

        model.load_model_from_pkl(
            model_file,
            model_name,
            set_self=True,
        )
        return model

    def run_single_sbi(
        self,
        backend: str = "jax",
        num_training_simulations: int = 10_000,
        train_test_fraction: float = 0.9,
        random_seed: int = 42,
        set_self: bool = True,
        verbose: bool = True,
        load_existing_model: bool = True,
        name_append: str = "timestamp",
        save_method: str = "joblib",
        task_func: Optional[Callable] = None,
        model_config_dict_overrides: Optional[Dict[str, Any]] = None,
        sde_config_dict: Optional[Dict[str, Any]] = None,
        train_config_dict_overrides: Optional[Dict[str, Any]] = None,
        sde_config_dict_overrides: Optional[Dict[str, Any]] = None,
        attention_mask_type: str = "full",
        evaluate_model: bool = True,
    ):
        """Train a Simformer model using the provided configurations.

        This method sets up the Simformer task, prepares the data, trains
        the model using the specified configurations, and saves the trained
        model to a pickle file.

        Args:
            backend (str, optional): Backend to use for training ('jax' or 'torch').
                Defaults to "jax".
            num_training_simulations (int, optional): Number of training simulations
                to generate. Only used if `has_simulator` is True, otherwise
                the `feature_array` is used. Defaults to 10,000.
            train_test_fraction (float, optional): Fraction of samples to use for
                training. Defaults to 0.9.
            random_seed (int, optional): Random seed for reproducibility.
                Defaults to 42.
            set_self (bool, optional): If True, sets the trained model and task
                to the instance attributes. Defaults to True.
            verbose (bool, optional): If True, prints progress information
                during training. Defaults to True.
            load_existing_model (bool, optional): If True, loads an existing
                model from a pickle file if it exists. Defaults to True.
            name_append (str, optional): String to append to the model name in
                the output file. Defaults to "timestamp".
            save_method (str, optional): Method to use for saving the model
                (e.g., 'torch', 'pickle'). Defaults to "joblib".
            task_func (Callable, optional): Function to create the task. If None,
                uses the default `GalaxyPhotometryTask`. Defaults to None.
            model_config_dict_overrides (dict, optional): Dictionary to override
                the default model configuration. Defaults to None.
            sde_config_dict (dict, optional): Dictionary to override configs
                for the SDE. Defaults to a pre-defined VPSDE configuration.
            train_config_dict_overrides (dict, optional): Dictionary to override the
                training configuration. Defaults to a pre-defined configuration.
            sde_config_dict_overrides (dict, optional): Dictionary to override
                the SDE configuration. Defaults to None.
            attention_mask_type (str, optional): Type of attention mask to use
                ('full', 'causal', or 'none'). Defaults to "full".
            evaluate_model (bool, optional): If True, evaluates the trained
                model on the validation set and prints metrics. Defaults to True.


        - Add support for constraint functions during sampling to allow intervals
            (e.g., from `scoresbibm.methods.guidance` import
            `generalized_guidance`, `get_constraint_fn`).
        """
        from omegaconf import OmegaConf
        from scoresbibm.methods.score_transformer import train_transformer_model

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
        }
        if model_config_dict_overrides is not None:
            model_config_dict.update(model_config_dict_overrides)

        sde_config_dict = {
            "name": "VPSDE",  # or "VESDE"
            "beta_min": 0.1,
            "beta_max": 20.0,
            "num_steps": 1000,
            "T_min": 1e-05,
            "T_max": 1.0,
        }
        if sde_config_dict_overrides is not None:
            sde_config_dict.update(sde_config_dict_overrides)

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
        if train_config_dict_overrides is not None:
            train_config_dict.update(train_config_dict_overrides)

        if task_func is None:
            from .simformer import GalaxyPhotometryTask as task_func

        if name_append == "timestamp":
            name_append = f"{self._timestamp}"

        if len(name_append) > 0 and name_append[0] != "_":
            name_append = f"_{name_append}"

        out_path = f"{code_path}/models/{self.name}/{self.name}{name_append}_posterior.pkl"
        if load_existing_model and os.path.exists(out_path):
            logger.info(f"Loading existing model from {out_path}")
            trained_score_model, meta = self.load_model_from_pkl(
                model_dir=f"{code_path}/models/{self.name}/",
                model_name=f"{self.name}{name_append}_posterior",
                set_self=True,
            )
            run = True
        else:
            run = False

        priors = self.create_priors()

        if self.has_simulator:
            logger.info("Using online simulator for training.")
            simulator_function = self.simulator
            learning_type = "online"
        else:
            logger.info("Using pre-generated samples for training.")
            assert self.feature_array is not None, (
                "Feature array must be provided for pre-generated samples."
            )
            simulator_function = None
            learning_type = "offline"

        if not self.has_simulator:
            # Split the dataset into training and validation sets.
            train_indices, test_indices = self.split_dataset(
                train_fraction=train_test_fraction,
                random_seed=random_seed,
                verbose=verbose,
            )

            x = jnp.array(self.feature_array, dtype=jnp.float32)
            theta = jnp.array(self.fitted_parameter_array, dtype=jnp.float32)

            training_data = {
                "theta": theta[train_indices],
                "x": x[train_indices],
            }

            validation_data = {
                "theta": theta[test_indices],
                "x": x[test_indices],
            }

        task = task_func(
            name="galaxy_photometry",
            backend=backend,
            prior_dict=priors,
            param_names_ordered=self.fitted_parameter_names,
            run_simulator_fn=simulator_function,
            num_filters=len(self.feature_names),
            test_X_data=copy.deepcopy(validation_data["x"]),
            test_theta_data=copy.deepcopy(validation_data["theta"]),
            attention_mask_type=attention_mask_type,
        )

        method_config_dict = {
            "device": str(self.device),  # Ensure this matches device setup
            "sde": sde_config_dict,
            "model": model_config_dict,
            "train": train_config_dict,
        }

        # Convert the main method_cfg to OmegaConf DictConfig
        method_cfg = OmegaConf.create(method_config_dict)
        master_rng_key = jax.random.PRNGKey(random_seed)

        if self.has_simulator:
            logger.info(f"Generating {num_training_simulations} training simulations...")
            training_data = task.get_data(num_samples=num_training_simulations)

            num_validation_simulations = int(num_training_simulations * train_test_fraction)

            validation_data = task.get_data(num_samples=num_validation_simulations)

        if not run:
            if verbose:
                logger.info(f"Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            trained_score_model = train_transformer_model(
                task=task,
                data=training_data,
                method_cfg=method_cfg,
                rng=master_rng_key,
            )
            if verbose:
                logger.info(f"Finished training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            if set_self:
                self.simformer_task = task
                self.posteriors = trained_score_model

        if verbose:
            logger.info(f"Saving model at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        self.save_model_to_pkl(
            task=task,
            posteriors=trained_score_model,
            output_folder=f"{code_path}/models/{self.name}/",
            name_append=name_append,
            method_config_dict=method_config_dict,
            save_method=save_method,
            extras={
                "model_config_dict": model_config_dict,
                "sde_config_dict": sde_config_dict,
                "train_config_dict": train_config_dict,
                "random_seed": random_seed,
                "num_training_simulations": num_training_simulations,
                "train_test_fraction": train_test_fraction,
                "attention_mask_type": attention_mask_type,
                "has_simulator": self.has_simulator,
                "learning_type": learning_type,
            },
        )

        # Evaluate model.
        test_x = validation_data["x"]
        theta_val = validation_data["theta"]

        if set_self:
            self._X_test = np.array(test_x)
            self._y_test = np.array(theta_val)

        if evaluate_model:
            if verbose:
                logger.info(f"Evaluating model at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            self.plot_diagnostics(
                task=task,
                X_test=test_x,
                y_test=theta_val,
                posteriors=trained_score_model,
                num_samples=1000,
                num_evaluations=25,
                rng_seed=random_seed,
                plots_dir=f"{code_path}/models/{self.name}/plots/{name_append}/",
                metric_path=f"{code_path}/models/{self.name}/{self.name}_{name_append}_metrics.json",
            )

    def load_model_from_pkl(
        self, model_dir: str, model_name: str = "simformer", set_self: bool = True
    ):
        """Load a Simformer model from a pickle file.

        Parameters:
        -----------
        model_dir : str
            Directory where the model file is located.
        model_name : str, optional
            Name of the model file to load. Default is "simformer".
        set_self : bool, optional
            If True, sets the loaded model and task to the instance attributes.
            Default is True.
        """
        from .simformer import load_full_model

        if not model_name.endswith("_posterior"):
            model_name = f"{model_name}_posterior"

        model, meta, task = load_full_model(
            model_dir,
            model_name,
        )

        if set_self:
            self.simformer_task = task
            self.posteriors = model

            for item in meta:
                val = meta[item]
                if isinstance(val, list):
                    try:
                        if val[0] not in [str, np.str_]:
                            val = np.array(val)
                    except Exception:
                        pass
                if isinstance(val, np.ndarray):
                    try:
                        if np.ndim(val) < 2:
                            val = list(val)
                    except Exception:
                        pass

                setattr(self, item, val)

            model.has_features = True

        return model, meta

    def save_model_to_pkl(
        self,
        task=None,
        posteriors=None,
        name_append="",
        out_dir: str = f"{code_path}/models/name/",
        save_method="joblib",
        **extras,
    ):
        """Save the Simformer model to a pickle file.

        Parameters:
        *************
        task : object, optional
            Task object containing the model and data.
            If None, uses the simformer_task attribute of the object.
        posteriors : object, optional
            Posteriors to save. If None, uses the posteriors attribute of the object.
        name_append : str, optional
            String to append to the model name in the output file.
            Default is an empty string.
        out_dir : str, optional
            Directory to save the model files.
            Default is f"{code_path}/models/name/".
        save_method : str, optional
            Method to use for saving the model.
            Options are 'torch', 'joblib', 'pickle', and 'hdf5'.
            Default is 'torch'.
        extras : dict, optional
            Additional parameters to save with the model.
            These will be added to the saved dictionary.
        """
        if task is None:
            task = self.simformer_task
        if posteriors is None:
            posteriors = self.posteriors

        if task is None or posteriors is None:
            raise ValueError("Task and posteriors must be provided.")

        out_dir = out_dir.replace("name", self.name)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if len(name_append) > 0 and name_append[0] != "_":
            name_append = f"_{name_append}"

        file_name = os.path.join(out_dir, f"{self.name}{name_append}_posterior.pkl")

        if save_method == "torch":
            from torch import save

            # Save trained model using PyTorch
            save(posteriors, file_name)
        elif save_method == "joblib":
            from joblib import dump

            # Save trained model using joblib
            dump(posteriors, file_name, compress=3)
        elif save_method == "pickle":
            import pickle

            # Save trained model using pickle
            with open(file_name, "wb") as f:
                pickle.dump(posteriors, f)
        elif save_method == "hdf5":
            raise NotImplementedError("HDF5 saving is not implemented yet.")
        elif save_method == "dill":
            import dill

            with open(file_name, "wb") as f:
                dill.dump(posteriors, f)
        else:
            raise ValueError(
                f"Invalid save_method: {save_method}. "
                "Choose from 'torch', 'joblib', 'pickle', 'dill', or 'hdf5'."
            )

        from scoresbibm.methods.score_transformer import get_z_score_fn

        # Why is this here?
        if (
            posteriors.z_score_params is not None
            and posteriors.z_score_params["z_score_fn"] is None
        ):
            zscore, un_zscore = get_z_score_fn(
                posteriors.z_score_params["mean_per_node_id"],
                posteriors.z_score_params["std_per_node_id"],
            )
            posteriors.z_score_params["z_score_fn"] = zscore
            posteriors.z_score_params["un_z_score_fn"] = un_zscore

        save_dict = {
            "_x_dim": task.get_x_dim(),
            "_theta_dim": task.get_theta_dim(),
            "prior_dict": task.prior_dist.prior_ranges,
            "param_names_ordered": task.param_names_ordered,
            "backend": task.backend,
        }

        save_dict.update(extras)

        # Move the posteriors to CPU and convert to numpy if needed. Saves problems recreating
        # the arrays later
        save_dict = make_serializable(save_dict, allowed_types=[np.ndarray])

        self.save_state(
            out_dir=out_dir,
            name_append=name_append,
            save_method=save_method,
            has_grid=~self.has_simulator,
            **save_dict,
        )

    def plot_diagnostics(
        self,
        X_test=None,
        y_test=None,
        num_samples=1000,
        num_evaluations=25,
        task=None,
        posteriors=None,
        rng_seed: int = 42,
        plots_dir: str = f"{code_path}/models/name/plots/",
        metric_path: str = f"{code_path}/models/name/name_metrics.json",
        overwrite: bool = False,
    ):
        """Plot diagnostics for the Simformer model.

        Args:
            X_test (np.ndarray, optional): Test data to evaluate the model.
                If None, uses the `X_test` attribute. Defaults to None.
            y_test (np.ndarray, optional): True values for the test data.
                If None, uses the `y_test` attribute. Defaults to None.
            num_samples (int, optional): Number of samples to draw from the
                posterior. Defaults to 1000.
            num_evaluations (int, optional): Number of evaluations to perform
                for coverage. Defaults to 25.
            task (object, optional): Task object containing the model and data.
                If None, uses the `simformer_task` attribute. Defaults to None.
            posteriors (object, optional): Posteriors to use for sampling.
                If None, uses the posteriors stored in the object.
                Defaults to None.
            rng_seed (int, optional): Random seed for reproducibility.
                Defaults to 42.
            plots_dir (str, optional): Directory to save the plots.
                Defaults to f"{code_path}/models/{name}/plots/".
            metric_path (str, optional): Path to save the metrics JSON file.
                Defaults to f"{code_path}/models/{name}/{name}_metrics.json".
            overwrite (bool, optional): If True, overwrites existing plots
                and metrics. Defaults to False.

        Returns:
            None:
                This function saves plots and metrics to disk and does not
                return anything.
        """
        if task is None:
            task = self.simformer_task

        if posteriors is None:
            posteriors = self.posteriors

        """
        eval_inference_task(
            task=task,
            model=posteriors,
            metric_fn=c2st,  # Use the c2st metric function
            metric_params={"condition_mask_fn": "posterior"},
            rng=master_rng_key,
            num_samples=num_samples,
            num_evaluations=num_evaluations,
        )
        """
        samples = self.sample_posterior(
            X_test=X_test,
            num_samples=num_samples,
            posteriors=posteriors,
            rng_seed=rng_seed,
        )

        metrics = self.evaluate_model(
            posteriors=posteriors,
            X_test=X_test,
            y_test=y_test,
            samples=samples,
        )
        metrics_path = metric_path.replace("name", self.name)
        if not os.path.exists(os.path.dirname(metrics_path)):
            os.makedirs(os.path.dirname(metrics_path))

        try:
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving metrics to {metrics_path}: {e}")

        self.plot_sample_accuracy(
            num_samples=num_samples,
            X_test=X_test,
            y_test=y_test,
            task=task,
            posteriors=posteriors,
            rng_seed=rng_seed,
            plots_dir=plots_dir,
            samples=samples,
        )

        self.plot_coverage(
            num_samples=num_samples,
            num_evaluations=num_evaluations,
            task=task,
            posteriors=posteriors,
            rng_seed=rng_seed,
            plots_dir=plots_dir,
            overwrite=overwrite,
        )

    def plot_sample_accuracy(
        self,
        num_samples=1000,
        X_test=None,
        y_test=None,
        task=None,
        posteriors=None,
        rng_seed: int = 42,
        plots_dir: str = f"{code_path}/models/name/plots/",
        samples=None,
    ):
        """Plot the accuracy of the sampled posterior distribution.

        Parameters:
        ------------
        num_samples : int, optional
            Number of samples to draw from the posterior. Default is 1000.
        X_test : np.ndarray, optional
            Test data to sample from the posterior. If None, uses the
            X_test attribute of the object.
        y_test : np.ndarray, optional
            True values for the test data. If None, uses the y_test
            attribute of the object.
        task : object, optional
            Task object containing the model and data. If None, uses the
            simformer_task attribute of the object.
        posteriors : object, optional
            Posteriors to use for sampling. If None, uses the posteriors
            stored in the object.
        rng_seed : int, optional
            Random seed for reproducibility. Default is 42.
        plots_dir : str, optional
            Directory to save the plots. Default is
            f"{code_path}/models/{name}/plots/".
        samples : np.ndarray, optional
            Pre-computed samples from the posterior. If None, samples will be
            drawn from the posterior using the sample_posterior method.
        """
        if task is None:
            task = self.simformer_task

        if posteriors is None:
            posteriors = self.posteriors

        posterior_condition_mask = jnp.array(
            [0] * task.get_theta_dim() + [1] * task.get_x_dim(), dtype=jnp.bool_
        )

        if samples is None:
            y_test_recovered = self.sample_posterior(
                X_test=X_test,
                num_samples=1000,
                posteriors=posteriors,
                rng_seed=rng_seed,
                attention_mask=posterior_condition_mask,
            )
        else:
            y_test_recovered = samples

        # Get 16, 50 and 84 percentiles for each parameter for each test sample

        fig, ax = plt.subplots(
            nrows=1,
            ncols=len(task.param_names_ordered),
            figsize=(len(task.param_names_ordered) * 3, 3),
        )

        for i, param in enumerate(task.param_names_ordered):
            # Get the 16th, 50th and 84th percentiles for the parameter
            p16, p50, p84 = np.percentile(
                y_test_recovered[:, :, i],
                [16, 50, 84],
                axis=1,
            ).squeeze()
            ax[i].errorbar(
                y_test[:, i],  # True values
                p50,
                yerr=[p50 - p16, p84 - p50],
                fmt="o",
                capsize=0,
                markersize=1,
                elinewidth=0.5,
                alpha=0.75,
            )
            # Add a 1:1 line
            ax[i].plot(
                [y_test[:, i].min(), y_test[:, i].max()],
                [y_test[:, i].min(), y_test[:, i].max()],
                "k--",
            )

            ax[i].set_title(param)
            ax[i].set_xlabel("True")
            if i == 0:
                ax[i].set_ylabel("Predicted")

        plt.tight_layout()

        plots_dir = plots_dir.replace("name", self.name)

        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        # TODO: Rename to match convention
        plt.savefig(os.path.join(plots_dir, "plot_sample_predictions.jpg"))

    def plot_coverage(
        self,
        num_samples=1000,
        num_evaluations=25,
        task=None,
        posteriors=None,
        rng_seed: int = 42,
        plots_dir: str = f"{code_path}/models/name/plots/",
    ):
        """Plot the coverage of the posterior distribution.

        Parameters:
        ------------
        num_samples : int, optional
            Number of samples to draw from the posterior. Default is 1000.
        num_evaluations : int, optional
            Number of evaluations to perform for coverage. Default is 25.
        task : object, optional
            Task object containing the model and data. If None, uses the
            simformer_task attribute of the object.
        posteriors : object, optional
            Posteriors to use for sampling. If None, uses the posteriors
            stored in the object.
        rng_seed : int, optional
            Random seed for reproducibility. Default is 42.
        plots_dir : str, optional
            Directory to save the plots. Default is
            f"{code_path}/models/{name}/plots/".
        """
        master_rng_key = jax.random.PRNGKey(rng_seed)

        from scoresbibm.evaluation.eval_task import eval_coverage

        metric_values, eval_time = eval_coverage(
            task=task,
            model=posteriors,
            metric_params={
                "num_samples": num_samples,
                "num_evaluations": num_evaluations,
                "condition_mask_fn": "posterior",  # posterior, joint, likelihood,
                # random or structured random
                "num_bins": 20,  # Number of bins for histogram
                "sample_kwargs": {},
                "log_prob_kwargs": {},
                "batch_size": 64,  # Batch size for sampling
            },
            rng=master_rng_key,
        )

        plt.plot(metric_values[0], metric_values[1], marker="o", label="Coverage")
        plt.plot([0, 1], [0, 1], "k--", label="Ideal Coverage")
        plt.xlabel("Predicted Percentile")
        plt.ylabel("Empirical Percentile")
        plt.legend()

        plt.title(f"Coverage Plot (num_samples={num_samples}, num_evaluations={num_evaluations})")

        plots_dir = plots_dir.replace("name", self.name)

        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        plt.savefig(os.path.join(plots_dir, f"coverage_plot_{self._timestamp}.png"))

    def plot_posterior(self):
        """Plot the posterior distribution."""
        pass

    def log_prob(self, X_test, condition_mask="full", posteriors=None, theta=None):
        """Compute the log probability of the data given the model.

        Parameters
        ----------
        X_test : np.ndarray
            Observed data of shape (n_observations, n_features) or (n_features,)
        condition_mask : np.ndarray or str
            Mask indicating which parts of the data are observed.
            If 'full', assumes all features are observed.
        posteriors : object, optional
            Posteriors to use for computing the log probability. If None,
            will use the posteriors stored in the object.
        theta : np.ndarray, optional
            Parameter samples of shape (n_samples, n_params). If None,
            will sample from posterior.

        Returns:
        -------
        np.ndarray
            Log probabilities of shape (n_observations, n_samples) where each
            element [i,j] is the log probability of observation i under
            posterior sample j.
        """
        if posteriors is None:
            posteriors = self.posteriors

        num_theta = len(self.fitted_parameter_names)
        num_x = len(self.feature_names)

        if condition_mask == "full":
            condition_mask = jnp.array([0] * num_theta + [1] * num_x, dtype=jnp.bool_)
        else:
            condition_mask = jnp.array(condition_mask, dtype=jnp.bool_)

        X_test = np.atleast_2d(X_test)
        n_observations = X_test.shape[0]

        # Get posterior samples if not provided
        if theta is None:
            theta = self.sample_posterior(
                X_test=X_test,
                posteriors=posteriors,
                condition_mask=condition_mask,
            )

        # Ensure theta is 2D: (n_samples, n_params)
        theta = np.atleast_2d(theta)
        n_samples = theta.shape[0]

        # Initialize result array
        log_probs = np.zeros((n_observations, n_samples))

        # Compute log probability for each observation and each posterior sample
        for i, x_obs in enumerate(X_test):
            x_o = jnp.array(x_obs, dtype=jnp.float32)

            for j in range(n_samples):
                theta_sample = jnp.array(theta[j], dtype=jnp.float32)

                log_prob = posteriors.log_prob(
                    theta=theta_sample, x_o=x_o, condition_mask=condition_mask
                )
                log_probs[i, j] = float(log_prob)

        # Return appropriate shape based on input
        if n_observations == 1 and n_samples == 1:
            return log_probs[0, 0]  # Single scalar
        elif n_observations == 1:
            return log_probs[0, :]  # 1D array of samples for single observation
        elif n_samples == 1:
            return log_probs[:, 0]  # 1D array of observations for single sample
        else:
            return log_probs  # 2D array

    def sample_posterior(
        self,
        X_test,
        num_samples: int = 1000,
        posteriors: object = None,
        rng_seed: int = 42,
        attention_mask: Union[str, np.ndarray] = "full",
        batch_size: int = 100,
        **kwargs,
    ):
        """Sample from the posterior distribution.

        Parameters
        ----------

        X_test : np.ndarray
            Test data to sample from the posterior.

        num_samples : int, optional
            Number of samples to draw from the posterior. Default is 1000.
        posteriors : object, optional
            Posteriors to use for sampling. If None, will use the posteriors
            stored in the object.
        attention_mask : Union[str, np.ndarray], optional
            Attention mask to use for sampling. Can be 'full' or a numpy array.
            Default is 'full'. Full means you have full observations for all bands.
            If you have missing bands, you can provide a numpy array with
            the shape

        TODO: Make this work for multidimensional X_test.

        """
        if posteriors is None:
            posteriors = self.posteriors
        master_rng_key = jax.random.PRNGKey(rng_seed)

        num_theta = len(self.fitted_parameter_names)
        num_x = len(self.feature_names)

        X_test = np.atleast_2d(X_test)

        assert X_test.shape[1] == num_x or attention_mask == "full", (
            "Must provide all features or a manual attention mask. "
        )

        if attention_mask == "full":
            mask = jnp.array([0] * num_theta + [1] * num_x, dtype=jnp.bool_)
        else:
            mask = attention_mask.astype(np.bool_)

        all_samples = []

        for x in tqdm(X_test, desc="Sampling from posterior"):
            samples = posteriors.sample_batched(
                num_samples=num_samples,
                x_o=x,
                condition_mask=mask,
                rng=master_rng_key,
            )

            samples = np.array(samples[0], dtype=np.float32)
            all_samples.append(samples)
        """ Batched sampling is slow.
        nbatches = int(np.ceil(X_test.shape[0] / batch_size))
        for batch_idx in trange(nbatches, desc="Sampling from posterior"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, X_test.shape[0])
            x_batch = jnp.array(X_test[start_idx:end_idx], dtype=jnp.float32)

            # Sample from the posterior for the batch
            samples_batch = posteriors.sample_batched(
                num_samples=num_samples,
                x_o=x_batch,
                condition_mask=mask,
                rng=master_rng_key,
            )

            # Convert to numpy and append to the list
            all_samples.extend(np.array(samples_batch, dtype=np.float32))
        """

        all_samples = np.array(all_samples, dtype=np.float32)

        # if X_test is 1-dimensional, flatten the output
        if X_test.shape[0] == 1:
            all_samples = all_samples[0]

        return all_samples

        """theta_samples = samples[:, :theta_dim]
        if attention_mask != 'full':
            phot_samples = samples[:, theta_dim:]
            return theta_samples, phot_samples
        else:
            return theta_samples"""

    def create_priors(
        self, override_prior_ranges: dict = {}, verbose: bool = True, set_self: bool = False
    ):
        """Create priors for the Simformer model.

        Parameters:
        -----------
        override_prior_ranges : dict, optional
            Dictionary to override the default prior ranges.
            Default is an empty dictionary.
        verbose : bool, optional
            If True, prints information about the prior creation.
            Default is True.
        set_self : bool, optional
            If True, sets the created prior to the instance attribute.

        """
        from .simformer import GalaxyPrior

        priors_sbi = (
            super()
            .create_priors(
                override_prior_ranges=override_prior_ranges,
                verbose=verbose,
            )
            .base_dist
        )

        prior_ranges = {}
        for i, name in enumerate(self.fitted_parameter_names):
            low = float(priors_sbi.low[i])
            high = float(priors_sbi.high[i])
            prior_ranges[name] = (low, high)

        prior = GalaxyPrior(prior_ranges, self.fitted_parameter_names)

        if set_self:
            self._prior = prior

        return prior

    def optimize_sbi(self):
        """Optimize the SBI model."""
        raise NotImplementedError("Simformer_Fitter does not implement optimize_sbi method. ")

    def fit_catalogue(
        self,
        observations: Union[Table, pd.DataFrame],
        columns_to_feature_names: dict = None,
        flux_units: Union[str, unyt_quantity, None] = None,
        missing_data_flag: Any = -99,
        quantiles: list = [0.16, 0.5, 0.84],
        num_samples: int = 1000,
        override_transformations: dict = {},
        append_to_input: bool = True,
        return_feature_array: bool = False,
        recover_SEDs: bool = False,
        plot_SEDs: bool = False,
        check_out_of_distribution: bool = True,
        simulator: Optional[GalaxySimulator] = None,
        rng_seed: int = 42,
        attention_mask: Union[str, np.ndarray] = "full",
        batch_size: int = 100,
        missing_data_mcmc: bool = False,
        log_times=False,
    ):
        """Wrapper for fit_catalogue in parent.

        To Do: Better attention mask.

        """
        sample_method: str = "direct"
        sample_kwargs: dict = {}
        timeout_seconds_per_row: int = 5

        return super().fit_catalogue(
            observations=observations,
            columns_to_feature_names=columns_to_feature_names,
            flux_units=flux_units,
            missing_data_flag=missing_data_flag,
            quantiles=quantiles,
            num_samples=num_samples,
            override_transformations=override_transformations,
            append_to_input=append_to_input,
            return_feature_array=return_feature_array,
            recover_SEDs=recover_SEDs,
            plot_SEDs=plot_SEDs,
            check_out_of_distribution=check_out_of_distribution,
            simulator=simulator,
            sample_method=sample_method,
            sample_kwargs=sample_kwargs,
            timeout_seconds_per_row=timeout_seconds_per_row,
            rng_seed=rng_seed,
            attention_mask=attention_mask,
            batch_size=batch_size,
            missing_data_mcmc=missing_data_mcmc,
            log_times=log_times,
        )
