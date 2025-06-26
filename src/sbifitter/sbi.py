"""SBI Inference classes for SED fitting."""

import copy
import glob
import json
import os
import queue
import signal
import threading
import time
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple, Union

import ili
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
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
from tqdm import tqdm
from unyt import Jy, nJy, unyt_array, unyt_quantity

# astropy, scipy, matplotlib, tqdm, synthesizer, unyt, h5py, numpy,
# ili, torch, sklearn, optuna, joblib, pandas, tarp, astropy.table
from .grid import CombinedBasis, EmpiricalUncertaintyModel, GalaxySimulator
from .utils import (
    FilterArithmeticParser,
    TimeoutException,
    create_sqlite_db,
    f_jy_err_to_asinh,
    f_jy_to_asinh,
    load_grid_from_hdf5,
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
    raw_photometry_names: list
        The names of the photometry filters in the grid.
    raw_photometry_grid: np.ndarray
        The raw photometry grid to use for fitting.
    parameter_array: np.ndarray
        The parameter array to use for fitting.
    raw_photometry_units: list
        The units of the raw photometry grid.
    simulator: callable
        The simulator function to use for generating synthetic data.
    feature_array: np.ndarray
        The feature array to use for fitting.
    feature_names: list
        The names of the features to use for fitting.
    feature_units: list
        The units of the features to use for fitting.
    grid_path: str
        The path to the grid file.
    supplementary_parameters: np.ndarray
        Any supplementary parameters to include in the fitting.
    supplementary_parameter_names: list
        The names of the supplementary parameters.
    supplementary_parameter_units: list
        The units of the supplementary parameters.
    device: str
        The device to use for fitting. Default is 'cuda' if available,
        otherwise 'cpu'.

    """

    device = "cuda" if hasattr(torch, "cuda") and torch.cuda.is_available() else "cpu"

    def __init__(
        self,
        name: str,
        parameter_names: list,
        raw_photometry_names: list,
        raw_photometry_grid: np.ndarray = None,
        parameter_array: np.ndarray = None,
        raw_photometry_units: list = nJy,
        simulator: callable = None,
        feature_array: np.ndarray = None,
        feature_names: list = None,
        feature_units: list = None,
        grid_path: str = None,
        supplementary_parameters: np.ndarray = None,
        supplementary_parameter_names: list = None,
        supplementary_parameter_units: list = None,
        device: str = device,
    ) -> None:
        """Class for SBI Fitting.

        Description:
        name: str
            The name of the model.
        parameter_names: list
            The names of the parameters to fit.
        raw_photometry_names: list
            The names of the photometry filters in the grid.
        raw_photometry_grid: np.ndarray
            The raw photometry grid to use for fitting.
        parameter_array: np.ndarray
            The parameter array to use for fitting.
        raw_photometry_units: list
            The units of the raw photometry grid.
        simulator: callable
            The simulator function to use for generating synthetic data.
        feature_array: np.ndarray
            The feature array to use for fitting.
        feature_names: list
            The names of the features to use for fitting.
        feature_units: list
            The units of the features to use for fitting.
        grid_path: str
            The path to the grid file.
        supplementary_parameters: np.ndarray
            Any supplementary parameters to include in the fitting.
        supplementary_parameter_names: list
            The names of the supplementary parameters.
        supplementary_parameter_units: list
            The units of the supplementary parameters.
        device: str
            The device to use for fitting. Default is 'cuda' if available,
            otherwise 'cpu'.

        """
        self.name = name
        self.raw_photometry_grid = raw_photometry_grid
        self.raw_photometry_units = raw_photometry_units
        self.raw_photometry_names = raw_photometry_names
        self.parameter_array = parameter_array
        self.parameter_names = parameter_names

        self.simulator = simulator
        self.has_simulator = simulator is not None

        assert (self.simulator is not None) or (self.raw_photometry_grid is not None), (
            "Either a simulator or raw photometry grid must be provided."
        )

        # This allows you to subset the parameters to fit
        # if you want to marginalize over some parameters.
        # See self.update_parameter_array() for more details.
        self.fitted_parameter_array = None
        self.fitted_parameter_names = parameter_names
        self.simple_fitted_parameter_names = [i.split("/")[-1] for i in parameter_names]

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
        self.grid_path = grid_path

        self.has_features = (self.feature_array is not None) and (
            self.feature_names is not None
        )

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
            print(
                "CUDA is not available. Falling back to CPU. "
                "Please check your PyTorch installation."
            )
            self.device = "cpu"

    def update_parameter_array(
        self,
        parameters_to_remove: list = [],
        delete_rows=[],
        n_scatters: int = 1,
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
        """
        print(len(delete_rows), "rows to delete from parameter array.")
        self.fitted_parameter_array = copy.deepcopy(self.parameter_array)
        self.fitted_parameter_names = self.parameter_names

        params = np.unique(self.provided_feature_parameters + parameters_to_remove)
        for param in params:
            if param in self.fitted_parameter_names:
                index = list(self.fitted_parameter_names).index(param)
                self.fitted_parameter_array = np.delete(
                    self.fitted_parameter_array, index, axis=1
                )
                self.fitted_parameter_names = np.delete(
                    self.fitted_parameter_names, index
                )
                self.simple_fitted_parameter_names = [
                    i.split("/")[-1] for i in self.fitted_parameter_names
                ]

        if n_scatters > 1:
            self.fitted_parameter_array = np.repeat(
                self.fitted_parameter_array, n_scatters, axis=0
            )

        # Remove any rows in delete_rows
        if len(delete_rows) > 0:
            self.fitted_parameter_array = np.delete(
                self.fitted_parameter_array, delete_rows, axis=0
            )

    @classmethod
    def init_from_hdf5(
        cls,
        model_name: str,
        hdf5_path: str,
        return_output=False,
        **kwargs,
    ):
        """Initialize the SBI fitter from an HDF5 file.

        Parameters:
            hdf5_path: Path to the HDF5 file.
            model_name: Name of the model to be used.
            return_output: If True, returns the output dictionary from the HDF5 file.
            **kwargs: Additional keyword arguments to pass to the SBI_Fitter constructor.

        Returns:
            An instance of the SBI_Fitter class.
        """
        # Needs to load the training data and parameters from HDF5 file.
        # Training data if unnormalized and not setup as correct features yet.

        output = load_grid_from_hdf5(hdf5_path)

        if return_output:
            return output

        raw_photometry_grid = output["photometry"]
        raw_photometry_names = output["filter_codes"]
        parameter_array = output["parameters"].T
        parameter_names = output["parameter_names"]
        raw_photometry_units = output["photometry_units"]

        if "supplementary_parameters" in output:
            supplementary_parameters = output["supplementary_parameters"]
            supplementary_parameter_names = output["supplementary_parameter_names"]
            supplementary_parameter_units = output["supplementary_parameter_units"]

        return cls(
            name=model_name,
            raw_photometry_grid=raw_photometry_grid,
            raw_photometry_names=raw_photometry_names,
            parameter_array=parameter_array,
            parameter_names=parameter_names,
            raw_photometry_units=raw_photometry_units,
            feature_array=None,
            feature_names=None,
            feature_units=None,
            grid_path=hdf5_path,
            supplementary_parameters=supplementary_parameters,
            supplementary_parameter_names=supplementary_parameter_names,
            supplementary_parameter_units=supplementary_parameter_units,
            **kwargs,
        )

    @classmethod
    def init_from_basis(
        cls,
        basis: CombinedBasis,
    ):
        """Initialize the SBI fitter from a basis.

        Args:
            basis: The basis to be used for fitting.

        Returns:
            An instance of the SBI_Fitter class.
        """
        # Needs to load the training data and parameters from the basis
        pass

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
        asinh_softening_parameter : List[unyt_array], optional
            If normed_flux_units is 'asinh', this can be a list of unyt_arrays
            with the same length as the number of filters.

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
            depths_std_expanded = np.repeat(
                depths_std[:, np.newaxis], N_scatters * n, axis=1
            )
        else:
            raise ValueError("depths must be 1D or 2D array")

        if min_flux_pc_error > 0.0:
            print(f"Applying minimum percentage error of {min_flux_pc_error}% to depths.")
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
        asinh_softening_parameters: Union[List[unyt_array], List[float]] = None,
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
            flux_units: The units of the fluxes in the photometry array.
            return_errors: Whether to return the errors as well.
            normed_flux_units: The units of the fluxes after normalization.
            asinh_softening_parameters: If normed_flux_units is 'asinh',
                this can be a list of unyt_arrays with the same length as the number of
                filters. These parameters are used to apply the asinh normalization.


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
                    f"""Filter {filter} in empirical_noise_models is not in phot_names.
                    Please provide a valid filter name."""
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
                noisy_flux, sampled_sigma = noise_model.apply_noise_to_flux(
                    flux,
                    true_flux_units=flux_units,
                    out_units=normed_flux_units,
                )
                # print(filter_name, noisy_flux.max())

                scattered_photometry_i = noisy_flux
                # Store the errors
                errors_i = sampled_sigma
            else:
                print(
                    f"No empirical noise model found for filter {filter_name}. Skipping."
                )

            return scattered_photometry_i, errors_i

        results = [apply_noise_model(filter_name) for filter_name in phot_names]
        # Stack the results correctly

        scattered_photometry = np.stack([result[0] for result in results], axis=0)
        errors = np.stack([result[1] for result in results], axis=0)

        assert np.shape(errors) == np.shape(errors_s)

        if return_errors:
            return scattered_photometry, errors

        return scattered_photometry

    def detect_misspecification(self, x_obs, X_train=None, retrain=False):
        """Tests misspecification of the model using the MarginalTrainer from sbi.

        X_test: The test data to check for misspecification.
        X_train: The training data to use for the misspecification check. If None,
        it will use the training data set in the class.

        This function uses the MarginalTrainer from sbi to train a density estimator on
        the training data, and then calculates the misspecification
        log probability of the test data.

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
        from sbi.diagnostics.misspecification import (
            calc_misspecification_logprob,
        )
        from sbi.inference.trainers.marginal import MarginalTrainer

        if X_train is None:
            if self._X_train is None:
                raise ValueError(
                    "No training data found. Please set the training data first."
                )
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
                raise ValueError(
                    "No training data found. Please set the training data first."
                )
            X_test = self._X_train

        if y_test is None:
            if self._y_train is None:
                raise ValueError(
                    "No training labels found. Please set the training labels first."
                )
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
    ):
        """Create a feature array from the raw photometry grid.

        A simpler wrapper for
        `create_feature_array_from_raw_photometry` with default values.
        This function will create a feature array from the raw photometry grid
        with no noise, and all photometry in mock catalogue used.
        """
        return self.create_feature_array_from_raw_photometry(
            normed_flux_units=flux_units, extra_features=extra_features
        )

    def create_feature_array_from_raw_photometry(
        self,
        normalize_method: str = None,
        extra_features: list = None,
        normed_flux_units: str = "AB",
        normalization_unit: str = "AB",
        verbose: bool = True,
        scatter_fluxes: Union[int, bool] = False,
        empirical_noise_models: Optional[Dict[str, EmpiricalUncertaintyModel]] = None,
        depths: Union[unyt_array, None] = None,
        include_errors_in_feature_array=False,
        min_flux_pc_error: float = 0.0,
        simulate_missing_fluxes=False,
        missing_flux_value=99.0,
        missing_flux_fraction=0.0,
        missing_flux_options: list = None,
        include_flags_in_feature_array=False,
        override_phot_grid: np.ndarray = None,
        override_phot_grid_units: str = None,
        norm_mag_limit: float = 50.0,
        remove_nan_inf: bool = True,
        parameters_to_remove: list = None,
        photometry_to_remove: list = None,
        drop_dropouts: bool = False,
        drop_dropout_fraction: float = 1.0,
        asinh_softening_parameters: Union[
            unyt_array, List[unyt_array], Dict[str, unyt_array], str
        ] = None,
    ) -> np.ndarray:
        """Create a feature array from the raw photometry grid.

        Parameters:
            normalize_method: The method to normalize the photometry.
                At the moment only the names of the filters, names
                of supplementary parameters, or None are accepted.
                This will be used to normalize the fluxes in the feature array.
            extra_features: Any extra features to be added. These should be
                generally written as functions of filter codes.
                E.g. NIRCam.F090W - NIRCam.F115W color as
                a feature would be written as ['F090W - F115W'].
                They can also be parameters in the parameter grid,
                in which case they won't be predicted by the model -e.g. redshift.
                or supplementary parameters (e.g. things generated)
                with grid, e.g. spectral indicices, mUV, morphology, etc.
            normed_flux_units: The units of the flux to normalize to. E.g. 'AB', 'nJy',
                etc. So when combined with normalize method,
                the fluxes for each filter will be relative to the normalization filter,
                in the given units. The overall
                normalization factor will be provided as well.
                Options include:
                    "AB" - AB magnitude normalization.
                    "asinh" - Asinh magnitude normalization.
                    any string or unyt_quantity equivalent to a flux density unit.
            normalization_unit: The unit of the normalization factor, if used.
                E.g. 'log10 nJy', 'nJy', AB', etc. This can be different to
                normed_flux_units, but should be a valid flux density unit. E.g.
                you could provide normed_flux_units="nJy", but given the normalisation
                in magnitudes. Can start with 'log10 unit' to indicate a logarithmic
                normalization, e.g. 'log10 nJy' (not for AB magnitudes as these are
                already logarithmic).
            scatter_fluxes: Whether to scatter fluxes with uncertainity. Either False,
                or an integer for the number of scatters to apply. Noise model used
                is either the empirical noise model, or the depths, depending on the
                supplied parameters.
            empirical_noise_models: A dictionary of empirical noise models to
                use for scattering the fluxes. The keys should be the filter names,
                and the values should be the EmpiricalUncertaintyModel objects.
            depths: Either None, or a unyt_array of length photometry.
            include_errors_in_feature_array: boolean, default False.
                Whether to include the RMS uncertainty in the input model.
                This would pass in the uncertainity as a seperate parameter.
                Could be either 2D, or (value, error) pairs in 1D training dataset.
                Could also allow options to under or
                overestimate model uncertainity in depths.
            min_flux_pc_error: The minimum percentage error to apply to the fluxes.
                This is used to set the minimum error on the fluxes when scattering.
            simulate_missing_fluxes:  boolean, default False.
                This would allow missing photometry for some fraction of the time,
                which could be marked with a specific filler flag,
                or a seperate boolean row/column to flag missing data.
            missing_flux_value: The value to use for missing fluxes. Default is 99.0.
            missing_flux_fraction: The fraction of missing fluxes. Default is 0.0.
            missing_flux_options: If simulate_missing_fluxes is True,
                this is a list of mask for the missing fluxes. E.g.
                rather than randomly masking galaxies, we have a set of possible options
                which are randomly selected from.
            include_flags_in_feature_array: boolean, default False.
            override_phot_grid: The photometry grid to use
                instead of the raw photometry grid.
                This is used to override the photometry grid for testing purposes.
            override_phot_grid_units: The units of the override photometry grid.
                This is used to override the units of the photometry grid
                for testing purposes.
            norm_mag_limit: The maximum magnitude limit for the normalized fluxes.
                This is used to set a maximum difference of some large amount -
                say 50 magnitudes.
            remove_nan_inf: boolean, default True.
                Whether to remove any rows with NaN or Inf values in the feature array.
            parameters_to_remove: List of parameters to remove from the parameter array.
            photometry_to_remove: List of photometry filters to remove from the grid.
                Generally if a filter is listed here, it is removed first, so other
                arguments which expect lists of arguments matching the length of the
                filter array (e.g. depths, empirical_noise_models,
                asinh_softening_parameters) will not include filters listed here.
                Dictionaries matching self.raw_photometry_names to these keys are
                also accepted for more explicit control.
            drop_dropouts: boolean, default False.
                Whether to drop the dropouts from the feature array.
            drop_dropout_fraction: float, default 1.0.
                The fraction of dropouts to drop from the feature array.
                E.g. if a galaxy dropouts out in more than this fraction of the filters,
                it will be dropped from the feature array.
            asinh_softening_parameters: float, list, dict or str, default None.
                The softening parameter for the asinh normalization.
                Only used if normed_flux_units is 'asinh'.
                If a single quantity, it is used for all filters.
                If a list, it should be the same length as the number of (raw) filters.
                If a dict, it should map filter names to the softening parameters.
                Or it can be 'SNR_{level} to set it from the noise model
                or depths.

            TODO: How should normalization work with the scattering?

        Returns:
            The feature array and feature names.
        """
        if extra_features is None:
            extra_features = []

        if parameters_to_remove is None:
            parameters_to_remove = []
        if photometry_to_remove is None:
            photometry_to_remove = []
        if override_phot_grid is not None:
            phot_grid = override_phot_grid
            raw_photometry_units = override_phot_grid_units
        else:
            phot_grid = copy.deepcopy(self.raw_photometry_grid)
            raw_photometry_units = self.raw_photometry_units

        raw_photometry_names = self.raw_photometry_names

        assert isinstance(photometry_to_remove, list), (
            "photometry_to_remove must be a list of filter names to remove."
        )
        if len(photometry_to_remove) > 0:
            # Remove the photometry from the grid
            photometry_to_remove = np.array(photometry_to_remove)
            remove_indices = [
                i
                for i, name in enumerate(self.raw_photometry_names)
                if name in photometry_to_remove
            ]
            if len(remove_indices) > 0:
                print(
                    f"""Removing {len(remove_indices)}
                    photometry filters: {photometry_to_remove}"""
                )
                phot_grid = np.delete(phot_grid, remove_indices, axis=0)
                raw_photometry_names = np.delete(
                    self.raw_photometry_names, remove_indices
                )
            else:
                raise ValueError(
                    f"""No matching photometry filters found in the
                    raw photometry names: {photometry_to_remove}"""
                )

            if len(raw_photometry_names) == 0:
                raise ValueError(
                    "No photometry filters left after removing the specified ones."
                )

        if normed_flux_units == "asinh":
            assert asinh_softening_parameters is not None, (
                "asinh_softening_parameters must be provided for asinh normalization."
            )
            if isinstance(
                asinh_softening_parameters, (list, np.ndarray)
            ) and not isinstance(asinh_softening_parameters, unyt_array):
                assert len(asinh_softening_parameters) == len(raw_photometry_names), (
                    "asinh_softening_parameter must be a list of the same length as "
                    "the number of photometry filters."
                )
                asinh_softening_parameter = [
                    asinh_softening_parameters[i].to(Jy).value
                    for i in raw_photometry_names
                ] * Jy
            elif isinstance(asinh_softening_parameters, unyt_array):
                asinh_softening_parameter = asinh_softening_parameters.to(Jy)
            elif isinstance(asinh_softening_parameters, str):
                assert asinh_softening_parameters.startswith("SNR_"), (
                    "If a string, asinh_softening_parameters must start with 'SNR_'."
                )
                assert (scatter_fluxes) and (
                    depths is not None or empirical_noise_models is not None
                ), """If setting asinh_softening_parameters from noise models,
                    depths or empirical_noise_models must be provided."""

                val = float(asinh_softening_parameters.split("_")[-1])
                asinh_softening_parameter = [val for name in raw_photometry_names]

        phot = unyt_array(phot_grid, units=raw_photometry_units)
        converted = False

        if scatter_fluxes:
            assert depths is not None or empirical_noise_models is not None, (
                "If scattering fluxes, depths or empirical noise models must be provided."
            )
            assert isinstance(phot, unyt_array)

            if depths is not None:
                print(
                    f"""Using depth-based noise models with \
                        {scatter_fluxes} scatters per row."""
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
                        asinh_softening_parameter[i]
                        * self.phot_depths[i].to("Jy").value
                        / 5.0
                        for i in range(len(raw_photometry_names))
                    ]
                    asinh_softening_parameter = unyt_array(
                        asinh_softening_parameter,
                        units=Jy,
                    )

                converted = False
            elif empirical_noise_models is not None:
                print(
                    f"""Using empirical noise models with
                    {scatter_fluxes} scatters per row."""
                )
                self.empirical_noise_models = empirical_noise_models

                phot, phot_errors = self._apply_empirical_noise_models(
                    phot,
                    raw_photometry_names,
                    empirical_noise_models,
                    N_scatters=scatter_fluxes,
                    min_flux_pc_error=min_flux_pc_error,
                    return_errors=True,
                    flux_units=phot.units,
                    normed_flux_units=normed_flux_units,
                    asinh_softening_parameters=asinh_softening_parameter,
                )
                converted = True

        else:
            phot_errors = None

        if normed_flux_units == "AB":
            if phot_errors is not None and not converted:
                phot_errors = (
                    2.5
                    * phot_errors.to("uJy").value
                    / (np.log(10) * phot.to("uJy").value)
                )

                print(phot_errors.max())
            if not converted:
                phot_mag = -2.5 * np.log10(phot.to("uJy").value) + 23.9
                mask = phot.to("uJy").value < 0
            else:
                phot_mag = phot
                mask = np.isnan(phot_mag) | np.isinf(phot_mag)
                print(
                    "number of NaN, Inf values in phot_mag:",
                    np.sum(np.isnan(phot_mag)),
                    np.sum(np.isinf(phot_mag)),
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
                phot = phot.to(normed_flux_units).value
                if phot_errors is not None:
                    phot_errors = phot_errors.to(normed_flux_units).value

            norm_func = np.divide

        delete_rows = []
        # Normalize the photometry grid
        if normalize_method is not None:
            if normalize_method in raw_photometry_names:
                norm_index = list(raw_photometry_names).index(normalize_method)

                normalization_factor = phot[norm_index, :]
                norm_factor_original = phot_grid[norm_index, :]

                # Create a copy of the raw photometry names for consistent reference
                raw_photometry_names = np.array(raw_photometry_names)
                raw_photometry_names = np.delete(raw_photometry_names, norm_index)
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
                                units=raw_photometry_units,
                            )
                            .to("uJy")
                            .value
                        )
                        + 23.9
                    )
                else:
                    normalization_factor_converted = (
                        unyt_array(norm_factor_original, units=raw_photometry_units)
                        .to(normalization_unit_cleaned)
                        .value
                    )

                if log:
                    normalization_factor_converted = np.log10(
                        normalization_factor_converted
                    )
                    normalization_factor_converted[
                        normalization_factor_converted == -np.inf
                    ] = 0.0
                    normalization_factor_converted[
                        normalization_factor_converted == np.inf
                    ] = 0.0

            elif normalize_method in self.supplementary_parameter_names:
                norm_index = list(self.supplementary_parameter_names).index(
                    normalize_method
                )
                norm_unit = self.supplementary_parameter_units[norm_index]
                normalization_factor = self.supplementary_parameters[norm_index, :]
                # count where normalzation is 0

                assert (
                    normalization_factor.shape[0] == phot_grid.shape[1]
                ), """Normalization factor should
                    have the same shape as the photometry grid."""
                assert (
                    norm_unit == raw_photometry_units
                ), """Normalization factor should have the
                        same units as the photometry grid."""

                if normed_flux_units == "AB":
                    normalization_factor_use = (
                        -2.5
                        * np.log10(
                            unyt_array(normalization_factor, units=norm_unit)
                            .to("uJy")
                            .value
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
                    normalization_factor = np.repeat(
                        normalization_factor, scatter_fluxes, axis=0
                    )

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
                            unyt_array(normalization_factor, units=norm_unit)
                            .to("uJy")
                            .value
                        )
                        + 23.9
                    )
                else:
                    normalization_factor_converted = (
                        unyt_array(normalization_factor, units=norm_unit)
                        .to(normalization_unit_cleaned)
                        .value
                    )

                raw_photometry_names = np.array(raw_photometry_names)

            else:
                raise NotImplementedError(
                    """Normalization method not implemented.
                    Please use a filter name for normalization."""
                )
        else:
            normed_photometry = phot
            normalization_factor_converted = np.ones(normed_photometry.shape[1])
            normalization_factor = normalization_factor_converted
            raw_photometry_names = np.array(raw_photometry_names)

            # Convert the photometry to the desired units

        if phot_errors is not None:
            # Will have errors on the end of the photometry array. Add names (unc_*)
            # and units (uncertainty) to the end of the feature names and units.
            error_names = [f"unc_{name}" for name in raw_photometry_names]
            error_units = [normed_flux_units] * len(error_names)
        else:
            error_names = []
            error_units = []

        if np.sum(normalization_factor == 0.0) > 0:
            # delete these indexes from the photometry
            print(
                f"""Warning: Normalization factor is 0.0 for
                {np.sum(normalization_factor == 0.0)} rows. These will be deleted."""
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
            len(raw_photometry_names)
            + len(extra_features)
            + norm
            + include_errors_in_feature_array * len(error_names)
            + include_flags_in_feature_array * len(raw_photometry_names)
        )
        # Create the feature array
        # Photometry + extra features + normalization factor
        feature_array = np.zeros((size, length))

        assert np.shape(feature_array[: len(raw_photometry_names), :]) == np.shape(
            normed_photometry
        ), f"""Shape mismatch: {np.shape(feature_array[: len(raw_photometry_names), :])}
            != {np.shape(normed_photometry)}"""
        # Fill the feature array with the normalized photometry
        feature_array[: len(raw_photometry_names), :] = normed_photometry

        if phot_errors is not None and include_errors_in_feature_array:
            # Add the errors to the feature array
            # Work out the starting index given total length of feature array
            start_index = len(raw_photometry_names)
            feature_array[start_index : start_index + len(error_names), :] = phot_errors

        flag_units = []
        flag_names = []

        if simulate_missing_fluxes:
            start_index = len(raw_photometry_names) + len(error_names)
            if missing_flux_options is not None:
                # For each row, pick a mask randomly from the missing_flux_options
                mask = np.zeros((len(raw_photometry_names), feature_array.shape[1]))
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
                    size=(len(raw_photometry_names), feature_array.shape[1]),
                    p=[1 - missing_flux_fraction, missing_flux_fraction],
                )
            if include_flags_in_feature_array:
                flag_units = [None] * len(raw_photometry_names)
                flag_names = [f"flag_{name}" for name in raw_photometry_names]
                feature_array[
                    start_index : start_index + len(raw_photometry_names), :
                ] = mask
            # Set the missing fluxes to the missing_flux_value
            feature_array[: len(raw_photometry_names), :][mask == 1.0] = (
                missing_flux_value
            )

            if len(error_names) > 0:
                feature_array[len(raw_photometry_names) : start_index, :][mask == 1.0] = (
                    missing_flux_value
                )

        if normalize_method is not None:
            # Add the normalization factor as the last column
            feature_array[-1, :] = normalization_factor_converted

        # Create the feature names
        nfeatures = feature_array.shape[0]
        feature_names = [""] * nfeatures

        # Add filter names
        for i in range(len(raw_photometry_names)):
            feature_names[i] = raw_photometry_names[i]

        if phot_errors is not None and include_errors_in_feature_array:
            # Add the error names
            for i in range(len(error_names)):
                feature_names[len(raw_photometry_names) + i] = error_names[i]

        if include_flags_in_feature_array:
            # Add the flag names
            for i in range(len(flag_names)):
                feature_names[len(raw_photometry_names) + len(error_names) + i] = (
                    flag_names[i]
                )

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
                    names = [i.split(".")[-1] for i in raw_photometry_names]
                    print(names, tokens, "token", feature)
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
                print(
                    f"""Warning: Deleting {num} rows with NaN or Inf
                    values in the feature array."""
                )

        if drop_dropouts:
            # Find all galaxies where more than a drop_fraction of
            # bands are at norm_mag_limit
            dropout_mask = (
                np.sum(
                    np.abs(feature_array[: len(raw_photometry_names), :])
                    >= norm_mag_limit,
                    axis=0,
                )
                >= len(raw_photometry_names) * drop_dropout_fraction
            )
            dropout_rows = np.where(dropout_mask)[0]
            if len(dropout_rows) > 0:
                print(
                    f"""Warning: Dropping {len(dropout_rows)} dropouts where more than
                    {drop_dropout_fraction * 100}% of bands are at the norm_mag_limit."""
                )
                delete_rows.extend(dropout_rows.tolist())

        # Remove any rows in delete_rows
        if len(delete_rows) > 0:
            feature_array = np.delete(feature_array, delete_rows, axis=1)

        # check if all rows got deleted
        if feature_array.shape[1] == 0:
            raise ValueError(
                "All rows in the feature array were deleted. "
                "Please check the input parameters."
            )

        assert "" not in feature_names, (
            "Feature names should not be empty. Please check the extra features."
        )

        self.feature_array = feature_array.astype(np.float32).T
        self.feature_names = feature_names
        self.feature_units = (
            [normed_flux_units] * len(raw_photometry_names)
            + error_units
            + flag_units
            + [None] * len(extra_features)
            + [normalization_unit]
        )
        self.has_features = True

        if verbose:
            print("---------------------------------------------")
            print(
                f"""Features: {self.feature_array.shape[1]} features over \
{self.feature_array.shape[0]} samples"""
            )
            print("---------------------------------------------")
            print("Feature: Min - Max")
            print("---------------------------------------------")

            for pos, feature_name in enumerate(feature_names):
                print(
                    f"""{feature_name}: {np.min(feature_array[pos]):.6f} - \
{np.max(feature_array[pos]):.3f} {self.feature_units[pos]}"""
                )
            print("---------------------------------------------")

        # Save all method inputs on self

        self.feature_array_flags = {
            "normalize_method": normalize_method,
            "extra_features": extra_features,
            "normed_flux_units": normed_flux_units,
            "normalization_unit": normalization_unit,
            "scatter_fluxes": scatter_fluxes,
            # "empirical_noise_models": empirical_noise_models,
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
            "drop_dropouts": drop_dropouts,
            "drop_dropout_fraction": drop_dropout_fraction,
            "raw_photometry_names": raw_photometry_names,
            "error_names": error_names,
            "flag_names": flag_names,
            "norm_name": norm_name,
            "asinh_softening_parameters": asinh_softening_parameters,
        }

        self.update_parameter_array(
            delete_rows=delete_rows,
            n_scatters=scatter_fluxes,
            parameters_to_remove=parameters_to_remove,
        )

        return feature_array, feature_names

    def create_features_from_observations(
        self,
        observations: Union[Table, pd.DataFrame],
        columns_to_feature_names: dict = None,
        flux_units: Union[str, unyt_quantity, None] = None,
        missing_data_flag: Any = -99,
        override_transformations: dict = {},
    ):
        """Create a feature array from observational data.

        Transformations applied to an existing feature array are
        saved in the self.feature_array_flags dictionary, but can be overridden
        with override_transformations. Available transformations are the arguments
        to the create_feature_array_from_raw_photometry method, although some,
        like scattering fluxes or error modelling, are not applicable here and
        will have no effect.

        Parameters
        ----------

        observations : Union[np.ndarray, Table, pd.DataFrame]
            The observational data to create the feature array from.
        flux_units : Union[str, unyt_quantity, None]
            Required if the observations do not match 'normed_flux_units'
            in self.feature_array_flags. If None, the units will be taken from the column
            metadata if available and otherwise assumed to be the same as the
            training data.
        columns_to_feature_names : dict
            A dictionary mapping the column names in the observations to feature names.
            If None, then assumed there will be a direct mapping between the columns
            in the observations and feature names in
            self.feature_array_flags['raw_photometry_names'].
        missing_data_flag : Any
            Value in columns delineating missing data. Depending on setup of features,
            this may be flagged to model or the galaxy will be ignored entirely.
        override_transformations : dict
            A dictionary of transformations to override the defaults in
            self.feature_array_flags. This can be used to change the normalization method,
             extra features, etc.

        """
        if len(self.feature_array_flags) == 0:
            raise ValueError(
                "No feature array flags found. Please create the feature array first."
            )

        if not getattr(self, "has_features", False):
            raise RuntimeError(
                "The feature creation pipeline has not been initialized. "
                "Please run `create_feature_array_from_raw_photometry` first."
            )

        feature_array_flags = copy.deepcopy(self.feature_array_flags)

        feature_array_flags.update(override_transformations)

        # Check if the observations are a Table or DataFrame
        if isinstance(observations, Table):
            observations = observations.to_pandas()

        elif not isinstance(observations, pd.DataFrame):
            raise TypeError(
                "Observations must be a pandas DataFrame or an astropy Table."
            )

        # Check

        if columns_to_feature_names is None:
            # Assume a direct mapping between the
            # columns in the observations and feature names
            columns_to_feature_names = {col: col for col in observations.columns}

        feature_names_to_columns = {v: k for k, v in columns_to_feature_names.items()}

        # Validate inputs

        if not self.feature_array_flags["simulate_missing_fluxes"]:
            # Should have a column for every photometry filter in the training data

            # Check all names are keys in columns_to_feature_names
            for name in feature_array_flags["raw_photometry_names"]:
                if name not in feature_names_to_columns:
                    raise ValueError(
                        f"""Column '{name}' not found in observations.
                        Please provide a mapping for all photometry filters."""
                    )

            if feature_array_flags["include_errors_in_feature_array"]:
                # Check all errors are keys in columns_to_feature_names
                for name in feature_array_flags["error_names"]:
                    if name not in feature_names_to_columns:
                        raise ValueError(
                            f"""Column '{name}' not found in observations.
                            Please provide a mapping for all errors."""
                        )

        if feature_array_flags["include_flags_in_feature_array"]:
            # Check all flags are keys in columns_to_feature_names
            for name in feature_array_flags["flag_names"]:
                if name not in feature_names_to_columns:
                    raise ValueError(
                        f"""Column '{name}' not found in observations.
                        Please provide a mapping for all flags."""
                    )

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
            feature_names_to_columns[name]
            for name in feature_array_flags["raw_photometry_names"]
        ]

        # Check if the flux units match the training data
        training_flux_units = feature_array_flags["normed_flux_units"]
        # if str, should match flux units.
        if isinstance(training_flux_units, str):
            assert (
                flux_units == training_flux_units
            ), f"""Flux units '{flux_units}' do not match
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
                    f"""Column '{col}' not found in observations.
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
                feature_array[index, :] = method(
                    feature_array[index, :], normalization_factor
                )

        removed_data = np.zeros(len(feature_array[0]), dtype=bool)
        # Replace NaN and Inf values with the missing_flux_value if applicable
        if feature_array_flags["remove_nan_inf"]:
            # Replace NaN and Inf values with the missing_flux_value
            mask = ~np.isfinite(feature_array)
            print(f"Removing {np.sum(mask)} NaN or Inf values from the feature array.")
            feature_array = feature_array[
                :, ~mask.any(axis=0)
            ]  # Remove columns with NaN or Inf values
            removed_data[mask.any(axis=0)] = True

        missing_mask = feature_array == missing_data_flag
        if feature_array_flags["simulate_missing_fluxes"]:
            print(
                f"""Replacing {np.sum(missing_mask)} NaN or Inf values with
                {feature_array_flags["missing_flux_value"]}."""
            )
            feature_array[missing_mask] = feature_array_flags["missing_flux_value"]
        else:
            print(f"Removing {np.sum(missing_mask)} observations with missing data.")
            feature_array = feature_array[
                :, ~missing_mask.any(axis=0)
            ]  # Remove columns with missing data
            removed_data[missing_mask.any(axis=0)] = True

        mask = feature_array > feature_array_flags["norm_mag_limit"]
        feature_array[mask] = feature_array_flags["norm_mag_limit"]

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
    ):
        """Infer posteriors for observational data.

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

        Returns:
        -------
        observations : Union[Table, pd.DataFrame]
            The original observations with additional columns for the quantiles of the
            fitted parameters.
        """
        feature_array, obs_mask = self.create_features_from_observations(
            observations,
            columns_to_feature_names=columns_to_feature_names,
            flux_units=flux_units,
            missing_data_flag=missing_data_flag,
            override_transformations=override_transformations,
        )

        samples = self.sample_posterior(
            X_test=feature_array,
            sample_method=sample_method,
            sample_kwargs=sample_kwargs,
            num_samples=num_samples,
            timeout_seconds_per_test=timeout_seconds_per_row,
        )

        print("Obtained posterior samples.")
        # Do quantiles, get column names from self.simple_fitted_parameter_names
        print(samples.shape)
        for i, param in enumerate(self.simple_fitted_parameter_names):
            samples_i = samples[:, i]
            samples_q = np.quantile(samples_i, quantiles, axis=1)
            for j, quant in enumerate(samples_q):
                print(param, quant)
                observations[f"{param}_{int(quantiles[j] * 100)}"] = np.nan
                observations[f"{param}_{int(quantiles[j] * 100)}"][~obs_mask] = quant

        return observations

    def create_dataframe(self, data="all"):
        """Create a DataFrame from the training data.

        Parameters:
        -----------
        data : str
            'all' to include all data, 'photometry' for only photometry,
              'parameters' for only parameters

        Returns:
        --------
        pd.DataFrame
            DataFrame with the requested data
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
            return pd.DataFrame(
                copy.deepcopy(self.feature_array), columns=self.feature_names
            )
        else:
            raise ValueError(
                "Invalid data type. Use 'all', 'photometry', or 'parameters'."
            )

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
            raise ValueError(
                "Feature array not created. Please create the feature array first."
            )

        num_samples = self.feature_array.shape[0]

        if verbose:
            print(
                f"""Splitting dataset with {num_samples} samples into training
                and testing sets with {train_fraction:.2f} train fraction."""
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
        prior=ili.utils.Uniform,
        verbose: bool = True,
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

        Returns:
            A prior object.

        """
        if not self.has_features and not self.has_simulator:
            raise ValueError(
                """Feature array not created and no simulator.
                Please create the feature array first."""
            )

        if self.fitted_parameter_array is None and not self.has_simulator:
            raise ValueError(
                "Parameter grid not created. Please create the parameter grid first."
            )
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
                low.append(override_prior_ranges[param][0])
                high.append(override_prior_ranges[param][1])
            else:
                low.append(np.min(self.fitted_parameter_array[:, i]))
                high.append(np.max(self.fitted_parameter_array[:, i]))

        low = np.array(low)
        high = np.array(high)

        # Print prior ranges
        if verbose:
            print("---------------------------------------------")
            print("Prior ranges:")
            print("---------------------------------------------")
            for i, param in enumerate(self.fitted_parameter_names):
                print(f"{param}: {low[i]:.2f} - {high[i]:.2f}")
            print("---------------------------------------------")

        low = torch.tensor(low, dtype=torch.float32, device=self.device)
        high = torch.tensor(high, dtype=torch.float32, device=self.device)

        # Create the priors
        param_prior = prior(low=low, high=high, device=self.device)

        return param_prior

    def create_restricted_priors(
        self,
        prior: Optional[ili.utils.Uniform] = None,
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
        random_seed: int = None,
        verbose: bool = False,
        persistent_storage: bool = False,
        out_dir: str = f"{code_path}/models/",
        score_metrics: Union[str, List[str]] = "log_prob-pit",
        direction: str = "maximize",
        timeout_minutes_trial_sampling: float = 15.0,
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
        - persistent_storage: Whether to use persistent storage for the study.
        - out_dir: Directory to save the study results.
        - score_metrics: Metrics to use for scoring the trials. Either a string
            or a list of metrics.
        - direction: Direction of optimization, either 'minimize' or 'maximize',
            or a list of directions if using multi-objective optimization.
        - timeout_minutes_trial_sampling: Timeout in minutes for each trial sampling.
            e.g. if sampling gets stuck, will prune this trial.


        """
        if not self.has_features and not self.has_simulator:
            raise ValueError(
                """Feature array not created and no simulator.
                Please create the feature array first."""
            )

        if self.fitted_parameter_array is None and not self.has_simulator:
            raise ValueError(
                "Parameter grid not created. Please create the parameter grid first."
            )

        if self.fitted_parameter_names is None and not self.has_simulator:
            raise ValueError(
                "Parameter names not created. Please create the parameter names first."
            )

        out_dir = os.path.join(os.path.abspath(out_dir), self.name)

        if isinstance(direction, (list, tuple)):
            directions = copy.deepcopy(direction)
            direction = None
        else:
            directions = None

        if persistent_storage:
            sqlite_path = os.path.join(out_dir, f"{study_name}_optuna_study.db")

            url = create_sqlite_db(sqlite_path)
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
            )
            train_indices = self._train_indices
            test_indices = self._test_indices

            X_test, y_test = None, None
        else:
            X_test, y_test = self.generate_pairs_from_simulator(5000)
            train_indices, test_indices = None, None

        def objective_func(trial):
            """Setup function here to use shared parameters."""
            return self.run_evaluate_sbi(
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

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        # Save the study to a file
        study_path = os.path.join(
            out_dir, f"{study_name}_optuna_study_{self._timestamp}.pkl"
        )
        dump(study, study_path, compress=3)

    def run_evaluate_sbi(
        self,
        trial: optuna.Trial,
        suggested_hyperparameters: dict,
        verbose: bool = False,
        train_indices: np.ndarray = None,
        test_indices: np.ndarray = None,
        X_test: np.ndarray = None,
        y_test: np.ndarray = None,
        score: Union[str, List[str]] = "log_prob-pit",
        timeout_minutes_trial_sampling: float = 15.0,
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
                        parameters[key] = trial.suggest_float(
                            key, value[0], value[1], log=log
                        )

        parameters.update(fixed_hyperparameters)

        posterior, stats = self.run_single_sbi(
            train_indices=train_indices,
            test_indices=test_indices,
            set_self=False,
            save_model=False,
            plot=False,
            verbose=verbose,
            **parameters,
        )

        if X_test is None:
            X_test = self.feature_array[test_indices]
        if y_test is None:
            y_test = self.fitted_parameter_array[test_indices]

        timeout_seconds = (
            timeout_minutes_trial_sampling * 60
        )  # Convert minutes to seconds

        def timeout_handler(signum, frame):
            raise optuna.exceptions.TrialPruned(
                f"Execution exceeded timeout of {timeout_seconds} seconds"
            )

        # Set the signal handler
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout_seconds))

        try:
            if isinstance(score, str):
                if score == "log_prob-pit":
                    # Do the evaluation
                    score = np.mean(self.log_prob(X_test, y_test, posterior))

                    # Continue with the PIT calculation and adjust the score
                    pit = self.calculate_PIT(
                        X_test, y_test, num_samples=5000, posteriors=posterior
                    )
                    dpit_max = np.max(np.abs(pit - np.linspace(0, 1, len(pit))))
                    score += -0.5 * np.log(dpit_max)
                elif score == "log_prob":
                    score = np.mean(self.log_prob(X_test, y_test, posterior))
                else:
                    raise ValueError(f"Unknown score type: {score}")

                return score

            elif isinstance(score, list):
                scores = []
                for s in score:
                    if s == "log_prob":
                        scores.append(np.mean(self.log_prob(X_test, y_test, posterior)))
                    elif s == "tarp":
                        scores.append(
                            self.calculate_TARP(X_test, y_test, posteriors=posterior)
                        )
                    else:
                        raise ValueError(f"Unknown score type: {s}")
                return scores
            else:
                raise ValueError(
                    f"Score should be a string or a list of strings. Got {score}"
                )
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            signal.alarm(0)  # Disable the alarm
            raise e
        finally:
            signal.signal(signal.SIGALRM, old_handler)

    def run_single_sbi(
        self,
        train_test_fraction: float = 0.8,
        random_seed: int = None,
        backend: str = "sbi",
        engine: Union[str, List[str]] = "NPE",
        train_indices: np.ndarray = None,
        test_indices: np.ndarray = None,
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
        feature_scalar=StandardScaler,
        target_scalar=StandardScaler,
        set_self: bool = True,
        learning_type: str = "offline",
        simulator: callable = None,
        num_simulations=1000,
        num_online_rounds=5,
        initial_training_from_grid: bool = False,
        override_prior_ranges: dict = {},
        online_training_xobs: np.ndarray = None,
        load_existing_model: bool = True,
        use_existing_indices: bool = True,
    ) -> tuple:
        """Run a single SBI training instance.

        Parameters:
            train_test_fraction: Fraction of the dataset to be used for training.
            random_seed: Random seed for reproducibility.
            backend: Backend to use for training. Either 'sbi', 'lampe', or 'pydelfi'.
              Pydelfi cannot be installed in the same environment as it requires a
              different Python version.
            engine: Engine to use for training. Either 'NPE', 'NLE', 'NRE
                or the sequential variants (SNPE, SNLE, SNRE).
            train_indices: Indices of the training set.
            test_indices: Indices of the test set. If None, no test set is used.
            n_nets: Number of networks to use in the ensemble.
            model_type: Type of model to use. Either 'mdn' or 'maf'.
            hidden_features: Number of hidden features in the neural network.
            num_components: Number of components in the mixture density network.
            num_transforms: Number of transforms in the masked autoregressive flow.
            training_batch_size: Batch size for training.
            learning_rate: Learning rate for the optimizer.
            validation_fraction: Fraction of the training set to use for validation.
                    This validation is used to prevent over-fitting
                    training is stopped if the validation loss
                    does not improve for stop_after_epochs epochs
                    for this fraction of the training set.
            stop_after_epochs: Number of epochs without improvement before stopping.
            clip_max_norm: Maximum norm for gradient clipping.
            additional_model_args: Additional arguments to pass to the model.
                E.g. check sbi, pydelfi config. For sbi could include num_layers,
                use_batch_norm, activation, dropout probability, etc.
            save_model: Whether to save the trained model.
            verbose: Whether to print verbose output.
            prior_method: Method to create the prior. Either 'manual' or 'ili'.
            feature_scalar: Scaler for the features.
            target_scalar: Scaler for the targets.
            out_dir: Directory to save the model.
            plot: Whether to plot the diagnostics.
            name_append: Append to the model name.
            set_self: Whether to set the self object with the trained model.
            learning_type: Type of learning. Either 'offline' or 'online'.
                If 'online', you need to provide a function which takes in model arguments
                and returns a torch.Tensor of shape (1, *data.shape)
            simulator: Function to simulate the data.
                    This is only used if learning_type is 'online'.
            num_simulations: Number of simulations to run in each call
                            if learning_type is 'online'.
            num_online_rounds: Number of rounds to run in online learning.
            num_bins: Number of bins used for the splines in `nsf`. Ignored if density
            estimator not `nsf`.
            initial_training_from_grid: Whether to use the initial trainin
                from the grid when learning_type is 'online'.
                Reduces number of calls to the simulator. WARNING! BROKEN.
            override_prior_ranges: Dictionary of prior ranges to
                override the default ranges.
            online_training_xobs: A single input observation
                to condition on for online training.
            load_existing_model: Whether to load an existing model if it exists.
            use_existing_indices: Whether to use
                existing train and test indices if they exist.

        Returns:
            A tuple containing the posterior distribution and training statistics.
        """
        assert learning_type in ["offline", "online"], (
            "Learning type should be either 'offline' or 'online'."
        )
        out_dir = os.path.join(os.path.abspath(out_dir), self.name)

        if name_append == "timestamp":
            name_append = f"_{self._timestamp}"

        print(f"{out_dir}/{self.name}{name_append}_params.pkl")
        if (
            os.path.exists(f"{out_dir}/{self.name}{name_append}_params.pkl")
            and save_model
        ):
            if load_existing_model:
                print(
                    f"""Loading existing model from
                    {out_dir}/{self.name}{name_append}_params.pkl"""
                )
                posterior, stats, params = self.load_model_from_pkl(
                    f"{out_dir}/{self.name}{name_append}_posterior.pkl",
                    set_self=set_self,
                )
                return posterior, stats
            else:
                print(
                    """Model with same name already exists.
                    Please change the name of this model or delete the existing one."""
                )
                return None

        if simulator is not None:
            self.has_simulator = True

        start_time = datetime.now()

        if learning_type == "offline" or initial_training_from_grid:
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
                    print("Using existing train and test indices.")
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
                prior = ili.utils.Uniform(low=prior_low, high=prior_high)
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
            assert engine in ["SNPE", "SNLE", "SNRE"], (
                "Engine should be either 'SNPE', 'SNLE' or 'SNRE'. for online learning."
            )

            # Do online learning
            if simulator is None:
                simulator = self.simulator

            assert callable(simulator), (
                "Simulator function must be provided for online learning."
            )
            assert num_simulations > 0, "Number of simulations must be greater than 0."

            if not os.path.exists(f"{out_dir}/online/"):
                os.makedirs(f"{out_dir}/online/")

            if initial_training_from_grid:
                # Save already created data to .npy files
                np.save(
                    f"{out_dir}/online/xobs.npy",
                    self.feature_array[train_indices],
                )
                np.save(
                    f"{out_dir}/online/theta.npy",
                    self.fitted_parameter_array[test_indices],
                )
            else:
                if online_training_xobs is not None:
                    xobs = np.squeeze(online_training_xobs)
                    print(f"Using provided xobs for online training: {xobs.shape}")
                    np.save(f"{out_dir}/online/xobs.npy", xobs)

                else:
                    print(
                        """Drawing random photometry from prior to conditon on.
                        Results probably won't generalize well."""
                    )
                    samples = prior.sample_n(1)
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

            loader = SBISimulator(
                in_dir=f"{out_dir}/online/",
                xobs_file="xobs.npy",
                thetafid_file="thetafid.npy",
                num_simulations=num_simulations,
                save_simulated=True,
                x_file="x.npy",  # if initial_training_from_grid else None,
                theta_file="theta.npy",  # , if initial_training_from_grid else None,
                simulator=simulator,
            )

        nets = []
        ensemble_model_types = []
        ensemble_model_args = []
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

        # Set up trainer
        trainer = InferenceRunner.load(
            backend=backend,
            engine=engine,
            prior=prior,
            nets=nets,
            train_args=train_args,
            out_dir=out_dir if save_model else None,
            name=f"{self.name}{name_append}_",
            device=self.device,
        )

        print(f"Training on {self.device}.")
        # Train the model
        try:
            if not verbose:
                # Suppress output if not verbose
                buffer = StringIO()
                with redirect_stdout(buffer):
                    posteriors, stats = trainer(loader)
            else:
                # Train with normal output
                posteriors, stats = trainer(loader)
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
            if learning_type == "online" or initial_training_from_grid:
                self._simulator = simulator
                self._num_simulations = num_simulations
                self._num_online_rounds = num_online_rounds
                self._initial_training_from_grid = initial_training_from_grid
            else:
                self._train_indices = train_indices
                self._test_indices = test_indices
                self._train_fraction = train_test_fraction

        # Save the params with the model if needed
        if save_model:
            param_dict = {
                "engine": engine,
                "learning_type": learning_type,
                "ensemble_model_types": ensemble_model_types,
                "ensemble_model_args": ensemble_model_args,
                "n_nets": n_nets,
                "feature_names": self.feature_names,
                "fitted_parameter_names": self.fitted_parameter_names,
                "train_args": train_args,
                "stats": stats,
                "timestamp": self._timestamp,
                "prior": self._prior,
                "grid_path": self.grid_path,
                "name": self.name,
            }

            if learning_type == "online" or initial_training_from_grid:
                param_dict["simulator"] = simulator
                param_dict["num_simulations"] = num_simulations
                param_dict["num_online_rounds"] = num_online_rounds
                param_dict["initial_training_from_grid"] = initial_training_from_grid
                param_dict["online_training_xobs"] = online_training_xobs

            if learning_type == "offline":
                param_dict["train_fraction"] = train_test_fraction
                param_dict["test_indices"] = test_indices
                param_dict["train_indices"] = train_indices
                param_dict["feature_array"] = self.feature_array
                param_dict["parameter_array"] = self.fitted_parameter_array
                param_dict["feature_array_flags"] = self.feature_array_flags

            dump(
                param_dict,
                f"{out_dir}/{self.name}{name_append}_params.pkl",
                compress=3,
            )

        end_time = datetime.now()

        elapsed_time = end_time - start_time
        print(f"Time to train model(s): {elapsed_time}")

        if plot:
            if learning_type == "offline":
                # Deal with the sampling method.
                if engine in ["NLE"]:
                    sample_method = "emcee"
                else:
                    sample_method = "direct"
                self.plot_diagnostics(
                    X_train=X_scaled,
                    y_train=y_scaled,
                    X_test=X_test,
                    y_test=y_test,
                    plots_dir=f"{out_dir}/plots/{name_append}/",
                    stats=stats,
                    sample_method=sample_method,
                    posteriors=posteriors,
                )
            else:
                self.plot_diagnostics(
                    plots_dir=f"{out_dir}/online/plots/{name_append}/",
                    stats=stats,
                    sample_method=sample_method,
                    posteriors=posteriors,
                    online=True,
                )

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
            print(
                f"""Creating {model_type} network with {engine} engine
                and {backend} backend."""
            )
            for key, value in model_args.items():
                print(f"     {key}: {value}")

        return net(
            engine=engine,
            model=model_type,
            # embedding_net=embedding_net,
            **backend_args,
            **model_args,
        )

    def recover_SED(
        self,
        X_test: np.ndarray,
        n_samples=1000,
        sample_method: str = "direct",
        sample_kwargs: dict = {},
        posteriors=None,
        simulator: callable = None,
        prior: object = None,
        plot: bool = True,
        marginalized_parameters: Dict[str, callable] = None,
        phot_unit="AB",
        true_parameters=[],
        plot_name=None,
        plots_dir=f"{code_path}/models/name/plots/",
        sample_color="violet",
        param_labels=None,
        plot_closest_draw_to={},
    ):
        """Recover the SED for a given observation, if a simulator is provided.

        Parameters:
            X_test: The input observation to recover the SED for.
            n_samples: Number of samples to draw from the posterior.
            sample_method: Method to sample from the posterior.
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
        """
        if posteriors is None:
            posteriors = self.posteriors

        if simulator is None:
            if not hasattr(self, "_simulator"):
                raise ValueError("Simulator must be provided or set in the object.")
            simulator = self._simulator

        if prior is None:
            if not hasattr(self, "_prior"):
                raise ValueError("Prior must be provided or set in the object.")
            prior = self._prior

        plots_dir = plots_dir.replace("name", self.name)

        # Draw samples, run through simulator,

        samples = self.sample_posterior(
            X_test=[X_test],
            sample_method=sample_method,
            num_samples=n_samples,
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
        if isinstance(simulator, GalaxySimulator):
            simulator.output_type = ["photo_fnu", "fnu", "sfh"]
            for i in range(n_samples):
                params = {
                    self.simple_fitted_parameter_names[j]: samples[i, j]
                    for j in range(len(self.fitted_parameter_names))
                }
                for parameter in marginalized_parameters.keys():
                    params[parameter] = marginalized_parameters[parameter](params)

                output = simulator(params)
                if i == 0:
                    phot_wav = output["photo_wav"]
                    filters = output["filters"]
                wav_draws.append(output["fnu_wav"])
                fnu_draws.append(output["fnu"])
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
        print(f"Number of NaN SFH: {counter}")

        extra_indexes = []
        extra_labels = []
        extra_lines = []
        if len(plot_closest_draw_to) > 0:
            for key, value in plot_closest_draw_to.items():
                if (
                    key in self.fitted_parameter_names
                    or key in self.simple_fitted_parameter_names
                ):
                    if key in self.simple_fitted_parameter_names:
                        index = list(self.simple_fitted_parameter_names).index(key)
                    else:
                        index = list(self.fitted_parameter_names).index(key)

                    closest_index = np.argmin(np.abs(samples[:, index] - value))
                    print(
                        f"""Closest draw to {key}={value} is
                        {samples[closest_index, index]} at index {closest_index}"""
                    )
                    print(f"Full draw: {samples[closest_index]}")
                    extra_indexes.append(closest_index)
                    extra_labels.append(f"{key}={value}")

        fnu_quantiles = np.quantile(fnu_draws, [0.16, 0.5, 0.84], axis=0)
        phot_fnu_quantiles = np.quantile(phot_fnu_draws, [0.16, 0.5, 0.84], axis=0)
        sfh_quantiles = np.nanquantile(sfh, [0.16, 0.5, 0.84], axis=0)
        wav = np.mean(wav_draws, axis=0)
        plot_sfh = True
        if plot:
            fig = plt.Figure(figsize=(8, 6), dpi=200, constrained_layout=True)
            gridspec = fig.add_gridspec(
                2, len(self.fitted_parameter_names), height_ratios=[1, 0.6]
            )
            ax = fig.add_subplot(gridspec[0, :])

            if plot_sfh:
                # inset axes for SFH inside ax
                inset_ax = fig.add_axes([0.71, 0.55, 0.25, 0.25])
                # plot the SFH
                # temp
                time = output["sfh_time"]
                inset_ax.plot(
                    time,
                    sfh_quantiles[1],
                    label="Median SFH",
                    color=sample_color,
                )
                inset_ax.fill_between(
                    time,
                    sfh_quantiles[0],
                    sfh_quantiles[2],
                    alpha=0.5,
                    label="68% CI SFH",
                    color=sample_color,
                )
                # Don't let the time go beyond 0
                inset_ax.set_xlim(0, 1000)
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
                label="68% CI SED",
                color=sample_color,
                zorder=7,
            )
            for f, lam in zip(fnu_draws, wav_draws):
                ax.plot(lam, f, color="violet", alpha=0.05, lw=0.2, zorder=5)

            if len(extra_indexes) > 0:
                for i, index in enumerate(extra_indexes):
                    if np.sum(fnu_draws[index]) == 0:
                        print(
                            f"Warning! The draw at index {index} has all zeros. Skipping."
                        )
                    line = ax.plot(
                        wav,
                        fnu_draws[index],
                        label=extra_labels[i],
                        lw=1,
                        zorder=10,
                    )
                    extra_lines.append(line)

            ax.set_xlabel("Wavelength (Angstrom)")
            ax.set_ylabel("Flux Density (AB mag")

            # Try and match filters names to feature names
            # and plot the photometry we have been given
            filter_codes = [i.split("/")[-1] for i in filters.filter_codes]

            if true_parameters is not None:
                # make dict
                true_parameters_dict = {
                    self.simple_fitted_parameter_names[j]: true_parameters[j]
                    for j in range(len(self.fitted_parameter_names))
                }
                for parameter in marginalized_parameters.keys():
                    true_parameters_dict[parameter] = marginalized_parameters[parameter](
                        true_parameters_dict
                    )

                true_sed_output = simulator(true_parameters_dict)
                true_sed = true_sed_output["fnu"]
                ax.plot(
                    wav,
                    true_sed,
                    label="True SED",
                    color="red",
                    lw=1,
                    zorder=11,
                )
                if plot_sfh:
                    true_sfh = true_sed_output["sfh"]
                    inset_ax.plot(time, true_sfh, label="True SFH", color="red")

                    if len(extra_indexes) > 0:
                        for i, index in enumerate(extra_indexes):
                            inset_ax.plot(
                                time,
                                true_sfh,
                                label=extra_labels[i],
                                lw=1,
                                zorder=10,
                                color=extra_lines[i][0].get_color(),
                            )

            # Match indexes of the filters to the feature names
            # Now plot the photometry we have been given (X_test).
            # TODO; Deal with the fact that this could be normalized
            labelled = False
            phots = []

            for pos, filter in enumerate(filter_codes):
                index = self.feature_names.index(filter)
                phot = X_test[index]
                phots.append(phot)

            phots = np.array(phots)
            median_phot = np.nanmedian(phots)
            max_phot_diff = 4
            show_lims = phots > (median_phot + max_phot_diff)
            phots[show_lims] = median_phot + max_phot_diff

            for pos, filter in enumerate(filter_codes):
                if filter in self.feature_names:
                    if f"unc_{filter}" in self.feature_names:
                        # Get the index of the filter
                        index = self.feature_names.index(f"unc_{filter}")
                        phot_unc = X_test[index]
                    else:
                        phot_unc = 0
                    index = self.feature_names.index(filter)
                    phot = X_test[index]
                    if show_lims[pos]:
                        # phot = median_phot + max_phot_diff
                        # Plot a downward arrow patch
                        from matplotlib.patches import FancyArrowPatch

                        phot = median_phot + max_phot_diff
                        ax.add_patch(
                            FancyArrowPatch(
                                (phot_wav[pos], phot - 0.2),
                                (phot_wav[pos], phot + 0.2),
                                arrowstyle="->",
                                mutation_scale=10,
                                color="black",
                                lw=1,
                                zorder=12,
                            )
                        )

                    else:
                        if phot_unc == 0:
                            ax.scatter(
                                phot_wav[pos],
                                phot,
                                label="Input Phot." if not labelled else "",
                                marker="o",
                                color="black",
                                zorder=12,
                                s=10,
                            )
                        else:
                            ax.errorbar(
                                phot_wav[pos],
                                phot,
                                yerr=phot_unc,
                                label="Input Phot." if not labelled else "",
                                marker="o",
                                color="black",
                                zorder=12,
                                markersize=5,
                                linestyle="None",
                            )
                    labelled = True

                # Plot the photometry we have drawn from the posterior

                phot = phot_fnu_quantiles[1][pos]

                phot_lower = phot_fnu_quantiles[0][pos] - phot
                phot_upper = phot - phot_fnu_quantiles[2][pos]

                if phot_unit != "AB":
                    err = [[phot_upper], [phot_lower]]
                else:
                    err = [[phot_lower], [phot_upper]]
                err = np.abs(err)
                ax.errorbar(
                    phot_wav[pos],
                    phot,
                    yerr=err,
                    label="Posterior Phot." if pos == 0 else "",
                    marker="s",
                    color=sample_color,
                    zorder=9,
                    markersize=5,
                    linestyle="None",
                )

            # Set min and max of the x axis based on observed photomery

            min_x, max_x = filters.get_non_zero_lam_lims()

            phots = np.array(phots)
            max_phot = 1.01 * np.nanmax(phots[phots < 35])

            ax.set_xlim(min_x, max_x)
            fnu_lam_mask = (wav > min_x) & (wav < max_x)
            max_f = min(1.01 * np.nanmax(fnu_quantiles[2][fnu_lam_mask]), max_phot)
            min_f = 0.99 * np.nanmin(fnu_quantiles[0][fnu_lam_mask])

            ax.set_ylim(min_f, max_f)
            ax.legend(fontsize=8, loc="upper left")

            # If AB mag, flip y axis
            if phot_unit == "AB":
                ax.invert_yaxis()
                ax.set_ylabel("Flux Density (AB mag)")

            if param_labels is None:
                param_labels = self.simple_fitted_parameter_names

            # Add a row of axis underneath and plot histograms of parameters
            for i, param in enumerate(param_labels):
                ax = fig.add_subplot(gridspec[1, i])
                ax.hist(
                    samples[:, i],
                    bins=50,
                    density=False,
                    alpha=0.9,
                    color=sample_color,
                )
                ax.set_xlabel(param)
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

            if plot_name is None:
                plot_name = f"{self.name}_SED_{self._timestamp}.png"
            fig.savefig(os.path.join(plots_dir, plot_name), dpi=200)

            return fig

        else:
            return fnu_quantiles, wav, phot_fnu_draws, phot_wav

    def sample_posterior(
        self,
        X_test: np.ndarray = None,
        sample_method: str = "direct",
        sample_kwargs: dict = {},
        posteriors: object = None,
        num_samples: int = 1000,
        timeout_seconds_per_test=30,
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
            raise ValueError(
                "Invalid sample method. Use 'direct', 'emcee'/'mcmc', 'pyro' or 'vi'."
            )

        sampler = sampler(posteriors, **sample_kwargs)

        # Properly check dimensionality and shape
        X_test_array = np.asarray(X_test)  # Ensure it's a numpy array

        # Handle single sample case
        if X_test_array.ndim == 1 or (
            X_test_array.ndim == 2 and X_test_array.shape[0] == 1
        ):
            return sample_with_timeout(
                sampler, num_samples, X_test_array, timeout_seconds_per_test
            )

        # Handle multiple samples case
        test_sample = sample_with_timeout(
            sampler, num_samples, X_test_array[0], timeout_seconds_per_test
        )
        shape = test_sample.shape

        print("drawin samples")

        # Draw samples from the posterior
        samples = np.zeros((len(X_test_array), num_samples, shape[1]))
        samples[0] = test_sample  # First sample is already drawn
        for i in tqdm(range(1, len(X_test_array))):
            try:
                samples[i] = sample_with_timeout(
                    sampler,
                    num_samples,
                    X_test_array[i],
                    timeout_seconds_per_test,
                )
            except TimeoutException:
                print(
                    f"""Timeout exceeded for sample {i}.
                    Returning empty array for this sample."""
                )
                samples[i] = np.nan
            except KeyboardInterrupt:
                print("Sampling interrupted by user. Returning samples collected so far.")
                break

        return samples

    def _evaluate_model(
        self,
        posteriors: list,
        X_test: np.ndarray,
        y_test: np.ndarray,
        num_samples: int = 1000,
    ) -> dict:
        """Evaluate the trained model on test data.

        Parameters:
            X_test: Test feature array.
            y_test: Test target array.
            X_scaler: Scaler for the features (if used).
            y_scaler: Scaler for the targets (if used).
            num_samples: Number of samples to draw from the posterior.

        Returns:
            A dictionary of evaluation metrics.
        """
        # Draw samples from the posterior
        samples = self.sample_posterior(
            X_test, num_samples=num_samples, posteriors=posteriors
        )

        # Calculate basic metrics
        mean_pred = np.mean(samples, axis=1)
        median_pred = np.median(samples, axis=1)

        # Calculate metrics
        metrics = {
            "mse": np.mean((y_test - mean_pred) ** 2),
            "rmse": np.sqrt(np.mean((y_test - mean_pred) ** 2)),
            "mae": np.mean(np.abs(y_test - mean_pred)),
            "median_ae": np.median(np.abs(y_test - median_pred)),
        }

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
    ) -> None:
        """Plot the diagnostics of the SBI model."""
        plots_dir = plots_dir.replace("name", self.name)
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        if stats is None:
            if hasattr(self, "stats"):
                stats = self.stats
            else:
                raise ValueError("No stats found. Please provide the stats.")

        if X_train is None or y_train is None and not online:
            if hasattr(self, "_X_train") and hasattr(self, "_y_train"):
                X_train = self._X_train
                y_train = self._y_train
            else:
                raise ValueError(
                    "X_train and y_train must be provided or set in the object."
                )
        if X_test is None or y_test is None and not online:
            if hasattr(self, "_X_test") and hasattr(self, "_y_test"):
                X_test = self._X_test
                y_test = self._y_test
            else:
                raise ValueError(
                    "X_test and y_test must be provided or set in the object."
                )

        # Plot the loss
        self.plot_loss(stats, plots_dir=plots_dir)

        if not online:
            # Plot the posterior
            self.plot_posterior(
                X=X_test,
                y=y_test,
                plots_dir=plots_dir,
                sample_method=sample_method,
                posteriors=posteriors,
            )

        # Plot the coverage
        self.plot_coverage(
            X=X_test,
            y=y_test,
            plots_dir=plots_dir,
            sample_method=sample_method,
            posteriors=posteriors,
        )

    def plot_loss(
        self,
        summaries: list = "self",
        plots_dir: str = f"{code_path}/models/name/plots/",
    ) -> None:
        """Plot the loss of the SBI model."""
        if summaries == "self":
            if hasattr(self, "stats"):
                summaries = self.stats
            else:
                raise ValueError("No summaries found. Please provide the summaries.")

        plots_dir = plots_dir.replace("name", self.name)

        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        # plot train/validation loss
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        for i, m in enumerate(summaries):
            ax.plot(m["training_log_probs"], ls="-", label=f"{i}_train")
            ax.plot(m["validation_log_probs"], ls="--", label=f"{i}_val")
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
    ):
        """Plot histogram of each parameter using astropy.visualization.hist."""
        fig, axes = plt.subplots(1, len(self.fitted_parameter_names), figsize=(15, 5))
        for i, param in enumerate(self.simple_fitted_parameter_names):
            axes[i].set_title(param)
            axes[i].set_xlabel("Value")
            axes[i].set_ylabel("Count")
            if seperate_test_train:
                hist(self._y_train[:, i], ax=axes[i], bins=bins, label="Train")
                hist(self._y_test[:, i], ax=axes[i], bins=bins, label="Test")
                axes[i].legend()
            else:
                hist(self.fitted_parameter_array[:, i], ax=axes[i], bins=bins)

        plt.tight_layout()

        plots_dir = plots_dir.replace("name", self.name)

        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        print("saving", f"{plots_dir}/param_histogram.png")
        fig.savefig(f"{plots_dir}/param_histogram.png", dpi=300)

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
            raise ValueError("X and y must be provided to plot the posterior.")

        if ind == "random":
            if seed is not None:
                np.random.seed(seed)
            ind = np.random.randint(0, X.shape[0])

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

        fig = metric(
            posterior=posteriors,
            x_obs=X[ind],
            theta_fid=y[ind],
            plot_kws=plot_kwargs,
            signature=f"{self.name}_{ind}_",
            **kwargs,
        )
        return fig

    def plot_posterior_samples(self):
        """Plot the posterior samples of the SBI model."""
        pass

    def plot_posterior_predictions(self):
        """Plot the posterior predictions of the SBI model."""
        pass

    def calculate_TARP(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_samples: int = 1000,
        posteriors: object = None,
        num_bootstrap=200,
    ) -> np.ndarray:
        """Calculate the total absolute residual probability (TARP).

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
            x = torch.tensor([X[i]], dtype=torch.float32, device=self.device)
            theta = torch.tensor([y[i]], dtype=torch.float32, device=self.device)
            lp[i] = posteriors.log_prob(x=x, theta=theta)  # norm_posterior=True)

        return lp

    def plot_latent_residual(self):
        """Plot the latent residual of the SBI model."""
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

        metric = PosteriorCoverage(
            num_samples=num_samples,
            sample_method=sample_method,
            labels=self.simple_fitted_parameter_names,
            plot_list=plot_list,
            out_dir=plots_dir,
            **sample_kwargs,
        )

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
                posterior=posteriors
                if posterior_plot_type == "total"
                else posteriors[posterior_plot_type],
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
        -------
        list of 1-dimensional arrays
        """
        if self.stats is None:
            raise RuntimeError("The regressor has not been fitted yet.")

        return [stat["validation_log_probs"] for stat in self.stats]

    @property
    def training_log_probs(self):
        """Training set log-probability of each epoch for each net.

        Returns:
        -------
        list of 1-dimensional array
        """
        if self.stats is None:
            raise RuntimeError("The regressor has not been fitted yet.")

        return [stat["training_log_probs"] for stat in self.stats]

    def load_model_from_pkl(
        self,
        model_file: str,
        set_self: bool = True,
    ) -> Tuple[list, dict, dict]:
        """Load the model from a pickle file.

        Parameters
        ----------
        model_file : str
            Path to the model file. Can be a directory or a file.
        set_self : bool, optional
            If True, set the attributes of the class to the loaded values.
            Default is True.

        Returns:
        -------
        posteriors : list
            List of posteriors loaded from the model file.
        stats : dict
            Dictionary of statistics loaded from the model file.
        params : dict
            Dictionary of parameters loaded from the model file.
        """
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

        with open(model_file, "rb") as f:
            posteriors = load(f)
        #
        stats = model_file.replace("posterior.pkl", "summary.json")

        if os.path.exists(stats):
            with open(stats, "r", encoding="utf-8") as f:
                stats = json.load(f)

            if set_self:
                self.stats = stats

        else:
            stats = None
            print(f"Warning: No summary file found for {model_file}.")

        if set_self:
            self.posteriors = posteriors

        params = model_file.replace("posterior.pkl", "params.pkl")
        if os.path.exists(params):
            with open(params, "rb") as f:
                params = load(f)

            if set_self:
                # Set attributes of class again.
                self.fitted_parameter_names = params["fitted_parameter_names"]
                self.simple_fitted_parameter_names = [
                    i.split("/")[-1] for i in self.fitted_parameter_names
                ]
                learning_type = params.get("learning_type", "offline")
                self.feature_names = params["feature_names"]

                # print(params.keys())

                if learning_type == "offline":
                    self.fitted_parameter_array = params["parameter_array"]
                    self.feature_array = params["feature_array"]
                    if "feature_array_flags" in params:
                        self.feature_array_flags = params["feature_array_flags"]
                    if self.feature_array is not None:
                        self.has_features = True
                    self.parameter_array = params["parameter_array"]
                    self._train_indices = params["train_indices"]
                    self._test_indices = params["test_indices"]
                    self._train_fraction = params["train_fraction"]
                    self._X_test = self.feature_array[self._test_indices]
                    self._y_test = self.fitted_parameter_array[self._test_indices]
                    self._X_train = self.feature_array[self._train_indices]
                    self._y_train = self.fitted_parameter_array[self._train_indices]
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
                    self._initial_training_from_grid = params[
                        "initial_training_from_grid"
                    ]

                self._train_args = params["train_args"]
                self._prior = params["prior"]

                self._ensemble_model_types = params["ensemble_model_types"]
                self._ensemble_model_args = params["ensemble_model_args"]

        else:
            print(f"Warning: No parameter file found for {model_file}.")
            params = None

        return posteriors, stats, params

    @property
    def _timestamp(self):
        """Get the current date and time as a string."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def generate_pairs_from_simulator(self, n_samples=1000):
        """Generate pairs of data from the simulator."""
        if not self.has_simulator:
            raise ValueError("No simulator found. Please provide a simulator.")

        if self.simulator is None:
            raise ValueError("No simulator found. Please provide a simulator.")

        if self._prior is None:
            raise ValueError("No prior found. Please provide a prior.")

        samples = self._prior.sample_n(n_samples)

        phot = []
        for i in range(len(samples)):
            p = self.simulator(samples[i]).cpu().numpy()
            phot.append(p)
        phot = np.array(phot)
        phot = np.squeeze(phot)

        # shape phot to be (num_simulations, num_features)
        # X, y
        return phot, samples.cpu().numpy()


class MissingPhotometryHandler:
    """Based on sbi++ approach."""

    def __init__(
        self,
        training_photometry,
        training_parameters,
        posterior_estimator=None,
        run_params=None,
        device="cpu",
    ):
        """Initialize the missing photometry handler.

        Parameters:
        -----------
        training_photometry : np.ndarray
            Training set photometry with shape (n_samples, n_bands)
        training_parameters : np.ndarray
            Training set parameters with shape (n_samples, n_params)
        posterior_estimator : callable, optional
            SBI model that returns posterior for a given SED
        run_params : dict, optional
            Dictionary of runtime parameters
        """
        self.y_train = training_photometry
        self.x_train = training_parameters
        self.posterior_estimator = posterior_estimator

        # Default run parameters if not provided
        self.run_params = {
            "ini_chi2": 5.0,  # Initial chi2 threshold
            "max_chi2": 50.0,  # Maximum chi2 threshold
            "nmc": 100,  # Number of Monte Carlo samples
            "nposterior": 1000,  # Number of posterior samples
            "tmax_per_obj": 30,  # Maximum time per object in seconds
            "tmax_all": 10,  # Maximum total time in minutes
            "verbose": False,  # Verbose output
        }

        if run_params is not None:
            self.run_params.update(run_params)
        self.device = device

    @classmethod
    def init_from_sbifitter(cls, sbifitter, **run_params):
        """Initialize from a fitted SBI model.

        Parameters:
        -----------
        sbifitter : object
            Fitted SBI model object

        Returns:
        --------
        MissingPhotometryHandler
            Instance of the handler
        """
        return cls(
            training_photometry=sbifitter.feature_array,
            training_parameters=sbifitter.fitted_parameter_array,
            posterior_estimator=sbifitter.posteriors,
            run_params=run_params,
        )

    def chi2dof(self, mags, obsphot, obsphot_unc):
        """Calculate reduced chi-square.

        Parameters:
        -----------
        mags : np.ndarray
            Model magnitudes/fluxes with shape (n_models, n_bands)
        obsphot : np.ndarray
            Observed photometry with shape (n_bands,)
        obsphot_unc : np.ndarray
            Observed photometry uncertainties with shape (n_bands,)

        Returns:
        --------
        np.ndarray
            Reduced chi-square values for each model
        """
        chi2 = np.nansum(((mags - obsphot) / obsphot_unc) ** 2, axis=1)
        return chi2 / np.sum(np.isfinite(obsphot))

    def gauss_approx_missingband(self, obs):
        """Nearest neighbor approximation of missing bands using KDE.

        Parameters:
        -----------
        obs : dict
            Dictionary with observed data containing:
            - mags_sbi: observed photometry
            - mags_unc_sbi: uncertainties
            - missing_mask: boolean mask (True for missing bands)

        Returns:
        --------
        tuple
            (list of KDEs for missing bands, success flag)
        """
        use_res = False
        y_train = self.y_train

        y_obs = np.copy(obs["mags_sbi"])
        sig_obs = np.copy(obs["mags_unc_sbi"])
        invalid_mask = np.copy(obs["missing_mask"])
        y_obs_valid_only = y_obs[~invalid_mask]
        valid_idx = np.where(~invalid_mask)[0]
        not_valid_idx = np.where(invalid_mask)[0]

        # Find chi-square values for training set using only observed bands
        look_in_training = y_train[:, valid_idx]
        chi2_nei = self.chi2dof(
            mags=look_in_training,
            obsphot=y_obs[valid_idx],
            obsphot_unc=sig_obs[valid_idx],
        )

        # Incrementally increase chi2 threshold until we have enough neighbors
        _chi2_thres = self.run_params["ini_chi2"]
        use_res = True

        while _chi2_thres <= self.run_params["max_chi2"]:
            idx_chi2_selected = np.where(chi2_nei <= _chi2_thres)[0]
            if len(idx_chi2_selected) >= 30:
                break
            else:
                _chi2_thres += 5

        # If we couldn't find enough neighbors, use top 100 neighbors
        if _chi2_thres > self.run_params["max_chi2"]:
            use_res = False
            chi2_selected = y_train[:, valid_idx]
            chi2_selected = chi2_selected[:100]
            guess_ndata = y_train[:, not_valid_idx]
            guess_ndata = guess_ndata[:100]
            if self.run_params["verbose"]:
                print("Failed to find sufficient number of nearest neighbors!")
                print(
                    f"_chi2_thres {_chi2_thres} > max_chi2 {self.run_params['max_chi2']}",
                    len(guess_ndata),
                )
        else:
            chi2_selected = y_train[:, valid_idx][idx_chi2_selected]
            # Get distribution of the missing bands
            guess_ndata = y_train[:, not_valid_idx][idx_chi2_selected]

        # Calculate Euclidean distances and weights
        dists = np.linalg.norm(y_obs_valid_only - chi2_selected, axis=1)

        # Handle a distance of 0
        dists[dists == 0] = 1e-10

        neighs_weights = 1 / dists

        # Create KDEs for each missing band
        kdes = []
        for i in range(guess_ndata.shape[1]):
            _kde = stats.gaussian_kde(guess_ndata.T[i], 0.2, weights=neighs_weights)
            kdes.append(_kde)

        return kdes, use_res

    def sbi_missingband(self, obs, noise_generator=None):
        """Process missing bands for SBI.

        Parameters:
        -----------
        obs : dict
            Dictionary with observed data containing:
            - mags_sbi: observed photometry
            - mags_unc_sbi: uncertainties
            - missing_mask: boolean mask (True for missing bands)
        noise_generator : callable, optional
            Function to generate noise for sampled fluxes

        Returns:
        --------
        tuple
            (averaged posteriors, reconstructed photometry,
                 success flag, timeout flag, count)
        """
        if self.run_params["verbose"]:
            print("Processing missing bands with SBI")

        ave_theta = []

        y_obs = np.copy(obs["mags_sbi"])
        # sig_obs = np.copy(obs["mags_unc_sbi"])
        invalid_mask = np.copy(obs["missing_mask"])

        # Full observed vector (fluxes + uncertainties)
        # observed = np.concatenate([y_obs, sig_obs])
        observed = y_obs

        # Indices for valid and invalid bands
        # valid_idx = np.where(~invalid_mask)[0]
        not_valid_idx = np.where(invalid_mask)[0]

        st = time.time()

        # Get KDEs for missing bands
        kdes, use_res = self.gauss_approx_missingband(obs)

        nbands = len(y_obs)  # Total number of bands
        # not_valid_idx_unc = not_valid_idx + nbands

        all_x = []
        cnt = 0
        cnt_timeout = 0
        timeout_flag = False

        # Draw Monte Carlo samples and get posteriors
        while cnt < self.run_params["nmc"]:
            # signal.alarm(self.run_params['tmax_per_obj'])  # Max time per object

            try:
                # Create a copy of observations to fill in missing bands
                x = np.copy(observed)

                # Sample from KDEs for each missing band
                for j in range(len(not_valid_idx)):
                    # Sample flux for missing band
                    x[not_valid_idx[j]] = kdes[j].resample(size=1)[0]

                    """
                    # This code would handle if model had uncertainties.
                    # Generate noise for sampled flux
                    if noise_generator is not None:
                        # Use provided noise generator
                        x[not_valid_idx_unc[j]] = noise_generator(
                            flux=x[not_valid_idx[j]],
                            verbose=self.run_params['verbose']
                        )[1]
                    else:
                        # Default noise (10% of flux)
                        x[not_valid_idx_unc[j]] = 0.1 * x[not_valid_idx[j]]
                    """

                all_x.append(x)

                # Get posterior samples using SBI
                if self.posterior_estimator is not None:
                    x_tensor = torch.as_tensor(x.astype(np.float32)).to(self.device)
                    noiseless_theta = self.posterior_estimator.sample(
                        (self.run_params["nposterior"],),
                        x=x_tensor,
                        show_progress_bars=False,
                    )
                    noiseless_theta = noiseless_theta.detach().cpu().numpy()
                    ave_theta.append(noiseless_theta)

                cnt += 1
                if self.run_params["verbose"] and cnt % 10 == 0:
                    print("Monte Carlo samples:", cnt)

            except TimeoutException:
                cnt_timeout += 1
            # else:
            #    signal.alarm(0)

            # Check if total time exceeded
            et = time.time()
            elapsed_time = et - st  # in seconds
            if elapsed_time / 60 > self.run_params["tmax_all"]:
                timeout_flag = True
                use_res = False
                break

        # Process results
        all_x = np.array(all_x)
        all_x_flux = all_x[:, :nbands]
        # all_x_unc = all_x[:, nbands:]

        # Calculate median fluxes and uncertainties
        """y_guess = np.concatenate([
            np.median(all_x_flux, axis=0),
            np.median(all_x_unc, axis=0)
        ])"""
        y_guess = np.median(all_x_flux, axis=0)
        y_filled_dist = all_x_flux[:, not_valid_idx]

        return ave_theta, y_guess, use_res, timeout_flag, cnt, y_filled_dist

    def get_average_posterior(self, ave_theta):
        """Average posterior samples.

        Parameters:
        -----------
        ave_theta : list
            List of posterior samples from multiple runs

        Returns:
        --------
        np.ndarray
            Averaged posterior parameters
        """
        if not ave_theta:
            return None

        # Convert to numpy array if needed
        if isinstance(ave_theta[0], torch.Tensor):
            ave_theta = [t.detach().cpu().numpy() for t in ave_theta]

        # Concatenate all posterior samples
        all_samples = np.concatenate(ave_theta, axis=0)

        # Calculate statistics
        mean_params = np.mean(all_samples, axis=0)
        median_params = np.median(all_samples, axis=0)
        std_params = np.std(all_samples, axis=0)

        return {
            "mean": mean_params,
            "median": median_params,
            "std": std_params,
            "samples": all_samples,
        }

    def process_observation(self, obs, noise_generator=None):
        """Process a single observation with missing bands.

        Parameters:
        -----------
        obs : dict
            Dictionary with observed data containing:
            - mags_sbi: observed photometry
            - mags_unc_sbi: uncertainties
            - missing_mask: boolean mask (True for missing bands)
        noise_generator : callable, optional
            Function to generate noise for sampled fluxes

        Returns:
        --------
        dict
            Dictionary with results
        """
        # Check if there are missing bands
        if np.sum(obs["missing_mask"]) == 0:
            if self.run_params["verbose"]:
                print("No missing bands, using standard SBI")

            # Use standard SBI with complete data
            if self.posterior_estimator is not None:
                # x = np.concatenate([obs['mags_sbi'], obs['mags_unc_sbi']])
                x = obs["mags_sbi"]

                x_tensor = torch.as_tensor(x.astype(np.float32)).to(self.device)
                samples = self.posterior_estimator.sample(
                    (self.run_params["nposterior"],),
                    x=x_tensor,
                    show_progress_bars=False,
                )
                samples = samples.detach().cpu().numpy()

                return {
                    "posterior": {
                        "mean": np.mean(samples, axis=0),
                        "median": np.median(samples, axis=0),
                        "std": np.std(samples, axis=0),
                        "samples": samples,
                    },
                    "reconstructed_photometry": None,
                    "success": True,
                    "timeout": False,
                }
            else:
                return {
                    "posterior": None,
                    "reconstructed_photometry": None,
                    "success": False,
                    "timeout": False,
                }

        # Process observation with missing bands
        (
            ave_theta,
            reconstructed_phot,
            success,
            timeout,
            count,
            missing_dist,
        ) = self.sbi_missingband(obs, noise_generator=noise_generator)

        # Calculate average posterior
        posterior = self.get_average_posterior(ave_theta) if ave_theta else None

        return {
            "posterior": posterior,
            "reconstructed_photometry": reconstructed_phot,
            "missing_photometry_dist": missing_dist,
            "success": success,
            "timeout": timeout,
            "count": count,
        }


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
