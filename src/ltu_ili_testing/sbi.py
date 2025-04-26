from .grid import CombinedBasis
import os
import h5py
import numpy as np
from unyt import unyt_array, nJy, Msun
import re
import operator
from typing import List, Dict, Union
import optuna
import sys
from io import StringIO
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
from joblib import dump, load
import json
from datetime import datetime

import torch
import torch.nn as nn

import ili
from ili.dataloaders import NumpyLoader
from ili.inference import InferenceRunner
from ili.validation.metrics import PlotSinglePosterior, PosteriorCoverage, PosteriorSamples
from ili.validation.runner import ValidationRunner
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


device = 'cuda' if torch.cuda.is_available() else 'cpu'

verbose = False
if verbose:
    print('Device:', device)
    print("Pytorch version: " + torch.__version__)
    print("ROCM HIP version: " + torch.version.hip)
    print("CUDA version: " + torch.version.cuda)
torch.cuda.set_device(f'{device}:0')

# get path of this file

code_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_grid_from_hdf5(hdf5_path: str, 
                        photometry_key: str = 'Grid/Photometry',
                        parameters_key: str = 'Grid/Parameters',
                        filter_codes_attr: str = 'FilterCodes',
                        parameters_attr: str = 'ParameterNames',
                        supp_key: str = 'Grid/SupplementaryParameters',
                        supp_attr: str = 'SupplementaryParameterNames',
                        supp_units_attr: str = 'SupplementaryParameterUnits',
                        phot_unit_attr: str = 'PhotometryUnits',
                        ) -> dict:
    """
    Load a grid from an HDF5 file.

    Args:
        hdf5_path: Path to the HDF5 file.

    Returns:
        The loaded grid.
    """
    
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'r') as f:
        # Load the photometry and parameters from the HDF5 file
        photometry = f[photometry_key][:]
        parameters = f[parameters_key][:]
        
        filter_codes = f.attrs[filter_codes_attr]
        parameter_names = f.attrs[parameters_attr]

        # Load the photometry units
        photometry_units = f.attrs[phot_unit_attr]

        output = {
            'photometry': photometry,
            'parameters': parameters,
            'filter_codes': filter_codes,
            'parameter_names': parameter_names,
            'photometry_units': photometry_units,
        }

        # Load supplementary parameters if available
        if supp_key in f:
            supplementary_parameters = f[supp_key][:]
            supplementary_parameter_names = f.attrs[supp_attr]
            supplementary_parameter_units = f.attrs[supp_units_attr]
            output['supplementary_parameters'] = supplementary_parameters
            output['supplementary_parameter_names'] = supplementary_parameter_names
            output['supplementary_parameter_units'] = supplementary_parameter_units

    return output


class SBI_Fitter:
    """
    Class to fit a model to the data using the ltu-ili package.

    Datasets are loaded from HDF5 files. The data is then split into training and testing sets.
    Flexible models including ensembles are supported.


    """

    device = device

    def __init__(self,
                name: str,
                raw_photometry_grid: np.ndarray,
                raw_photometry_names: list,
                parameter_array: np.ndarray,
                parameter_names: list,
                raw_photometry_units: list = nJy,
                feature_array: np.ndarray = None,
                feature_names: list = None,
                feature_units: list = None,
                grid_path: str = None,
                supplementary_parameters: np.ndarray = [],
                supplementary_parameter_names: list = [],
                supplementary_parameter_units: list = [],
                device: str = device,
                ) -> None:

        """
        Initialize the SBI fitter.

        Args:
            model: The model to be fitted.
            prior: The prior distribution.
            simulator: The simulator function.
        """
        
        self.name = name
        self.raw_photometry_grid = raw_photometry_grid
        self.raw_photometry_units = raw_photometry_units
        self.raw_photometry_names = raw_photometry_names
        self.parameter_array = parameter_array
        self.parameter_names = parameter_names
        
        # This allows you to subset the parameters to fit if you want to marginalize over some parameters.
        # See self.update_parameter_array() for more details.
        self.fitted_parameter_array = parameter_array
        self.fitted_parameter_names = parameter_names

        # Feature array and names
        self.feature_array = feature_array
        self.feature_names = feature_names
        self.feature_units = feature_units
        self.provided_feature_parameters = [] # This stores the parameters which are provided as features e.g. redshift. 
        # They are removed from the parameter array and names. 

        # Supplementary parameters
        self.supplementary_parameters = supplementary_parameters
        self.supplementary_parameter_names = supplementary_parameter_names
        self.supplementary_parameter_units = supplementary_parameter_units

        # Grid path
        self.grid_path = grid_path

        self.has_features = (self.feature_array is not None) and (self.feature_names is not None)

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

        # Set the device
        if device is not None:
            self.device = device

    def update_parameter_array(self,
                               parameters_to_remove: list = [],
                               ) -> None:
        '''
        Removes any parameter in self.provided_feature_parameters from the parameter array and parameter names
        ''' 

        for param in self.provided_feature_parameters + parameters_to_remove:
            if param in self.parameter_names:
                print(f"Removing parameter {param} from parameter array and names.")
                index = list(self.parameter_names).index(param)
                self.fitted_parameter_array = np.delete(self.parameter_array, index, axis=1)
                self.fitted_parameter_names = np.delete(self.parameter_names, index)

    @classmethod
    def init_from_hdf5(cls, 
                       model_name: str,
                       hdf5_path: str,
                       ):

        """
        Initialize the SBI fitter from an HDF5 file.

        Args:
            hdf5_path: Path to the HDF5 file.

        Returns:
            An instance of the SBI_Fitter class.
        """

        # Needs to load the training data and parameters from HDF5 file.
        # Training data if unnormalized and not setup as correct features yet. 

        output = load_grid_from_hdf5(hdf5_path)

        raw_photometry_grid = output['photometry']
        raw_photometry_names = output['filter_codes']
        parameter_array = output['parameters'].T 
        parameter_names = output['parameter_names']
        raw_photometry_units = output['photometry_units']

        if 'supplementary_parameters' in output:
            supplementary_parameters = output['supplementary_parameters']
            supplementary_parameter_names = output['supplementary_parameter_names']
            supplementary_parameter_units = output['supplementary_parameter_units']

           
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
        )

    @classmethod
    def init_from_basis(cls,
                        basis: CombinedBasis,
    ):
        """
        Initialize the SBI fitter from a basis.

        Args:
            basis: The basis to be used for fitting.

        Returns:
            An instance of the SBI_Fitter class.
        """
        # Needs to load the training data and parameters from the basis
        pass

    def simulator(self, index: int) -> np.ndarray:
        """
        Simulate the model for a given index.

        Args:
            index: The index of the model to be simulated.

        Returns:
            The simulated model.
        """
        # This function should return the simulated model for the given index
        # For now, we will just return the raw photometry grid
        return self.raw_photometry_grid[index]
        
    def create_feature_array_from_raw_photometry(self,
                                                normalize_method: str = 'mUV',
                                                extra_features: list = ['redshift'],
                                                normed_flux_units: str = 'AB',
                                                normalization_unit: str = 'AB',
                                                verbose: bool = True,
                                                ) -> np.ndarray:
        """
        Create a feature array from the raw photometry grid.
        Args:
            normalize_method: The method to normalize the photometry. At the moment only the names of the filters, names
                of supplementary parameters, or None are accepted. 
                This will be used to normalize the fluxes in the feature array.
            extra_features: Any extra features to be added. These should be
                generally written as functions of filter codes. E.g. NIRCam.F090W - NIRCam.F115W color as
                a feature would be written as ['F090W - F115W']. They can also be parameters in the parameter grid,
                in which case they won't be predicted by the model. 
            normed_flux_units: The units of the flux to normalize to. E.g. 'AB', 'nJy', etc. So when combined with normalize method,
                the fluxes for each filter will be relative to the normalization filter, in the given units. The overall
                normalization factor will be provided as well. 
            normalization_unit: The unit of the normalization factor. E.g. 'log10 nJy', 'nJy, AB', etc.

        Returns:
            The feature array and feature names.
        """
        if normed_flux_units == 'AB':
            phot = -2.5 * np.log10(unyt_array(self.raw_photometry_grid, units=self.raw_photometry_units).to('uJy').value) + 23.9
            norm_func = np.subtract
        else:
            phot = unyt_array(self.raw_photometry_grid, units=self.raw_photometry_units).to(normed_flux_units).value
            norm_func = np.divide
        
        # Normalize the photometry grid
        if normalize_method is not None:
            if normalize_method in self.raw_photometry_names:
                norm_index = list(self.raw_photometry_names).index(normalize_method)

                normalization_factor = phot[norm_index, :]
                norm_factor_original = self.raw_photometry_grid[norm_index, :]
                
                # Create a copy of the raw photometry names for consistent reference
                raw_photometry_names = np.array(self.raw_photometry_names)
                raw_photometry_names = np.delete(raw_photometry_names, norm_index)
                phot = np.delete(phot, norm_index, axis=0)

                # Create normalized photometry while preserving the original shape
                normed_photometry = norm_func(phot, normalization_factor)

                # Convert the normalization factor to the desired unit
                if normalization_unit.startswith('log10'):
                    log = True
                    normalization_unit_cleaned = normalization_unit.split(' ')[1]
                else:
                    log = False
                    normalization_unit_cleaned = normalization_unit

                if normalization_unit_cleaned == 'AB':
                    normalization_factor_converted = -2.5 * np.log10(unyt_array(norm_factor_original, units=self.raw_photometry_units).to('uJy').value) + 23.9
                else:
                    normalization_factor_converted = unyt_array(norm_factor_original, units=self.raw_photometry_units).to(normalization_unit_cleaned).value
                
                if log:
                    normalization_factor_converted = np.log10(normalization_factor_converted)
                    normalization_factor_converted[normalization_factor_converted == -np.inf] = 0.0
                    normalization_factor_converted[normalization_factor_converted == np.inf] = 0.0

            elif normalize_method in self.supplementary_parameter_names:
                norm_index = list(self.supplementary_parameter_names).index(normalize_method)
                norm_unit = self.supplementary_parameter_units[norm_index]
                normalization_factor = self.supplementary_parameters[norm_index, :]

                assert normalization_factor.shape[0] == self.raw_photometry_grid.shape[1], "Normalization factor should have the same shape as the photometry grid."
                assert norm_unit == self.raw_photometry_units, "Normalization factor should have the same units as the photometry grid."
                
                if normed_flux_units == 'AB':
                    normalization_factor_use = -2.5 * np.log10(unyt_array(normalization_factor, units=norm_unit).to('uJy').value) + 23.9
                else:
                    normalization_factor_use = normalization_factor


                normed_photometry = norm_func(phot, normalization_factor_use)

                # Convert the normalization factor to the desired unit
                if normalization_unit.startswith('log10'):
                    log = True
                    normalization_unit_cleaned = normalization_unit.split(' ')[1]
                else:
                    log = False
                    normalization_unit_cleaned = normalization_unit

                if normalization_unit_cleaned == 'AB':
                    normalization_factor_converted = -2.5 * np.log10(unyt_array(normalization_factor, units=norm_unit).to('uJy').value) + 23.9
                else:
                    normalization_factor_converted = unyt_array(normalization_factor, units=norm_unit).to(normalization_unit_cleaned).value

                raw_photometry_names = np.array(self.raw_photometry_names)

            else:
                raise NotImplementedError("Normalization method not implemented. Please use a filter name for normalization.")
        else:
            normed_photometry = self.raw_photometry_grid
            normalization_factor_converted = np.ones(normed_photometry.shape[1])
            raw_photometry_names = np.array(self.raw_photometry_names)

            # Convert the photometry to the desired units

            if normed_flux_units == 'AB':
                normed_photometry = -2.5 * np.log10(unyt_array(normed_photometry, units=self.raw_photometry_units).to('uJy').value) + 23.9
            else:
                normed_photometry = unyt_array(normed_photometry, units=self.raw_photometry_units).to(normed_flux_units).value
    
        if normed_flux_units == 'AB':
            # very small fluxes can blow up here and go to infinity. Set a maximum difference of some large amount - say 50
            # If not normalized, set minmum flux to 50 AB.
            # If normalized to some AB, set minimum normalized flux to 50 (e.g. norm value + 50).
            normed_photometry[normed_photometry > 50] = 50


        # Create the feature array
        # Photometry + extra features + normalization factor
        feature_array = np.zeros((
            len(raw_photometry_names) + len(extra_features) + 1,
            self.raw_photometry_grid.shape[1]
        ))


        # Fill the feature array with the normalized photometry
        feature_array[:len(raw_photometry_names), :] = normed_photometry
        
        # Add the normalization factor as the last column
        feature_array[-1, :] = normalization_factor_converted
        
        # Create the feature names
        nfeatures = feature_array.shape[0]
        feature_names = [''] * nfeatures
        
        # Add filter names
        for i in range(len(raw_photometry_names)):
            feature_names[i] = raw_photometry_names[i]
        
        # Add the normalization factor name
        feature_names[-1] = f'norm_{normalize_method}_{normalization_unit}'
        
        # Process extra features if any
        if len(extra_features) > 0:
            parser = FilterArithmeticParser()
            for i, feature in enumerate(extra_features):

                if feature in self.parameter_names:
                    self.provided_feature_parameters.append(feature)
                    index = list(self.parameter_names).index(feature)
                    feature_array[len(raw_photometry_names) + i, :] = self.parameter_array[:, index]
                    feature_names[len(raw_photometry_names) + i] = feature
                else:  
                    # Parse the feature expression
                    tokens = parser.tokenize(feature)
                    value = parser.evaluate(tokens, dict(zip(raw_photometry_names, normed_photometry.T)))
                    feature_array[len(raw_photometry_names) + i, :] = value

                    # Add the feature name
                    feature_names[len(raw_photometry_names) + i] = feature

        
        assert '' not in feature_names, "Feature names should not be empty. Please check the extra features."

        self.feature_array = feature_array.astype(np.float32).T
        self.feature_names = feature_names
        self.feature_units = [normed_flux_units] * len(raw_photometry_names) + [None] * len(extra_features) + [normalization_unit]
        self.has_features = True

        if verbose:
            print('---------------------------------------------')
            print(f'Features: {self.feature_array.shape[0]} features over {self.feature_array.shape[1]} samples')
            print('---------------------------------------------')
            print('Feature: Min - Max')
            print('---------------------------------------------')

            for pos, feature_name in enumerate(feature_names):
                print(f'{feature_name}: {np.min(feature_array[pos]):.3f} - {np.max(feature_array[pos]):.3f} {self.feature_units[pos]}')
            print('---------------------------------------------')

        self.update_parameter_array()

        return feature_array, feature_names

    def split_dataset(self,
                      train_fraction: float = 0.8,
                        random_seed: int = None,
                        ) -> tuple:
        """
        Split the dataset into training and testing sets.
        Args:
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

        print(f"Splitting dataset with {num_samples} samples into training and testing sets with {train_fraction:.2f} train fraction.")
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        train_size = int(num_samples * train_fraction)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        return train_indices, test_indices
    
    def create_priors(self,
                      override_prior_ranges: dict = {},
                      prior = ili.utils.Uniform,
                      ):
        
        """
        Create the priors for the parameters. By default we will use the range of the parameters in the grid.
        If override_prior_ranges is provided for a given parameter, then that will be used instead.
        Args:
            override_prior_ranges: A dictionary containing the prior ranges for the parameters.
            prior: The prior distribution to be used.
        Returns:
            A prior object. 

        """
        if not self.has_features:
            raise ValueError("Feature array not created. Please create the feature array first.")
        
        if self.fitted_parameter_array is None:
            raise ValueError("Parameter grid not created. Please create the parameter grid first.")
        if self.fitted_parameter_names is None:
            raise ValueError("Parameter names not created. Please create the parameter names first.")
        
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

        low = torch.tensor(low, dtype=torch.float32, device=self.device)
        high = torch.tensor(high, dtype=torch.float32, device=self.device)

        # Create the priors
        param_prior = prior(low=low, high=high, device=self.device)

        return param_prior

    def optimize_sbi(self,
                    suggested_hyperparameters: dict,
                    fixed_hyperparameters: dict = None,
                    n_trials: int = 100,
                    n_jobs: int = 1,
                    random_seed: int = None,
                    ) -> None:
        '''
        Use Optuna to optimize the SBI model hyperparameters.
        
        '''
        
        if not self.has_features:
            raise ValueError("Feature array not created. Please create the feature array first.")
        
        if self.fitted_parameter_array is None:
            raise ValueError("Parameter grid not created. Please create the parameter grid first.")
        
        if self.fitted_parameter_names is None:
            raise ValueError("Parameter names not created. Please create the parameter names first.")
        
    def run_evaluate_sbi(self,
                        train_indices: np.ndarray,
                        test_indices: np.ndarray,
                        **arguments
                        ) -> tuple:
        
        posterior, stats = self.run_single_sbi(
            train_indices=train_indices,
            test_indices=test_indices,
            set_self = False,
            plot=False,
            **arguments
        )
              
    def run_single_sbi(self,
                train_test_fraction: float = 0.8,
                random_seed: int = None,
                backend: str = 'sbi',
                engine: Union[str, List[str]] = 'NPE',
                train_indices: np.ndarray = None,
                test_indices: np.ndarray = None,
                n_nets: int = 1,
                model_type: Union[str,  List[str]] = 'mdn',
                hidden_features: Union[int, List[int]] = 50,
                num_components: Union[int, List[int]]  = 4,
                num_transforms: Union[int, List[int]] = 4,
                training_batch_size: int = 64,
                learning_rate: float = 1e-4,
                validation_fraction: float = 0.2,
                stop_after_epochs: int = 15,
                clip_max_norm: float = 5.0,
                save_model: bool = True,
                verbose: bool = True,
                prior_method: str = 'ili',
                out_dir: str = f'{code_path}/models/',
                plot: bool = True,
                name_append: str = 'timestamp',
                feature_scalar = StandardScaler,
                target_scalar = StandardScaler,
                set_self: bool = True,
                ) -> tuple:
        """
        Run a single SBI training instance.

        Args:
            train_indices: Indices of the training set.
            test_indices: Indices of the test set. If None, no test set is used.
            model_type: Type of model to use. Either 'mdn' or 'maf'.
            hidden_features: Number of hidden features in the neural network.
            num_components: Number of components in the mixture density network.
            num_transforms: Number of transforms in the masked autoregressive flow.
            training_batch_size: Batch size for training.
            learning_rate: Learning rate for the optimizer.
            validation_fraction: Fraction of the training set to use for validation.
            stop_after_epochs: Number of epochs without improvement before stopping.
            clip_max_norm: Maximum norm for gradient clipping.
            save_model: Whether to save the trained model.
            verbose: Whether to print verbose output.
            prior_method: Method to create the prior. Either 'manual' or 'ili'.
            feature_scalar: Scaler for the features.
            target_scalar: Scaler for the targets.
            out_dir: Directory to save the model.
            plot: Whether to plot the diagnostics.
            name_append: Append to the model name.
            set_self: Whether to set the self object with the trained model.

        Returns:
            A tuple containing the posterior distribution and training statistics.
        """
        if not self.has_features:
            raise ValueError("Feature array not created. Please create the feature array first.")
        
        if self.fitted_parameter_array is None:
            raise ValueError("Parameter grid not created. Please create the parameter grid first.")
        
        out_dir = os.path.join(os.path.abspath(out_dir), self.name)

        if self.fitted_parameter_names is None:
            raise ValueError("Parameter names not created. Please create the parameter names first.")
        
        if name_append == 'timestamp':
            name_append = f'_{self._timestamp}'
        
        if os.path.exists(f'{out_dir}/{self.name}{name_append}_single_sbi_params.pkl') and save_model:
            print('Model with same name already exists. Please change the name of this model or delete the existing one.')
            return None

        if train_indices is None:
            train_indices, test_indices = self.split_dataset(train_fraction=train_test_fraction, random_seed=random_seed)

        # Prepare data

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

        if prior_method == 'manual':
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
            
            prior_low = torch.tensor(y_min - 3 * y_std, dtype=torch.float32, device=self.device)
            prior_high = torch.tensor(y_max + 3 * y_std, dtype=torch.float32, device=self.device)
            prior = ili.utils.Uniform(low=prior_low, high=prior_high)
        elif prior_method == 'ili':
            # Create the prior using the parameter array
            prior = self.create_priors()
            if set_self:
                self._prior = prior
            X_scaled = X_train
            y_scaled = y_train
        else:
            raise ValueError("Invalid prior method. Use 'manual' or 'ili'.")
        
        nets = []
        ensemble_model_types = []
        ensemble_model_args = []
        for i in range(n_nets):
            # Configure model
            model_args = {}
            model_type = model_type if isinstance(model_type, str) else model_type[i]
            eng = engine if isinstance(engine, str) else engine[i]

            if model_type == 'mdn':
                model_args = {
                    "hidden_features": hidden_features[i] if isinstance(hidden_features, list) else hidden_features,
                    "num_components": num_components[i] if isinstance(num_components, list) else num_components,
                }
            elif model_type in ['maf', 'nsf', 'made']:
                model_args = {
                    "hidden_features": hidden_features[i] if isinstance(hidden_features, list) else hidden_features,
                    "num_transforms": num_transforms[i] if isinstance(num_transforms, list) else num_transforms,
                }
            else:
                raise ValueError(f"Unknown model type: {model_type}. Use 'mdn' or 'maf'.")
            

            # Create neural network
            net = self._create_network(model_type, model_args, engine=eng)
            nets.append(net)
            ensemble_model_types.append(model_type)
            ensemble_model_args.append(model_args)

                    
        # Setup trainer arguments
        train_args = {
            "training_batch_size": training_batch_size,
            "learning_rate": learning_rate,
            "validation_fraction": validation_fraction,
            "stop_after_epochs": stop_after_epochs,
            "clip_max_norm": clip_max_norm
        }
        
       
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
        
        # Create data loader
        loader = NumpyLoader(X_scaled, y_scaled)
        
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
            self.posteriors = posteriors
            self.stats = stats
            self._train_args = train_args
            self._ensemble_model_types = ensemble_model_types
            self._ensemble_model_args = ensemble_model_args
            self._train_indices = train_indices
            self._test_indices = test_indices
            self._train_fraction = train_test_fraction

        '''
        # Test the model if test indices are provided
        if test_indices is not None:
            test_metrics = self._evaluate_model(
                posteriors, 
                self.feature_array[test_indices], 
                self.fitted_parameter_array[test_indices],
            )
            stats["test_metrics"] = test_metrics
        '''

        # Save the params with the model if needed
        if save_model:
            param_dict = {
                "ensemble_model_types": ensemble_model_types,
                "ensemble_model_args": ensemble_model_args,
                "n_nets": n_nets,
                "feature_names": self.feature_names,
                "fitted_parameter_names": self.fitted_parameter_names,
                "train_args": train_args,
                "stats": stats,
                "test_indices": test_indices,
                "train_indices": train_indices,
                "train_fraction": train_test_fraction,
                "timestamp": self._get_timestamp(),
                "prior": self._prior,
                "feature_array": self.feature_array,
                "parameter_array": self.fitted_parameter_array,
            }
            dump(param_dict, f"{out_dir}/{self.name}{name_append}_params.pkl", compress=3)
        
        if plot:
            self.plot_diagnostics(X_train=X_scaled, y_train=y_scaled, 
                                  X_test=X_test,
                                  y_test=y_test,
                                  plots_dir=out_dir)

        return posteriors, stats

    def _create_feature_scaler(self, X: np.ndarray) -> object:
        """
        Create and fit a scaler for the features.
        
        Args:
            X: Feature array.
            
        Returns:
            A fitted scaler for the features.
        """
       
        scaler = self._feature_scalar
        scaler.fit(X)
        return scaler

    def _create_target_scaler(self, y: np.ndarray) -> object:
        """
        Create and fit a scaler for the targets.
        
        Args:
            y: Target array.
            
        Returns:
            A fitted scaler for the targets.
        """
        scaler = self._target_scalar
        scaler.fit(y)
        return scaler

    def _create_network(self, model_type: str, model_args: dict, engine='NPE') -> nn.Module:
        """
        Create a neural network for the SBI model.
        
        Args:
            model_type: Type of model to use. Either 'mdn' or 'maf'.
            model_args: Arguments for the model.
            
        Returns:
            A neural network.
        """
        # Import the necessary function from ili
    
        return ili.utils.load_nde_sbi(
            engine=engine, 
            model=model_type, 
            **model_args
        )
    
    def sample_posterior(self,
                         X_test: np.ndarray,
                         y_test: np.ndarray,
                         posteriors: object = None,
                         num_samples: int = 1000,
    ) -> np.ndarray:
        
        if posteriors is None:
            posteriors = self.posteriors


        # Draw samples from the posterior
        samples = np.zeros((len(X_test), num_samples, y_test.shape[1]))
        for i in range(len(X_test)):
            x = torch.tensor(X_test[i], dtype=torch.float32, device=self.device)
            posterior_samples = posteriors.sample(
                x=x, 
                sample_shape=(num_samples,),
                show_progress_bars=False
            ).cpu().numpy()
            samples[i] = posterior_samples

        return samples
                     
    def _evaluate_model(self, 
                    posteriors: list,
                    X_test: np.ndarray, 
                    y_test: np.ndarray,
                    num_samples: int = 1000) -> dict:
        """
        Evaluate the trained model on test data.
        
        Args:
            X_test: Test feature array.
            y_test: Test target array.
            X_scaler: Scaler for the features (if used).
            y_scaler: Scaler for the targets (if used).
            num_samples: Number of samples to draw from the posterior.
            
        Returns:
            A dictionary of evaluation metrics.
        """
        
        # Draw samples from the posterior
        samples = self.sample_posterior(X_test, y_test, num_samples=num_samples, posteriors=posteriors)
       
        
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

    def plot_diagnostics(self,
                        X_train: np.ndarray = None,
                        y_train: np.ndarray = None,
                        X_test: np.ndarray = None,
                        y_test: np.ndarray = None,
                        plots_dir: str = f'{code_path}/models/name/plots/',
                        ) -> None:
        """
        Plot the diagnostics of the SBI model.
        """

        plots_dir = plots_dir.replace('name', self.name)
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        if X_train is None or y_train is None:
            if hasattr(self, '_X_train') and hasattr(self, '_y_train'):
                X_train = self._X_train
                y_train = self._y_train
            else:
                raise ValueError("X_train and y_train must be provided or set in the object.")
        if X_test is None or y_test is None:
            if hasattr(self, '_X_test') and hasattr(self, '_y_test'):
                X_test = self._X_test
                y_test = self._y_test
            else:
                raise ValueError("X_test and y_test must be provided or set in the object.")
        
        # Plot the loss
        self.plot_loss(self.stats, plots_dir=plots_dir)

        # Plot the posterior
        self.plot_posterior(
            X=X_train,
            y=y_train,
            plot_dir=plots_dir,
        )

        # Plot the coverage
        self.plot_coverage(X=X_test, y=y_test, plot_dir=plots_dir)

    def plot_loss(self,
                  summaries: list,
                  plots_dir: str = f'{code_path}/models/name/plots/',
                  ) -> None:
        """
        Plot the loss of the SBI model.
        """

        plots_dir = plots_dir.replace('name', self.name)

        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        # plot train/validation loss
        fig, ax = plt.subplots(1, 1, figsize=(6,4))

        for i, m in enumerate(summaries):
            ax.plot(m['training_log_probs'], ls='-', label=f"{i}_train")
            ax.plot(m['validation_log_probs'], ls='--', label=f"{i}_val")
        ax.set_xlim(0)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Log probability')
        ax.legend()

        if not os.path.exists(f'{plots_dir}/{self.name}'):
            os.makedirs(f'{plots_dir}/{self.name}')

        fig.savefig(f'{plots_dir}/loss.png', dpi=300)

    def plot_posterior(self,
                    ind: int = 'random',
                    X: np.ndarray = None,
                    y: np.ndarray = None,
                    seed: int = None,
                    num_samples: int = 1000,
                    sample_method: str = 'direct',
                    plots_dir: str = f'{code_path}/models/name/plots/',
                    plot_kwargs=dict(fill=True),
                    **kwargs: dict,
                    ) -> None:
        """
        Plot the posterior of the SBI model.
        """

        
        plots_dir = plots_dir.replace('name', self.name)

        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        if X is None or y is None:
            raise ValueError("X and y must be provided to plot the posterior.")
        
        if ind == 'random':
            if seed is not None:
                np.random.seed(seed)
            ind = np.random.randint(0, X.shape[0])



        # use ltu-ili's built-in validation metrics to plot the posterior for this point
        metric = PlotSinglePosterior(
            num_samples=num_samples, sample_method=sample_method,
            labels=self.fitted_parameter_names,
            out_dir=plots_dir,
        )
        fig = metric(
            posterior=self.posteriors,
            x_obs = X[ind], theta_fid=y[ind],
            plot_kws=plot_kwargs,
            signature=f'{self.name}_{ind}_',
            **kwargs,
        )
        return fig

    def plot_posterior_samples(self):
        """
        Plot the posterior samples of the SBI model.
        """
        pass

    def plot_posterior_predictions(self):
        """
        Plot the posterior predictions of the SBI model.
        """
        pass
    
    def calculate_PIT(self,
                    X: np.ndarray,
                    y: np.ndarray,
                    num_samples: int = 1000) -> np.ndarray:
        """
        Calculate the probability integral transform (PIT) for the samples
        produced by the regressor.

        Parameters
        ----------
        X : 2-dimensional array of shape (num_samples, n_features)
            Feature array.
        y : 1-dimensional array of shape (num_samples,)
            Target variable.

        Returns
        -------
        pit : 1-dimensional array
            The PIT values.
        """

        samples = self.sample_posterior(X, y, num_samples=num_samples)

        ytf = self.transform_target(y)

        pit = np.empty(len(y))
        for i in range(len(y)):
            pit[i] = np.mean(samples[i] < ytf[i])

        pit = np.sort(pit)
        pit /= pit[-1]

        return pit

    def log_prob(self,
                    X: np.ndarray,
                    y: np.ndarray,
                ) -> np.ndarray:
        
        lp = np.empty((len(X), len(self.fitted_parameter_names)))

        for i in range(len(X)):
            x = torch.tensor([X[i]], dtype=torch.float32, device=self.device)
            theta = torch.tensor([y[i]], dtype=torch.float32, device=self.device)
            lp[i, :] = self._posterior.log_prob(x=x, theta=theta,
                                             norm_posterior=True)
            
        return lp

    def plot_latent_residual(self):
        """
        Plot the latent residual of the SBI model.
        """
        pass
    
    def plot_coverage(self,
                    X: np.ndarray,
                    y: np.ndarray,
                    posteriors: Union[str, int] = 'total',
                    num_samples: int = 1000,
                    sample_method: str = 'direct',
                    plot_list = ["coverage", "histogram", "predictions", "tarp", "logprob"],
                    plots_dir: str = f'{code_path}/models/name/plots/',
                      ) -> None:

        """
        Plot the coverage of the SBI model.
        Args:
            X: Feature array.
            y: Target array.
            posteriors: which posterior to plot from ensemble. Either 'total', 'seperate', or an index.
            num_samples: Number of samples to draw from the posterior.
            sample_method: Method to sample from the posterior. Either 'direct' or 'rejection'.
            plot_list: List of plots to create. 
            plot_dir: Directory to save the plots.
        """

        plots_dir = plots_dir.replace('name', self.name)

        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        metric = PosteriorCoverage(
            num_samples=num_samples, sample_method=sample_method, 
            labels=self.fitted_parameter_names,
            plot_list = plot_list,
            out_dir=plots_dir,
        )

        fig = None
        if posteriors == 'seperate':
            for i in range(len(self.posteriors)):
                fig = metric(
                    posterior=self.posteriors[i],
                    x=X, theta=y, fig=fig, 
                )
        else:
            fig = metric(
                posterior=self.posteriors if posteriors == 'total' else self.posteriors[posteriors],
                x=X, theta=y, # X and y are the feature and target arrays 
            )

        return fig

    def run_validation_from_file(self,
                        validation_file: str,
                        plot_dirs: str = f'{code_path}/plots/',
                        ) -> None:
        """
        Run the validation from a file.
        """
        posterior = ValidationRunner.load_posterior_sbi(validation_file)

    @property
    def validation_log_probs(self):
        """
        Validation set log-probability of each epoch for each net.

        Returns
        -------
        list of 1-dimensional arrays
        """
        if self.stats is None:
            raise RuntimeError("The regressor has not been fitted yet.")

        return [stat["validation_log_probs"] for stat in self.stats]

    @property
    def training_log_probs(self):
        """
        Training set log-probability of each epoch for each net.

        Returns
        -------
        list of 1-dimensional array
        """
        if self.stats is None:
            raise RuntimeError("The regressor has not been fitted yet.")

        return [stat["training_log_probs"] for stat in self.stats]
    
    def load_model_from_pkl(self,
                        model_file: str,
                        set_self: bool = True,
                        ) -> None:
        """
        Load the model from a pickle file.
        Args:
            model_file: Path to the pickle file.
        """
        with open(model_file, 'rb') as f:
            posteriors = load(f)
        #
        stats = model_file.replace('posterior.pkl', 'summary.json')
        if os.path.exists(stats):
            with open(stats, 'r') as f:
                stats = json.load(f)

            if set_self:
                self.stats = stats

        else:
            stats = None
            print(f"Warning: No summary file found for {model_file}.")
            
        if set_self:
            self.posteriors = posteriors

        return posteriors, stats

    @property
    def _timestamp(self):
        """
        Get the current date and time as a string.
        """
        return datetime.now().strftime("%Y%m%d_%H%M%S")





class FilterArithmeticParser:
    """
    Stolen from my own code - https://github.com/tHarvey303/BD-Finder/blob/main/src/BDFit/StarFit.py

    Parser for filter arithmetic expressions.
    Supports operations like:
    - Basic arithmetic: +, -, *, /
    - Parentheses for grouping
    - Constants and coefficients
    
    Examples:
        "F356W"                    -> single filter
        "F356W + F444W"           -> filter addition
        "2 * F356W"               -> coefficient multiplication
        "(F356W + F444W) / 2"     -> average of filters
        "F356W - 0.5 * F444W"     -> weighted subtraction
    """

    def __init__(self):
        self.operators = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv
        }

        # Regular expression pattern for tokenizing
        self.pattern = r'(\d*\.\d+|\d+|[A-Za-z]\d+[A-Za-z]+|\+|\-|\*|\/|\(|\))'

    def tokenize(self, expression: str) -> List[str]:
        """Convert string expression into list of tokens."""
        tokens = re.findall(self.pattern, expression)
        return [token.strip() for token in tokens if token.strip()]

    def is_number(self, token: str) -> bool:
        """Check if token is a number."""
        try:
            float(token)
            return True
        except ValueError:
            return False

    def is_filter(self, token: str) -> bool:
        """Check if token is a filter name."""
        return bool(re.match(r'^[A-Za-z]\d+[A-Za-z]+$', token))

    def evaluate(self, tokens: List[str], filter_data: Dict[str, Union[float, np.ndarray]]) -> Union[float, np.ndarray]:
        """
        Evaluate a list of tokens using provided filter data.
        
        Args:
            tokens: List of tokens from the expression
            filter_data: Dictionary mapping filter names to their values
            
        Returns:
            Result of the arithmetic operations
        """
        output_stack = []
        operator_stack = []

        precedence = {'+': 1, '-': 1, '*': 2, '/': 2}

        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token == '(':
                operator_stack.append(token)

            elif token == ')':
                while operator_stack and operator_stack[-1] != '(':
                    self._apply_operator(operator_stack, output_stack, filter_data)
                operator_stack.pop()  # Remove '('

            elif token in self.operators:
                while (operator_stack and operator_stack[-1] != '(' and
                       precedence.get(operator_stack[-1], 0) >= precedence[token]):
                    self._apply_operator(operator_stack, output_stack, filter_data)
                operator_stack.append(token)

            else:  # Number or filter name
                if self.is_number(token):
                    value = float(token)
                elif self.is_filter(token):
                    if token not in filter_data:
                        raise ValueError(f"Filter {token} not found in provided data")
                    value = filter_data[token]
                else:
                    raise ValueError(f"Invalid token: {token}")
                output_stack.append(value)

            i += 1

        while operator_stack:
            self._apply_operator(operator_stack, output_stack, filter_data)

        if len(output_stack) != 1:
            raise ValueError("Invalid expression")

        return output_stack[0]

    def _apply_operator(self, operator_stack: List[str], output_stack: List[Union[float, np.ndarray]],
                       filter_data: Dict[str, Union[float, np.ndarray]]) -> None:
        """Apply operator to the top two values in the output stack."""
        operator = operator_stack.pop()
        b = output_stack.pop()
        a = output_stack.pop()
        output_stack.append(self.operators[operator](a, b))

    def parse_and_evaluate(self, expression: str, filter_data: Dict[str, Union[float, np.ndarray]]) -> Union[float, np.ndarray]:
        """
        Parse and evaluate a filter arithmetic expression.
        
        Args:
            expression: String containing the filter arithmetic expression
            filter_data: Dictionary mapping filter names to their values
            
        Returns:
            Result of evaluating the expression
        """
        tokens = self.tokenize(expression)
        return self.evaluate(tokens, filter_data)

# TODO:
# Fix issue with photometry range.
# Add more validation metrics. Consider if more manual normalization is needed e.g. with StandardScaler.
# Allow loading in pickles. 
# Setup better naming/validation for the models.
# Ensure we are saving everything we need to - e.g. training data, validation data, hyperparameters, results, etc.
# Add hyperparameter optimization using Optuna.
# Build out other plotting functions. Consider what other functions are needed.
# Method to test model on single data point.
# Investigate bayesian model comparison methods. 