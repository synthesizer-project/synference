import sys
import os
import h5py
import inspect
import scipy.stats
import warnings
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import synthesizer
from synthesizer.parametric import Galaxy
from synthesizer.emission_models import PacmanEmission, TotalEmission, EmissionModel, IntrinsicEmission
from synthesizer.emission_models.attenuation import PowerLaw
from synthesizer.emissions import plot_spectra
from synthesizer.emission_models.dust.emission import Greybody
from synthesizer.grid import Grid
from synthesizer.parametric import SFH, Stars, ZDist
from synthesizer.instruments import Instrument, FilterCollection, Filter
from synthesizer.particle.stars import sample_sfzh
from synthesizer.conversions import fnu_to_lnu, lnu_to_fnu
from typing import Dict, Any, List, Tuple, Union, Optional, Type
from abc import ABC, abstractmethod
from collections import defaultdict
import copy
from scipy.stats import uniform, loguniform
from unyt import unyt_array, unyt_quantity, erg, cm, s, Angstrom, um, Hz, m, nJy, K, Msun, Myr, yr, Unit, kg, Mpc, dimensionless, Jy, uJy
from astropy.cosmology import Planck18, Cosmology, z_at_value
from synthesizer.emission_models.attenuation import Inoue14
import astropy.units as u
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from tqdm import tqdm
from synthesizer.pipeline import Pipeline
import matplotlib.patheffects as PathEffects
from scipy.stats import qmc

warnings.filterwarnings('ignore')

def calculate_muv(galaxy, cosmo=Planck18):
    z = galaxy.redshift
    tophats = {
        "MUV": {"lam_eff": 1500 *  Angstrom, "lam_fwhm": 100 * Angstrom},
    }
    
    filter = FilterCollection(tophat_dict=tophats, verbose=False)

    phots = {}

    for key in list(galaxy.stars.spectra.keys()):
        lnu = galaxy.stars.spectra[key].get_photo_lnu(filter).photo_lnu[0]
        phot = lnu_to_fnu(lnu, cosmo=cosmo, redshift=z)
        phots[key] = phot

    return phots

def generate_sfh_grid(
    sfh_type: Type[SFH.Common],
    sfh_priors: Dict[str, Dict[str, Any]],
    redshift: Union[Dict[str, Any], float], 
    max_redshift: float = 15,
    cosmo: Type[Cosmology] = Planck18,
) -> Tuple[List[Type[SFH.Common]], np.ndarray]:
    """
    Generate a grid of star formation histories based on prior distributions for redshift and SFH parameters.
    
    This function creates a grid of SFH models by combining possible parameter values, which can
    depend explicitly on the redshift. It first draws redshifts, calculates maximum stellar ages at each
    redshift, and then creates parameter combinations within valid ranges.
    
    Parameters
    ----------
    sfh_type : Type[SFH]
        The star formation history class to instantiate
    sfh_priors : Dict[str, Dict[str, Any]]
        Dictionary of prior distributions for SFH parameters. Each parameter should have:
        - 'prior': scipy.stats distribution
        - 'min': minimum value
        - 'max': maximum value
        - 'size': number of samples to draw
        - 'units': astropy unit (optional)
        - 'name': parameter name for SFH constructor (optional, defaults to the key)
        - 'depends_on': special flag if parameter depends on redshift ('max_redshift')
    redshift : Union[Dict[str, Any], float]
        Either a single redshift value or a dictionary with:
        - 'prior': scipy.stats distribution
        - 'min': minimum redshift
        - 'max': maximum redshift 
        - 'size': number of redshift samples
    max_redshift : float, optional
        Maximum possible redshift to consider for age calculations, by default 15
    cosmo : Type[Cosmology], optional
        Cosmology to use for age calculations, by default Planck18
        
    Returns
    -------
    Tuple[List[SFH], np.ndarray]
        - List of SFH objects with parameters drawn from the priors
        - Array of parameter combinations, where the first column is redshift followed by SFH parameters
    
    Notes
    -----
    For parameters that depend on redshift (marked with 'depends_on': 'max_redshift'), 
    the maximum allowed value will be dynamically adjusted based on the age of the universe
    at that redshift.
    """

    # Draw redshifts
    if isinstance(redshift, dict):
        redshifts = redshift['prior'].rvs(
            size=int(redshift['size']), 
            loc=redshift['min'], 
            scale=redshift['max'] - redshift['min']
        )
    else:
        redshifts = np.array([redshift])

    # Calculate maximum ages at each redshift
    max_ages = (cosmo.age(redshifts) - cosmo.age(max_redshift)).to(u.Myr).value

    # Prepare parameter arrays for each parameter
    param_arrays = [redshifts]
    param_names = ["redshift"]  # Track parameter names for later use
    
    # Process each SFH parameter
    for key in sfh_priors.keys():
        param_data = sfh_priors[key]
        param_size = int(param_data['size'])
        param_min = param_data['min']
        param_max = param_data['max']
        
        # If parameter depends on redshift, adjust for each redshift
        if 'depends_on' in param_data and param_data['depends_on'] == 'max_redshift':
            # Create parameter values for each redshift
            all_values = []
            for z, max_age in zip(redshifts, max_ages):
                # Adjust maximum value based on the age of the universe at this redshift
                adjusted_max = min(param_max, max_age)
                if 'units' in param_data and param_data['units'] is not None:
                    adjusted_max = adjusted_max * param_data['units']
                    if hasattr(adjusted_max, 'value'):
                        adjusted_max = adjusted_max.value
                
                # Draw values for this parameter at this redshift
                values = param_data['prior'].rvs(
                    size=param_size // len(redshifts),  # Distribute samples across redshifts
                    loc=param_min,
                    scale=adjusted_max - param_min
                )
                all_values.append(values)
            
            # Combine values from all redshifts
            param_values = np.concatenate(all_values)
        else:
            # Parameter doesn't depend on redshift, draw from the same distribution for all
            param_values = param_data['prior'].rvs(
                size=param_size,
                loc=param_min,
                scale=param_max - param_min
            )
        
        
        param_arrays.append(param_values)
        param_names.append(param_data.get('name', key))  # Use specified name or key

    # Create parameter combinations using meshgrid
    mesh_arrays = np.meshgrid(*param_arrays, indexing='ij')
    param_combinations = np.stack([arr.flatten() for arr in mesh_arrays], axis=1)
    
    # Create SFH objects for each parameter combination
    sfhs = []
    for params in tqdm(param_combinations):
        z = params[0]  # First parameter is always redshift
        max_age = (cosmo.age(z) - cosmo.age(max_redshift)).to(u.Myr).value
        
        # Create parameter dictionary for SFH constructor
        sfh_params = {
            param_names[i+1]: params[i+1] for i in range(len(param_names)-1)
        }

        # Apply units if not None
        for key, value in sfh_params.items():
            if 'units' in sfh_priors[key] and sfh_priors[key]['units'] is not None:
                sfh_params[key] = value * sfh_priors[key]['units']
        
        # Add max_age parameter
        sfh_params['max_age'] = max_age * Myr
        
        # Create and append SFH instance
        sfh = sfh_type(**sfh_params)
        sfhs.append(sfh)
    
    return sfhs, param_combinations

def generate_metallicity_distribution(
    zmet_dist: ZDist.Common,
    zmet:dict = {'prior':'loguniform', 'min':-3, 'max':0.3, 'size':6, 'units':None},
):
    """
    Generate a grid of metallicity distributions based on prior distributions.
    
    Parameters
    ----------
    zmet_dist : Type[ZDist]
        The metallicity distribution class to instantiate. E.g., ZDist.DeltaConstant or ZDist.Normal
    z : dict
        Dictionary of prior distributions for metallicity parameters. Each parameter should have:
        - 'prior': scipy.stats distribution
        - 'min': minimum value
        - 'max': maximum value
        - 'size': number of samples to draw
        - 'units': unyt unit (optional)
        - 'name': parameter name for constructor (optional, defaults to the key)
    """

    # choose values based on prior. Can provide metallicity as either metallicity or log10metallicity

    if isinstance(zmet, dict):
        zmet_values = zmet['prior'].rvs(
            size=int(zmet['size']), 
            loc=zmet['min'], 
            scale=zmet['max'] - zmet['min']
        )

    else:
        zmet_values = np.array([zmet])

    # Create parameter combinations using meshgrid
    zmet_array = np.meshgrid(zmet_values, indexing='ij')
    zmet_combinations = np.stack([arr.flatten() for arr in zmet_array], axis=1)

    # Create ZDist objects for each parameter combination
    zmet_dists = []
    for params in tqdm(zmet_combinations):
        # Create parameter dictionary for ZDist constructor
        zmet_params = {
            'zmet': params[0]
        }

        # Apply units if not None
        if 'units' in zmet and zmet['units'] is not None:
            zmet_params['zmet'] = params[0] * zmet['units']
        
        # Create and append ZDist instance
        zdist = zmet_dist(**zmet_params)
        zmet_dists.append(zdist)

def generate_emission_models(emission_model: Type[EmissionModel], 
                            varying_params: dict,
                            grid: Grid,
                            fixed_params: dict = None,
                            ):
    """
    Generate a grid of emission models based on varying parameters.
    Parameters
    ----------

    emission_model : Type[EmissionModel]
        The emission model class to instantiate. E.g., TotalEmission or PacmanEmission

    varying_params : dict
        Dictionary of varying parameters for the emission model. Each parameter should have:
        - 'prior': scipy.stats distribution
        - 'min': minimum value
        - 'max': maximum value
        - 'size': number of samples to draw
        - 'units': unyt unit (optional)
        - 'name': parameter name for constructor (optional, defaults to the key)
    grid : Grid
        Grid object containing the SPS grid.
    fixed_params : dict, optional
        Dictionary of fixed parameters for the emission model. Each parameter should have:
        - 'value': fixed value
        - 'units': unyt unit (optional)
        - 'name': parameter name for constructor (optional, defaults to the key)
    """

    # Create parameter combinations using meshgrid
    varying_param_arrays = []
    for key, param_data in varying_params.items():
        args = {}
        args.update(param_data)
        if 'min' in args:
            args['loc'] = param_data['min']
        if 'max' in args:
            args['scale'] = param_data['max'] - param_data['min']
    
        if 'units' in args:
            args.pop('units')
        if 'name' in args:
            args.pop('name')
        if 'prior' in args:
            args.pop('prior')


        # Draw values for this parameter
        param_values = param_data['prior'].rvs(
            **args,
        )
        
        varying_param_arrays.append(param_values)

    # Create parameter combinations using meshgrid
    varying_param_mesh = np.meshgrid(*varying_param_arrays, indexing='ij')
    varying_param_combinations = np.stack([arr.flatten() for arr in varying_param_mesh], axis=1)
    # Create emission model objects for each parameter combination

    emission_models = []
    out_params = {}
    for i, params in tqdm(enumerate(varying_param_combinations)):
        # Create parameter dictionary for emission model constructor
        emission_params = {
            key: params[j] for j, key in enumerate(varying_params.keys())
        }


        # Apply units if not None
        for key, value in emission_params.items():
            if 'units' in varying_params[key] and varying_params[key]['units'] is not None:
                emission_params[key] = value * varying_params[key]['units']
            if key not in out_params:
                out_params[key] = []
            out_params[key].append(emission_params[key])


        # store value of varying parameter(s) in dictionary for this emission model

    
        
        emission_params.update(fixed_params)
        
        # Create and append emission model instance
        emission_model_instance = emission_model(grid=grid, **emission_params)
        emission_models.append(emission_model_instance)


    return emission_models, out_params

def draw_from_hypercube(N: int = 1e6,
                        param_ranges: dict ={}, 
                        model: Type[qmc.QMCEngine] = qmc.LatinHypercube,
                        rng: Optional[np.random.Generator] = None):
    """
    Draw N samples from a hypercube defined by the parameter ranges.
    
    Parameters
    ----------
    N : int
        Number of samples to draw.
    param_ranges : dict, optional
        Dictionary where keys are parameter names and values are tuples (min, max) defining the ranges.
    model : Type[qmc], optional
        The sampling model to use, by default LatinHypercube.
    rng : Optional[np.random.Generator], optional

    Returns
    -------
    np.ndarray
        Array of shape (N, len(param_ranges)) containing the drawn samples.
    """
    
    # Create a Latin Hypercube sampler
    sampler = model(d=len(param_ranges), rng=rng)
    
    # Generate samples in the unit hypercube
    sample = sampler.random(int(N))
    
    # Scale samples to the specified ranges
    scaled_samples = qmc.scale(sample, 
                               np.array([param_ranges[key][0] for key in param_ranges.keys()]), 
                               np.array([param_ranges[key][1] for key in param_ranges.keys()]))
    

    return scaled_samples.astype(np.float32)

def load_hypercube_from_npy(file_path: str):
    """
    Load a hypercube from a .npy file.
    
    Parameters
    ----------
    file_path : str
        Path to the .npy file containing the hypercube data.
    
    Returns
    -------
    np.ndarray
        Array of shape (N, M) containing the loaded hypercube data.
    """
    
    # Load the hypercube data from the .npy file
    hypercube = np.load(file_path)
    
    return hypercube.astype(np.float32)

def generate_sfh_basis(
        sfh_type: Type[SFH.Common],
        sfh_param_names: List[str],
        sfh_param_arrays: List[np.ndarray],
        sfh_param_units: List[Union[None, Unit]],
        redshifts: Union[Dict[str, Any], float, np.ndarray],
        max_redshift: float = 15,
        calculate_min_age: bool = False,
        min_age_frac = 0.001, 
        cosmo: Type[Cosmology] = Planck18,
        iterate_redshifts: bool = True,
    ) -> Tuple[List[Type[SFH.Common]], np.ndarray]:
    """
    Generate a grid of the N basis SFHs based on prior distributions for redshift, and pairs of parameters.

    Parameters
    ----------

    sfh_type : Type[SFH]
        The star formation history class to instantiate
    sfh_param_names : List[str]
        List of parameter names for SFH constructor
    sfh_param_arrays : List[np.ndarray]
        List of parameter arrays for SFH constructor.
        Should have the same length in the first dimension as sfh_param_names. If values are lambda functions the
        input will be the maximum age given the max_redshift.
    redshifts : Union[Dict[str, Any], float]
        Either a single redshift value, an array of redshifts, or a dictionary with:
        - 'prior': scipy.stats distribution
        - 'min': minimum redshift
        - 'max': maximum redshift 
        - 'size': number of redshift samples
    max_redshift : float, optional
        Maximum possible redshift to consider for age calculations, by default 15
    cosmo : Type[Cosmology], optional
        Cosmology to use for age calculations, by default Planck18

    calculate_min_age : bool, optional
        If True, calculate the lookback time at which only min_age_frac of total mass is formed, by default True
    min_age_frac : float, optional
        Fraction of total mass formed to calculate the minimum age, by default 0.001
    iterate_redshifts : bool, optional
        If True, iterate over redshifts and create SFH for each, by default True
        If False, assume input redshift SFH param array is a 1:1 mapping of redshift to SFH parameters.
    Returns
    -------
    Tuple[List[SFH], np.ndarray]
        - List of SFH objects with parameters drawn from the priors
        - Array of parameter combinations, where the first column is redshift followed by SFH parameters
    
    """

    if isinstance(redshifts, dict):
        redshifts = redshifts['prior'].rvs(
            size=int(redshifts['size']), 
            loc=redshifts['min'], 
            scale=redshifts['max'] - redshifts['min']
        )
    elif isinstance(redshifts, float):
        redshifts = np.array([redshifts])
    elif isinstance(redshifts, np.ndarray):
        redshifts = redshifts
    else:
        raise ValueError("redshifts must be a dictionary, float, or numpy array")   
    
    # Calculate maximum ages at each redshift

    max_ages = (cosmo.age(redshifts) - cosmo.age(max_redshift)).to(u.Myr).value

    sfhs = []

    all_redshifts = []
    param_names_i = [i.replace('_norm', '') for i in sfh_param_names]


    if iterate_redshifts:
        for i, redshift in enumerate(redshifts):
            params = copy.deepcopy(sfh_param_arrays)
            for j, row in enumerate(params):
                row_params = {}
                for k, param in enumerate(row):
                    # Check if the parameter is a function
                    if callable(param):
                        # Call the function with the maximum age
                        row_params[k] = param(max_ages[i])
                    else:
                        # Otherwise, just use the parameter value
                        row_params[k] = param

                    if sfh_param_units[k] is not None:
                        # Apply units if not None
                        row_params[k] = row_params[k] * sfh_param_units[k]

                # Create parameter dictionary for SFH constructor
                sfh_params = {
                    sfh_param_names[l]: row_params[l] for l in range(len(sfh_param_names))
                }

                if 'sfh_timescale' in sfh_param_names:
                    sfh_params['max_age'] = sfh_params['min_age'] + sfh_params['sfh_timescale']
                    sfh_params.pop('sfh_timescale')


                # Add max_age parameter
                if 'max_age' in sfh_param_names:
                    sfh_params['max_age'] = min(max_ages[i]  * Myr, sfh_params['max_age'])
                else:
                    sfh_params['max_age'] = max_ages[i] * Myr
                # Create and append SFH instance

                sfh = sfh_type(**sfh_params)
                sfh.redshift = redshift
                sfhs.append(sfh)
                all_redshifts.append(redshift)
    else:
        assert len(redshifts) == len(sfh_param_arrays), "If iterate_redshifts is False, redshifts and sfh_param_arrays must be the same length"
        for i, redshift in tqdm(enumerate(redshifts), desc="Creating SFHs"):
            params = copy.deepcopy(sfh_param_arrays[i])
            row_params = {}
            for j, param in enumerate(params):
                # Check if the parameter is a function
                if callable(param):
                    # Call the function with the maximum age
                    row_params[j] = param(max_ages[i])
                elif sfh_param_names[j].endswith('_norm'):
                    # If the parameter is normalized to the maximum age, multiply by max_age
                    row_params[j] = param * max_ages[i] * Myr
                else:
                    # Otherwise, just use the parameter value
                    row_params[j] = param

                if sfh_param_units[j] is not None:
                    # Apply units if not None
                    row_params[j] = row_params[j] * sfh_param_units[j]

            # Create parameter dictionary for SFH constructor

            sfh_params = {
                param_names_i[l]: row_params[l] for l in range(len(param_names_i))
            }

            # remove _norm from parameter names
            if 'sfh_timescale' in sfh_param_names:
                sfh_params['max_age'] = sfh_params['min_age'] + sfh_params['sfh_timescale']
                sfh_params.pop('sfh_timescale')

            # Add max_age parameter
            if 'max_age' in sfh_param_names:
                sfh_params['max_age'] = min(max_ages[i]  * Myr, sfh_params['max_age'])
            else:
                sfh_params['max_age'] = max_ages[i] * Myr
            # Create and append SFH instance
            sfh = sfh_type(**sfh_params)
            sfh.redshift = redshift
            sfhs.append(sfh)
            all_redshifts.append(redshift)


    if calculate_min_age:
        # Calculate lookback time at which only min_age_frac of total mass is formed
        min_ages = []
        for i, sfh in enumerate(sfhs):
            max_age = sfh.max_age
            # Calculate the cumulative mass formed
            age, sfr = sfh.calculate_sfh(t_range=(0, 1.1*max_age), dt=1e6*yr)                                       
            sfr = sfr / sfr.max()
            mass_formed = np.cumsum(sfr[::-1])[::-1]/np.sum(sfr)
            total_mass = mass_formed[0]
            
            # Find the age at which min_age_frac of total mass is formed
            # interpolate
            min_age = np.interp(min_age_frac * total_mass, mass_formed, age)
            min_ages.append(min_age)
            

    return np.array(sfhs), redshifts

def generate_constant_R(R=300, start=1*Angstrom, end=9e5*Angstrom, auto_start_stop=False, filterset=None, **kwargs):

    if auto_start_stop and filterset is not None:
        start, end = calculate_min_max_wav_grid(filterset, **kwargs)
    
    x=[start.to(Angstrom).value]

    while x[-1] < end.to(Angstrom).value:
        x.append(x[-1] * (1.0 + 0.5 / R))
    
    return np.array(x) * Angstrom

def list_parameters(distribution):
    """List parameters for scipy.stats.distribution.
    # Arguments
        distribution: a string or scipy.stats distribution object.
    # Returns
        A list of distribution parameter strings.
    # from https://stackoverflow.com/questions/30453097/getting-the-parameter-names-of-scipy-stats-distributions
    """
    if isinstance(distribution, str):
        distribution = getattr(scipy.stats, distribution)
    if distribution.shapes:
        parameters = [name.strip() for name in distribution.shapes.split(',')]
    else:
        parameters = []
    if distribution.name in scipy.stats._discrete_distns._distn_names:
        parameters += ['loc']
    elif distribution.name in scipy.stats._continuous_distns._distn_names:
        parameters += ['loc', 'scale']
    else:
        sys.exit("Distribution name not found in discrete or continuous lists.")
    return parameters

def rename_overlapping_parameters(lists_dict):
    """
    Check if N lists have any overlapping parameters and rename them if they do.
    
    Args:
        lists_dict: Dictionary where keys are list names and values are the lists
    
    Returns:
        Dictionary with renamed parameters where overlapping occurred
    """
    # Collect all parameters across all lists
    all_params = {}
    for list_name, params in lists_dict.items():
        for param in params:
            if param not in all_params:
                all_params[param] = []
            all_params[param].append(list_name)
    
    # Build the result with renamed parameters where needed
    result = {}
    for list_name, params in lists_dict.items():
        result[list_name] = []
        for param in params:
            # If parameter appears in multiple lists, rename it
            if len(all_params[param]) > 1:
                result[list_name].append(f"{list_name}_{param}")
            else:
                result[list_name].append(param)
    
    return result

class GalaxyBasis:
    def __init__(self, 
                model_name: str,
                redshifts: np.ndarray,
                grid: Grid,
                emission_model: Type[EmissionModel],
                galaxy_params: dict = {},
                sfhs: List[Type[SFH.Common]] = None,
                metal_dists: List[Type[ZDist.Common]] = None,
                cosmo: Type[Cosmology] = Planck18,
                instrument: Instrument = None,
                stellar_masses: unyt_array = None,
                redshift_dependent_sfh: bool = False,
                params_to_ignore: List[str] = [],
                build_grid: bool = True,
    ) -> None:
        """
        Initialize the GalaxyBasis object with SFHs, redshifts, and other parameters.
        
        Parameters
        ----------
        sfhs : List[Type[SFH.Common]]
            List of SFH objects.
        redshifts : np.ndarray
            Array of redshift values.
        grid : Grid
            Grid object containing the SPS grid.
        emission_model : Type[EmissionModel]
            Emission model class to instantiate.
        emission_model_params : dict
            Dictionary of parameters for the emission model.
        galaxy_params : dict
            Dictionary of parameters for the galaxy.
        metal_dists : List[Type[ZDist.Common]], optional
            List of metallicity distribution objects, by default None
        cosmo : Type[Cosmology], optional
            Cosmology object, by default Planck18
        instrument : Instrument, optional
            Instrument object containing the filters, by default None
        stellar_masses : unyt_array, optional
            Array of stellar masses for the galaxies, by default None
        redshift_dependent_sfh : bool, optional
            If True, the SFH will depend on redshift, by default False. If True, expect each
            SFH to have a redshift attribute.
        params_to_ignore : List[str], optional
            List of parameters to ignore as being different when calculating which parameters are varying.
            E.g. max_age may be dependent on redshift, so we don't want to include it in the varying parameters as the model can learn this. 
        build_grid : bool, optional
            If True, build the grid of galaxies, by default True.
            If False, assume all dimensions of parameters are the same size and build the grid from the parameters.
            # Don't generate combinations of parameters, just use the parameters as they are.
        """
        
        self.model_name = model_name
        self.sfhs = sfhs
        self.redshifts = redshifts
        self.grid = grid
        self.emission_model = emission_model
        self.galaxy_params = galaxy_params
        self.metal_dists = metal_dists
        self.cosmo = cosmo
        self.instrument = instrument
        self.stellar_masses = stellar_masses
        self.redshift_dependent_sfh = redshift_dependent_sfh
        self.params_to_ignore = params_to_ignore
        self.build_grid = build_grid

        self.galaxies = []

        if isinstance(self.metal_dists, ZDist.Common):
            self.metal_dists = [self.metal_dists]

        if isinstance(self.sfhs, SFH.Common):
            self.sfhs = [self.sfhs]

        self.per_particle = False

        # Check if any galaxy parameters are dictionaries with keys like 'prior', 'min', 'max', etc.
        for key, value in galaxy_params.items():
            if isinstance(value, dict):
                # If the value is a dictionary, process it as a prior
                self.galaxy_params[key] = self.process_priors(value)
        

        if not build_grid:
            print('Generating grid directly from provided parameter samples.')
        elif self.redshift_dependent_sfh:
            # Check if the SFHs have a redshift attribute
            for sfh in self.sfhs:
                if not hasattr(sfh, 'redshift'):
                    raise ValueError("SFH must have a redshift attribute if redshift_dependent_sfh is True")
                
            if not isinstance(self.redshifts, np.ndarray):
                self.redshifts = np.array(self.redshifts)

            # Check all redshifts are in self.redshifts
            for sfh in self.sfhs:
                if sfh.redshift not in self.redshifts:
                    raise ValueError(f"SFH redshift {sfh.redshift} not in redshifts array")
                
            # Make self.SFHs a dictionary with redshift as key
            sfh_dict = {}
            for sfh in self.sfhs:
                if sfh.redshift not in sfh_dict:
                    sfh_dict[sfh.redshift] = []
                sfh_dict[sfh.redshift].append(sfh)
            self.sfhs = sfh_dict
        else:
            self.sfhs = {z: self.sfhs for z in self.redshifts}

    def process_priors(self, 
                       prior_dict: Dict[str, Any],
                       ) -> unyt_array:

        assert isinstance(prior_dict, dict), "prior_dict must be a dictionary"
        
        assert 'prior' in prior_dict, "prior_dict must contain a 'prior' key"
        assert 'size' in prior_dict, "prior_dict must contain a 'size' key"
        
        stats_params = list_parameters(prior_dict['prior'])

        # Check required parameters are present
        params = {}
        for param in stats_params:
            assert param in prior_dict, f"prior_dict must contain a '{param}' key"
            params[param] = prior_dict[param]

        # draw values for this parameter

        values = prior_dict['prior'].rvs(
            size=int(prior_dict['size']),
            **params
        )

        if 'units' in prior_dict and prior_dict['units'] is not None:
            values = unyt_array(values, units=prior_dict['units'])

        return values
    
    def create_galaxies(self, 
                        stellar_masses: Union[unyt_array, unyt_quantity, None] = None,
                        base_mass: unyt_quantity = 1e9 * Msun
                        ) -> List[Type[Galaxy]]:
        """
        Create galaxies with the specified SFHs, redshifts, and other parameters.
        
        Parameters
        ----------
        stellar_masses : np.ndarray
            Array of stellar masses for the galaxies.
        
        Returns
        -------
        List[Type[Galaxy]]
            List of Galaxy objects.
        """

        if stellar_masses is None:
            stellar_masses = self.stellar_masses
            if stellar_masses is None:
                raise ValueError("stellar_masses must be provided or set in the constructor")
        
        varying_param_values = [i for i in self.galaxy_params.values() if type(i) in [list, np.ndarray]]
        # generate all combinations of the varying parameters
        if len(varying_param_values) == 0:
            param_list = [{}]
            fixed_params = self.galaxy_params

        else:
            varying_param_combinations = np.array(np.meshgrid(*varying_param_values)).T.reshape(-1, len(varying_param_values))
            column_names = [i for i, j in zip(self.galaxy_params.keys(), varying_param_values) if type(j) in [list, np.ndarray]]
            fixed_params = {key: value for key, value in self.galaxy_params.items() if type(value) not in [list, np.ndarray]}
            param_list = [{column_names[i]: j for i, j in enumerate(row)} for row in varying_param_combinations]


        galaxies = []
        all_parameters = {}
        for i, redshift in tqdm(enumerate(self.redshifts), desc=f"Creating {self.model_name} galaxies", total=len(self.redshifts)):
            # get the sfh for this redshift
            sfh_models = self.sfhs[redshift]
            for sfh_model in sfh_models:
                sfh_parameters = sfh_model.parameters
                for k, Z_dist in enumerate(self.metal_dists):
                    Z_parameters = Z_dist.parameters

                    # Create a new galaxy with the specified parameters
                    for params in param_list:
                        params.update(fixed_params)
                        gal = self.create_galaxy(sfh=sfh_model,
                                                    redshift=redshift,
                                                    metal_dist=Z_dist,
                                                    base_mass=base_mass,
                                                    stellar_mass=stellar_masses,
                                                    **params)
                        save_params = copy.deepcopy(params)
                        save_params['redshift'] = redshift
                        save_params.update(sfh_parameters)
                        save_params.update(Z_parameters)

                        # This stores all input parameters for the galaxy so we can work out which parameters
                        # are varying and which are fixed later.
                        gal.all_params = save_params

                        # add all_parameters to dictionary if that key doesn't exist

                        for key, value in save_params.items():
                            if key not in all_parameters:
                                all_parameters[key] = []
                            
                            if value not in all_parameters[key]:
                                all_parameters[key].append(value)
                            else:
                                pass

                        galaxies.append(gal)

        self.galaxies = galaxies

        # Remove any paremters which are just [None]
        to_remove = []
        fixed_param_names = []
        varying_param_names = []

        for key, value in all_parameters.items():
            if len(value) == 1 and value[0] is None:
                to_remove.append(key)
                continue
            # check if all values are the same. 
            if len(np.unique(value)) == 1:
                fixed_param_names.append(key)
            else:
                varying_param_names.append(key)
        
        for param in self.params_to_ignore:
            if param in varying_param_names:
                varying_param_names.remove(param)
            
        # Sanity check all varying parameters combinations on self.galaxies are unique
        print('Checking parameters are unique.')
        hashes = []
        for gal in self.galaxies:
            relevant_params = {key: gal.all_params[key] for key in varying_param_names 
                    if key in gal.all_params}
            # Calculate hash for each parameter and sum them
            hash_i = sum(
                hash(float(param.value)) if isinstance(param, (unyt_array, unyt_quantity)) else hash(param)
                for param in relevant_params.values()
            )
            hashes.append(hash_i)

        if len(hashes) != len(set(hashes)):
            raise ValueError("Varying parameters are not unique across galaxies. Check your input parameters.")

        self.varying_param_names = varying_param_names
        self.fixed_param_names = fixed_param_names
        self.all_parameters = all_parameters
        

        for key in to_remove:
            all_parameters.pop(key)

        print('Finished creating galaxies.')

        return self.galaxies
    
    def create_galaxy(self, 
                      sfh: Type[SFH.Common],
                      redshift: float,
                      metal_dist: Type[ZDist.Common],
                      base_mass: unyt_quantity = 1e9 * Msun,
                      stellar_mass: Union[unyt_array, unyt_quantity, None] = None,
                      **galaxy_kwargs) -> Type[Galaxy]:
        """
        Create a new galaxy with the specified parameters.
        """
        
        # Initialise the parametric Stars object
        param_stars = Stars(
            log10ages=self.grid.log10ages,
            metallicities=self.grid.metallicity,
            sf_hist=sfh,
            metal_dist=metal_dist,
            initial_mass=base_mass,
            **galaxy_kwargs, # most parameters want to be on the emitter
        )

        # Define the number of stellar particles
        n = stellar_mass.size if isinstance(stellar_mass, unyt_array) else 1
        

        if n > 1:
            # Sample the parametric SFZH to create "fake" stellar particles
            part_stars = sample_sfzh(
                sfzh=param_stars.sfzh,
                log10ages=np.log10(param_stars.ages),
                log10metallicities=np.log10(param_stars.metallicities),
                nstar=n,
                current_masses=stellar_mass,
                redshift=redshift,
                coordinates=np.random.normal(0, 0.01, (n, 3)) * Mpc,
                centre=np.zeros(3) * Mpc,
            )
            self.per_particle = True
        else:
            part_stars = param_stars


        # And create the galaxy
        galaxy = Galaxy(
            stars=part_stars,
            redshift=redshift,
        )

        return galaxy

    def create_matched_galaxies(self,
                                base_mass: unyt_quantity = 1e9 * Msun
                                ) -> List[Type[Galaxy]]:
        '''
        Equivalent of create_galaxies but assumes that we don't need to draw
        parameter combinations.
        '''

        if len(self.metal_dists) == 1:
            # Just reference the first one 
            self.metal_dists = [self.metal_dists[0]] * len(self.sfhs)

        assert len(self.sfhs) == len(self.redshifts), f"If iterate_redshifts is False, sfhs and redshifts must be the same length, got {len(self.sfhs)} and {len(self.redshifts)}"
        assert len(self.sfhs) == len(self.metal_dists), f"If iterate_redshifts is False, sfhs and metal_dists must be the same length, got {len(self.sfhs)} and {len(self.metal_dists)}"

        varying_param_values = [i for i in self.galaxy_params.values() if type(i) in [list, np.ndarray]]

        # generate all combinations of the varying parameters
        if len(varying_param_values) == 0:
            param_list = [{}]
            fixed_params = self.galaxy_params
            varying_param_names = []

        else:
            column_names = [i for i, j in zip(self.galaxy_params.keys(), varying_param_values) if type(j) in [list, np.ndarray]]
            fixed_params = {key: value for key, value in self.galaxy_params.items() if type(value) not in [list, np.ndarray]}
            varying_param_names = [i for i in self.galaxy_params.keys() if i not in fixed_params.keys()]
            assert all(len(self.galaxy_params[i]) == len(self.sfhs) for i in varying_param_names), f"All varying parameters must be the same length, got {len(self.sfhs)} and {len(self.galaxy_params[i])} for {i}"
        
        for i in tqdm(range(len(self.sfhs)), desc='Creating galaxies.'):
            sfh = self.sfhs[i]
            redshift = self.redshifts[i]
            metal_dist = self.metal_dists[i]

            params = {}
            params.update(fixed_params)
            for j, key in enumerate(varying_param_names):
                params[key] = self.galaxy_params[key][i]

            # Create a new galaxy with the specified parameters
            gal = self.create_galaxy(sfh=sfh,
                                        redshift=redshift,
                                        metal_dist=metal_dist,
                                        base_mass=base_mass,
                                        stellar_mass=self.stellar_masses,
                                        **params)

            save_params = copy.deepcopy(params)
            save_params['redshift'] = redshift
            save_params.update(sfh.parameters)
            save_params.update(metal_dist.parameters)
            gal.all_params = save_params

            self.galaxies.append(gal)

        
        # Use sets instead of lists for faster lookups and unique values
        self.all_parameters = defaultdict(set)
        self.all_params = {}
        
        # Process all galaxies in one pass
        for i, gal in enumerate(self.galaxies):
            # Store the galaxy parameters directly
            self.all_params[i] = gal.all_params
            
            # Add unique values to sets using update operation
            for key, value in gal.all_params.items():
                if isinstance(value, (unyt_quantity, unyt_array)):
                    value = value.value
                if isinstance(value, np.ndarray):
                    value = value.tolist()

                self.all_parameters[key].add(value)
        
        # Convert sets to lists at the end if needed
        self.all_parameters = {k: list(v) for k, v in self.all_parameters.items()}
            
        # Remove any paremters which are just [None]
        to_remove = []
        fixed_param_names = []
        varying_param_names = []

        for key, value in self.all_parameters.items():
            if len(value) == 1 and value[0] is None:
                to_remove.append(key)
                continue
            # check if all values are the same. 
            if len(np.unique(value)) == 1:
                fixed_param_names.append(key)
            else:
                varying_param_names.append(key)
        
        for param in self.params_to_ignore:
            if param in varying_param_names:
                varying_param_names.remove(param)
            
        self.varying_param_names = varying_param_names
        self.fixed_param_names = fixed_param_names
        
        for key in to_remove:
            self.all_parameters.pop(key)

        print('Finished creating galaxies.')

        return self.galaxies

    def process_galaxies(self, 
                        galaxies: List[Type[Galaxy]],
                        out_name: str = 'auto',
                        out_dir: str = 'self',
                        n_proc: int = 4,
                        verbose: int = 1,
                        save: bool = True,
                        emission_model_keys=None,
                        **extra_analysis_functions
                        ) -> Pipeline:

        self.emission_model.set_per_particle(self.per_particle)

        if emission_model_keys is not None:
            self.emission_model.save_spectra(emission_model_keys)

        print("Creating pipeline.")

        pipeline = Pipeline(
            emission_model=self.emission_model,
            nthreads=n_proc,
            verbose=verbose,
        )


        for key in self.all_parameters.keys():
            pipeline.add_analysis_func(lambda gal, key=key: gal.all_params[key], result_key=key)
            
        pipeline.add_analysis_func(lambda gal: gal.stars.initial_mass, result_key='mass')

        print('Added analysis functions to pipeline.')

        # Add any extra analysis functions requested by the user. 

        for key, params in extra_analysis_functions.items():
            if callable(params):
                func = params
            else:
                func = params[0]
                params = params[1:]
            
            pipeline.add_analysis_func(func, f'supp_{key}', *params)

        pipeline.add_galaxies(galaxies)
        #pipeline.get_spectra() # Switch off so they aren't saved
        pipeline.get_observed_spectra(self.cosmo)

        #pipeline.get_photometry_luminosities() # Switch off so they aren't saved
        pipeline.get_photometry_fluxes(self.instrument)

        ngal = len(galaxies)
        print(f'Running pipeline at {datetime.now()} for {ngal} galaxies')
        pipeline.run()
        print(f'Finished running pipeline at {datetime.now()} for {ngal} galaxies')
        if save:
            # Save the pipeline to a file

            if out_dir == 'self':
                out_dir = self.out_dir

            if out_name == 'auto':
                out_name = self.model_name

            fullpath = os.path.join(out_dir, out_name)

            pipeline.write(fullpath, verbose=0)

            wav = self.grid.lam.to(Angstrom).value

            # it will put the file at fullpath_0.hdf5, so we need to rename it
            pipe_fullpath = fullpath.replace('.hdf5', f'_0.hdf5')
            os.rename(pipe_fullpath, fullpath)

            with h5py.File(fullpath, 'r+') as f:
                # Add the varying and fixed parameters to the file
                f.attrs['varying_param_names'] = self.varying_param_names
                f.attrs['fixed_param_names'] = self.fixed_param_names

                # Store grid wavelengths since I can't see them in the pipeline output (the filter wavelength array doesn't match)

                f.create_dataset('Wavelengths', data=wav)
                f['Wavelengths'].attrs['Units'] = 'Angstrom'
            
            print(f'Written pipeline to disk at {fullpath}.')

        return pipeline

    def plot_random_galaxy(self, masses, **kwargs):
        """
        Plot a random galaxy from the list of galaxies.
        """

        if not self.build_grid:
            idx = np.random.randint(0, len(self.redshifts))
            mass = masses[idx]
            return self.plot_galaxy(idx, stellar_mass=mass*Msun, **kwargs)

    def plot_galaxy(self,
                    idx,
                    save: bool = True,
                    stellar_mass: unyt_quantity = 1e9*Msun,
                    emission_model_keys: List[str] = ['total'],
                    out_dir='./'
                    ):
        """
        Plot the galaxy with the given index.
        """
        if not self.build_grid and len(self.galaxies) == 0:
            galaxy_params = {}
            for param in self.galaxy_params.keys():
                if isinstance(self.galaxy_params[param], dict):
                    galaxy_params[param] = self.process_priors(self.galaxy_params[param])
                elif isinstance(self.galaxy_params[param], (list, np.ndarray)):
                    galaxy_params[param] = self.galaxy_params[param][idx]
                else:
                    galaxy_params[param] = self.galaxy_params[param]

            # Get idx's from requirements and build galaxy directly
            galaxy = self.create_galaxy(sfh=self.sfhs[idx],
                                        redshift=self.redshifts[idx],
                                        metal_dist=self.metal_dists[idx],
                                        base_mass=stellar_mass,
                                        **galaxy_params)
        else:
            if idx >= len(self.galaxies):
                raise ValueError(f"Index {idx} out of range for galaxies.")

            galaxy = self.galaxies[idx]

        # Generate spectra

        galaxy.stars.get_spectra(self.emission_model)
        galaxy.get_observed_spectra(cosmo=self.cosmo, igm=Inoue14)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5), layout='constrained') 
        
        plot_dict = {key: galaxy.stars.spectra[key] for key in emission_model_keys}
        
        colors = {key: plt.cm.viridis(i/len(emission_model_keys)) for i, key in enumerate(plot_dict.keys())}
        
        plot_spectra(
            plot_dict,
            show=False,
            fig=fig,
            ax = ax[0],
            x_units=um,
            quantity_to_plot='fnu',
            draw_legend=False,
        )

        # change color of line with label key to the color of the key

        for line in ax[0].lines:
            label = line.get_label()
            test_colors = [i.title() for i in colors.keys()]
            if label in test_colors:
                pos = test_colors.index(label)
                label = list(colors.keys())[pos]
                line.set_color(colors[label])
                line.set_linewidth(1.5)
                line.set_label(label)

        ax[0].set_yscale('log')

        def custom_xticks(x, pos):
            if x == 0:
                return '0'
            else:
                return f'{x/1e4:.1f}'

        ax[0].xaxis.set_major_formatter(FuncFormatter(custom_xticks))

        min_x, max_x = 1e10 * um, 0 * um
        min_y, max_y = 1e10 * nJy, 0 * nJy

        text_gal = {}
        for emission_model in emission_model_keys:

            sed = galaxy.stars.spectra[emission_model]

            # Plot photometry
            phot = sed.get_photo_fnu(filters=self.instrument.filters)

            min_x = min(min_x, np.nanmin(phot.filters.pivot_lams))
            max_x = max(max_x, np.nanmax(phot.filters.pivot_lams))
            min_y = min(min_y, np.nanmin(phot.photo_fnu))
            max_y = max(max_y, np.nanmax(phot.photo_fnu))

            ax[0].plot(phot.filters.pivot_lams, phot.photo_fnu, '+', color=colors[emission_model], path_effects=[PathEffects.withStroke(linewidth=4, foreground='white')])

            # Get the redshift
            redshift = galaxy.redshift

            # Get the SFH
            stars_sfh = galaxy.stars.get_sfh()
            stars_sfh = stars_sfh / np.diff(10**(self.grid.log10age), prepend=0) / yr
            t, sfh = galaxy.stars.sf_hist_func.calculate_sfh()

            ax[1].plot(10**(self.grid.log10age-6), stars_sfh, label=f'{emission_model} SFH', color=colors[emission_model])
            ax[1].plot(t/1e6, sfh/np.max(sfh) * np.max(stars_sfh), label=f'Requested {emission_model} SFH', color=colors[emission_model], linestyle='--')

            mass = galaxy.stars.initial_mass
            if mass == 0:
                text_gal[emission_model] = f'{emission_model}     \nNo stars'
            else:
                age = galaxy.stars.calculate_mean_age()
                zmet = galaxy.stars.calculate_mean_metallicity()

                text_gal[emission_model] = f'{emission_model}     \nAge {age.to(Myr):.0f}\n$\log_{{10}}$Z: {np.log10(zmet.value):.2f}\nM$_\star$: {np.log10(mass):.1f} M$_\odot$'

        ax[0].legend(loc='upper right', fontsize=6, ncols=3)
        # Set the x-axis limits
        if max_y > 1 * nJy:
            min_y = max(min_y, 1 * nJy)
        ax[0].set_xlim(0.9*min_x, 1.1*max_x)
        ax[0].set_ylim(0.9*min_y, 2*max_y)       

        textstr = '\n'.join((
            r'$z = %.2f$' % (redshift),
        ))

        textstr += '\n' + '\n'.join(text_gal.values())

        # add galaxy params

        textstr += '\n' + '\n'.join([f'{key}: {value:.2f}' for key, value in galaxy_params.items()])

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # place a text box in upper left in axes coords
        ax[1].text(0.95, 0.72, textstr, transform=ax[1].transAxes, fontsize=12, horizontalalignment='right',
                verticalalignment='top', bbox=props)

        # add a secondary axis with AB magnitudes

        # 31.4

        def ab_to_jy(f):
            return 1e9 * 10**(f/(-2.5) -8.9)  

        def jy_to_ab(f):
            f = f/1e9
            return -2.5 * np.log10(f) + 8.9

        secax = ax[0].secondary_yaxis('right', functions=(jy_to_ab, ab_to_jy))
        # set scalar formatter

        max_age = self.cosmo.age(redshift) - self.cosmo.age(20)
        max_age = max_age.to(u.Myr).value 

        ax[1].set_xlim(0, 1.05*max_age)

        secax.yaxis.set_major_formatter(ScalarFormatter())
        secax.yaxis.set_minor_formatter(ScalarFormatter())
        secax.set_ylabel('Flux Density [AB mag]')

        ax[1].set_xlabel('Time [Myr]')
        ax[1].set_ylabel('SFH [M$_\odot$ yr$^{-1}$]')
        #ax[1].set_yscale('log')
        ax[1].legend()

        self.tmp_redshift = redshift    
        self.tmp_time_unit = u.Myr
        secax = ax[1].secondary_xaxis("top", functions=(self._time_convert, self._z_convert))
        secax.set_xlabel("Redshift")

        # Put a vertical line at maximum redshift

        secax.set_xticks([6, 7, 8, 10, 12, 14, 15, 20])

        if save:
            fig.savefig(f'{out_dir}/{self.model_name}_{idx}.png', dpi=300)
            plt.close(fig)

        return fig

    def _time_convert(self, lookback_time):
        time_unit = getattr(self, 'tmp_time_unit', u.yr)
        lookback_time = lookback_time * time_unit
        return z_at_value(
            self.cosmo.lookback_time,
            self.cosmo.lookback_time(self.tmp_redshift) + lookback_time,
        ).value

    def _z_convert(self, z):
        if type(z) in [list, np.ndarray] and len(z) == 0:
            return np.array([])
        
        time_unit = getattr(self, 'tmp_time_unit', u.yr)
        
        return (
            self.cosmo.lookback_time(z) - self.cosmo.lookback_time(self.tmp_redshift)
        ).to(time_unit).value


class CombinedBasis:
    def __init__(self, 
                bases: List[Type[GalaxyBasis]],
                total_stellar_masses: unyt_array,
                redshifts: np.ndarray,
                base_emission_model_keys: List[str],
                combination_weights: np.ndarray,
                out_name: str ='combined_basis',
                out_dir: str ='../output/',
                base_mass: unyt_array = 1e9 * Msun,
                draw_parameter_combinations: bool = True,
                ) -> None:
        

        self.bases = bases
        self.total_stellar_masses = total_stellar_masses
        self.redshifts = redshifts
        self.combination_weights = combination_weights
        self.out_name = out_name
        self.out_dir = out_dir
        self.base_mass = base_mass
        self.base_emission_model_keys = base_emission_model_keys
        self.draw_parameter_combinations = draw_parameter_combinations

        if self.combination_weights is None:
            assert len(self.bases) == 1
            self.combination_weights = [1.] * len(redshifts)


        # stellar mass just scales the basis SED. So computing spectra/photometry for each mass/combination is just a weighted sum of the basis SEDs.
        # We can just compute the basis SEDs for a single mass, and then scale them by the combination weights.

        # Based on total stellar masses and combination weights, calculate the stellar masses we need to
        # compute for each basis
        # combination weights would look like this:
        # [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5]] for 2 bases, or [[0.1, 0.1, 0.1, 0.1, 0.5], [...]] for 5 bases
        # stellar masses is e.g. [1e5, 1e6, 1e7, 1e8, 1e9] * Msun.
        # Calculate the stellar masses for each basis

    def process_bases(self, 
                      n_proc: int = 6,
                      overwrite: Union[bool, List[bool]] = False,
                      verbose=False,
                      **extra_analysis_functions,
                      ) -> None:
        """
        Process the bases and save the output to files.
        Parameters
        ----------
        n_proc : int
            Number of processes to use for the pipeline.
        overwrite : bool or list of bools
            If True, overwrite the existing files. If False, skip the files that already exist.
            If a list of bools is provided, it should have the same length as the number of bases.
        extra_analysis_functions : dict
            Extra analysis functions to add to the pipeline. The keys should be the names of the functions,
            and the values should be the functions themselves, or a tuple of (function, args). The function
            should take a Galaxy object as the first argument, and the args should be the arguments to pass to the function.
            The function should return a single value, an array of values, or a dictionary of values (with the same keys for all galaxies).
                """
        if not isinstance(overwrite, (tuple, list, np.ndarray)):
            overwrite = [overwrite] * len(self.bases)
        else:
            if len(overwrite) != len(self.bases):
                raise ValueError("overwrite must be a boolean or a list of booleans with the same length as bases")
            

        for i, base in enumerate(self.bases):
            full_out_path = f"{self.out_dir}/{base.model_name}.hdf5"

            if os.path.exists(full_out_path) and not overwrite[i]:
                print(f"File {full_out_path} already exists. Skipping.")
                continue
            elif os.path.exists(full_out_path) and overwrite[i]:
                print(f"File {full_out_path} already exists. Overwriting.")
                os.remove(full_out_path)
            elif not os.path.exists(self.out_dir):
                os.makedirs(self.out_dir)

            # Create galaxies for each base. Only create for one mass. We will scale later.
            if self.draw_parameter_combinations:
                galaxies = base.create_galaxies(base_mass=self.base_mass)
            else:
                galaxies = base.create_matched_galaxies(base_mass=self.base_mass)

            print(f'Created {len(galaxies)} galaxies for base {base.model_name}')
            # Process the galaxies
            pipeline = base.process_galaxies(galaxies,  
                                            f"{base.model_name}.hdf5",
                                            out_dir=self.out_dir,
                                            n_proc=n_proc, verbose=verbose, save=True, 
                                            emission_model_keys=self.base_emission_model_keys[i],
                                            **extra_analysis_functions,
                                            )
           
    def load_bases(self, indexes: List[int] = None) -> dict:
        # Load the bases from the files

        outputs = {}
        for i, base in enumerate(self.bases):

            print(f'Emission model key for base {base.model_name}: {self.base_emission_model_keys[i]}')

            full_out_path = f"{self.out_dir}/{base.model_name}.hdf5"
            if not os.path.exists(full_out_path):
                raise ValueError(f"File {full_out_path} does not exist")
            
            properties = {}
            supp_properties = {}

            with h5py.File(full_out_path, 'r') as f:
                
                # Load in which parameters are varying and fixed
                base.varying_param_names = f.attrs['varying_param_names']
                base.fixed_param_names = f.attrs['fixed_param_names']

                galaxies = f['Galaxies']

                property_keys = list(galaxies.keys())
                property_keys.remove('Stars')

                for key in property_keys:
                    if key.startswith('supp_'):
                        dic = supp_properties
                        use_key = key[5:]
                    else:
                        dic = properties
                        use_key = key

                    if isinstance(galaxies[key], h5py.Group):
                        dic[use_key] = {}
                        for subkey in galaxies[key].keys():
                            dic[use_key][subkey] = galaxies[key][subkey][()]
                            if hasattr(galaxies[key][subkey], 'attrs'):
                                if 'Units' in galaxies[key][subkey].attrs:
                                    unit = galaxies[key][subkey].attrs['Units']
                                    dic[use_key][subkey] = unyt_array(dic[use_key][subkey], unit)
                    else:                        
                        dic[use_key] = galaxies[key][()]
                        if hasattr(galaxies[key], 'attrs'):
                            if 'Units' in galaxies[key].attrs:
                                unit = galaxies[key].attrs['Units']
                                dic[use_key] = unyt_array(dic[use_key], unit)

                # Get the spectra
                spec = galaxies['Stars']['Spectra']['SpectralFluxDensities']
                assert self.base_emission_model_keys[i] in spec.keys(), f"Emission model key {self.base_emission_model_keys[i]} not found in {spec.keys()}"
                observed_spectra = spec[self.base_emission_model_keys[i]]
                observed_spectra = unyt_array(observed_spectra, units=observed_spectra.attrs['Units'])


                observed_photometry = galaxies['Stars']['Photometry']['Fluxes'][self.base_emission_model_keys[i]][base.instrument.label]

                phot = {}
                for key in observed_photometry.keys():
                    phot[key] = observed_photometry[key][()]

                
                outputs[base.model_name] = {
                    'properties': properties,
                    'observed_spectra': observed_spectra,
                    'wavelengths': unyt_array(f['Wavelengths'][()], units=f['Wavelengths'].attrs['Units']),
                    'observed_photometry': phot,
                    'supp_properties': supp_properties,
                }

        self.pipeline_outputs = outputs

        return outputs
        
    def create_grid(self, 
                override_instrument: Union[Instrument, None] = None,
                save: bool = True,
                overload_out_name: str = '',
                overwrite: bool = False,
                ) -> dict:    
        """
        Create a grid of SEDs for the given bases and stellar masses from the pipeline outputs.
        Can override the instrument used to compute the SEDs if you want to subset which filters are used in the grid.
        E.g. the base model is run for all NIRCam wide and medium filters, but you want to create a grid for just the wide filters.
        """

        if not self.draw_parameter_combinations:
            return self.create_full_grid(override_instrument, overwrite=overwrite, save=save, overload_out_name=overload_out_name)
        

        if os.path.exists(f"{self.out_dir}/{out_name}") and not overwrite:
            print(f"File {self.out_dir}/{out_name} already exists. Skipping.")
            self.load_grid_from_file(f"{self.out_dir}/{out_name}")
            return
        
        
        pipeline_outputs = self.load_bases()

        # Pipeline

        # given the true self.total_stellar_masses and self.combination_weights, create a grid of photometry. All masses 
        # for computed photometry are given in pipeline_outputs['properties']['mass']. Need to calculate scaling factor
        # to account for both total mass and mass split between bases.

        # create the output array. It should be (size(base1) * size(base2) * ... * size(basen) * len(self.combination_weights), len(self.filters))
        
        # calculate size of every combination of bases

        base_filters = self.bases[0].instrument.filters.filter_codes
        for i, base in enumerate(self.bases):
            if base.instrument.filters.filter_codes != base_filters:
                raise ValueError(f"Base {i} has different filters to base 0. Cannot combine bases with different filters.")
            
        if override_instrument is not None:
            # Check all filters in override_instrument are in the base filters
            for filter_code in override_instrument.filters.filter_codes:
                if filter_code not in base_filters:
                    raise ValueError(f"Filter {filter_code} not found in base filters. Cannot override instrument.")
                
            filter_codes = override_instrument.filters.filter_codes
        else:
            filter_codes = base_filters

        filter_codes = [i.split('/')[-1] for i in filter_codes]

        all_outputs = []
        all_params = []
        all_supp_params = []

        ignore_keys = ['redshift']

        total_property_names = {}
        for i, base in enumerate(self.bases):
            total_property_names[base.model_name] = [f'{base.model_name}/{i}' for i in base.varying_param_names if i not in ignore_keys]
            params = pipeline_outputs[base.model_name]['properties']
            rename_keys = [i for i in base.varying_param_names if i not in ignore_keys]
            for key in list(params.keys()):
                if key in rename_keys:
                    # rename the key to be the base name + parameter name
                    params[f'{base.model_name}/{key}'] = params[key]
        
        supp_param_keys = list(pipeline_outputs[self.bases[0].model_name]['supp_properties'].keys())
        assert all([i in pipeline_outputs[self.bases[0].model_name]['supp_properties'] for i in supp_param_keys]), f"Not all bases have the same supplementary parameters. {supp_param_keys} not found in {pipeline_outputs[self.bases[0].model_name]['supp_properties'].keys()}"


        # Deal with any supplementary model parameters. Currently we require that all bases have the same supplementary parameters and add them 
        supp_params = {}
        supp_param_units = {}
        for i, base in enumerate(self.bases):
            supp_params[base.model_name] = {}
            for key in supp_param_keys:
                if isinstance(pipeline_outputs[base.model_name]['supp_properties'][key], dict):
                    subkeys = list(pipeline_outputs[base.model_name]['supp_properties'][key].keys())
                    # Check if the emission model key is in the subkeys
                    if self.base_emission_model_keys[i] not in subkeys:
                        raise ValueError(f"Emission model key {self.base_emission_model_keys[i]} not found in {subkeys}. Don't know how to deal with dictionary supplementary parameters with other keys.")
                    value = pipeline_outputs[base.model_name]['supp_properties'][key][self.base_emission_model_keys[i]]
                else:
                    value = pipeline_outputs[base.model_name]['supp_properties'][key]
                
                supp_params[base.model_name][key] = value
                    


        # Check if any of the bases have the same varying parameters
        all_combined_param_names = []
        for key, value in total_property_names.items():
            all_combined_param_names.extend(value)
        
        # Add our standard parameters that are always included
        param_columns = ['redshift', 'log_mass']

        if len(self.bases) > 1:
            param_columns.append('weight_fraction')

        param_columns.extend(all_combined_param_names)

        for redshift in tqdm(self.redshifts):
            for total_mass in self.total_stellar_masses:
                for combination in self.combination_weights:
                    mass_weights = np.array(combination) * total_mass

                    scaled_photometries = []
                    base_param_values = []
                    supp_params_values = []

                    for i, base in enumerate(self.bases):
                        outputs = pipeline_outputs[base.model_name]
                        z_base = outputs['properties']['redshift']
                        mask = z_base == redshift
                        mass = outputs['properties']['mass'][mask]

                        # Calculate the scaling factor for each base
                        scaling_factors = mass_weights[i] / mass

                        base_photometry = np.array([pipeline_outputs[base.model_name]['observed_photometry'][filter_code][mask] for filter_code in filter_codes], dtype=np.float32)

                        # Scale the photometry by the scaling factor
                        scaled_photometry = base_photometry * scaling_factors

                        scaled_photometries.append(scaled_photometry)

                        # Get the varying parameters for this base
                        base_params = {}
                        for param_name in total_property_names[base.model_name]:
                            # Extract the original parameter name without the base prefix
                            orig_param = param_name.split('/')[-1]
                            if f'{base.model_name}/{orig_param}' in outputs['properties']:
                                base_params[param_name] = outputs['properties'][f'{base.model_name}/{orig_param}'][mask]
                            elif orig_param in outputs['properties']:
                                base_params[param_name] = outputs['properties'][orig_param][mask]
                        
                        base_param_values.append(base_params)
                        # Get the supplementary parameters for this base
                        # For any supp params that are a flux or luminosity, scale them by the scaling factor
                        scaled_supp_params = {}
                        for key, value in supp_params[base.model_name].items():
                            #print(key, value)
                            if isinstance(value, dict):
                                scaled_supp_params[key] = {}
                                for subkey, subvalue in value.items():
                                    if isinstance(subvalue, unyt_array):
                                        scaled_supp_params[key][subkey] = subvalue[mask] * scaling_factors
                                    else:
                                        scaled_supp_params[key][subkey] = subvalue[mask]
                            else:
                                if isinstance(value, unyt_array):
                                    scaled_supp_params[key] = value[mask] * scaling_factors
                                else:
                                    scaled_supp_params[key] = value[mask]

                        supp_params_values.append(scaled_supp_params)

                    # Calculate the total number of combinations
                    dimension = np.prod([i.shape[-1] for i in scaled_photometries])

                    output_array = np.zeros((scaled_photometries[0].shape[0], dimension))
                    params_array = np.zeros((len(param_columns), dimension))
                    supp_array = np.zeros((len(supp_param_keys), dimension))

                    # Create all combinations of indices
                    combinations = np.meshgrid(*[np.arange(i.shape[-1]) for i in scaled_photometries], indexing='ij')
                    combinations = np.array(combinations).T.reshape(-1, len(scaled_photometries))
                    
                    # Fill the output and parameter array.out_name
                    for i, combo_indices in enumerate(combinations):
                        # Add the scaled photometries for each base
                        for j, base in enumerate(self.bases):
                            output_array[:, i] += scaled_photometries[j][:, combo_indices[j]]
                        
                        # Fill parameter values
                        param_idx = 0
                        
                        # Add standard parameters first
                        params_array[param_idx, i] = redshift
                        param_idx += 1
                        
                        params_array[param_idx, i] = np.log10(total_mass.value)
                        param_idx += 1
                        
                        if len(self.bases) > 1:
                            params_array[param_idx, i] = combination[0]  # assuming this is weight fraction
                            param_idx += 1
                            
                        # Add all varying parameters from each base
                        for j, base in enumerate(self.bases):
                            for param_name in total_property_names[base.model_name]:
                                if param_name in base_param_values[j]:
                                    params_array[param_idx, i] = base_param_values[j][param_name][combo_indices[j]]
                                param_idx += 1

                        # Add supplementary parameters. Sum parameters from all bases
                        for j, base in enumerate(self.bases):
                            for k, param_name in enumerate(supp_param_keys):
                                data = supp_params_values[j][param_name][combo_indices[j]]
                                if isinstance(data, unyt_array):
                                    if j == 0:
                                        supp_param_units[param_name] = str(data.units)
                                    data = data.value
                                else:
                                    if j == 0:
                                        supp_param_units[param_name] = str(dimensionless)
                                
                                supp_array[k, i] += data


                    all_outputs.append(output_array)
                    all_params.append(params_array)
                    all_supp_params.append(supp_array)
        
        supp_param_units = [i for i in supp_param_units.values()]
        # Combine all outputs and parameters
        combined_outputs = np.hstack(all_outputs)
        combined_params = np.hstack(all_params)
        combined_supp_params = np.hstack(all_supp_params)

        '''print(f"Combined outputs shape: {combined_outputs.shape}")
        print(f"Combined parameters shape: {combined_params.shape}")
        print(f"Combined supplementary parameters shape: {combined_supp_params.shape}")
        print(f"Combined parameters: {combined_params}")
        print(f"Filter codes: {filter_codes}")'''

        out = {
            'photometry': combined_outputs,
            'parameters': combined_params,
            'parameter_names': param_columns,
            'filter_codes': filter_codes,
            'supplementary_parameters': combined_supp_params,
            'supplementary_parameter_names': supp_param_keys,
            'supplementary_parameter_units': supp_param_units,
        }

        self.grid_photometry = combined_outputs
        self.grid_parameters = combined_params
        self.grid_parameter_names = param_columns
        self.grid_filter_codes = filter_codes
        self.grid_supplementary_parameters = combined_supp_params
        self.grid_supplementary_parameter_names = supp_param_keys

        if save:
            self.save_grid(out, out_name=out_name, overwrite=overwrite)      

    def save_grid(self,
                  grid_dict: dict, # E.g. output from create_grid
                  overload_out_name: str = '',
                  overwrite: bool = False,
                  grid_params_to_save=['model_name'],
                  ) -> None:
        
        """
        Save the grid to a file.
        Parameters
        ----------
        grid_dict : dict
            Dictionary containing the grid data.
            Expected keys are 'photometry', 'parameters', 'parameter_names', and 'filter_codes'.

        out_name : str, optional
            Name of the output file, by default 'grid.hdf5'
        """
        # Check if the output directory exists, if not create it
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        if overload_out_name != '':
            out_name = overload_out_name
        else:
            out_name = self.out_name

        if not out_name.endswith('.hdf5'):
            out_name = f'{out_name}.hdf5'
        # Create the full output path
        full_out_path = os.path.join(self.out_dir, out_name)
        # Check if the file already exists
        if os.path.exists(full_out_path) and not overwrite:
            print(f"File {full_out_path} already exists. Skipping.")
            return
        elif os.path.exists(full_out_path) and overwrite:
            print(f"File {full_out_path} already exists. Overwriting.")
            os.remove(full_out_path)
        # Create a new HDF5 file
        with h5py.File(full_out_path, 'w') as f:
            # Create a group for the grid data
            grid_group = f.create_group('Grid')
            # Create datasets for the photometry and parameters
            grid_group.create_dataset('Photometry', data=grid_dict['photometry'], compression='gzip')

            grid_group.create_dataset('Parameters', data=grid_dict['parameters'], compression='gzip')
            grid_group.create_dataset('SupplementaryParameters', data=grid_dict['supplementary_parameters'], compression='gzip')
            #grid_group['Parameters'].attrs['Units'] = grid_dict['parameters'].units # no units stored

            # Create a dataset for the parameter names
            f.attrs['ParameterNames'] = grid_dict['parameter_names']
            f.attrs['FilterCodes'] = grid_dict['filter_codes']
            f.attrs['SupplementaryParameterNames'] = grid_dict['supplementary_parameter_names']
            f.attrs['SupplementaryParameterUnits'] = grid_dict['supplementary_parameter_units']
            f.attrs['PhotometryUnits'] = 'nJy'

            for param in grid_params_to_save:
                out = []
                for base in self.bases:
                    out.append(str(getattr(base, param)))
                f.attrs[param] = out
            
            # Add a timestamp

            f.attrs['Grids'] = [base.grid.grid_name for base in self.bases]

            f.attrs['CreationDT'] = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Add anything else as a dataset
            for key, value in grid_dict.items():
                if key not in ['photometry', 'parameters', 'parameter_names', 'filter_codes',
                               'supplementary_parameters', 'supplementary_parameter_names',
                               'supplementary_parameter_units']:
                    if isinstance(value, (np.ndarray, list)) and isinstance(value[0], str):
                        f.attrs[key] = value
                    else:
                        grid_group.create_dataset(key, data=value, compression='gzip')
                        if isinstance(value, (unyt_array, unyt_quantity)):
                            grid_group[key].attrs['Units'] = value.units
            
    def plot_galaxy_from_grid(self,
        index: int,
        show: bool = True,
        save: bool = False,
    ):
        '''
        Given an index, plot the galaxy from the grid. Open the relevant hdf5 files and load the spectra.
        
        '''

        if self.grid_photometry is None:
            raise ValueError("Grid photometry not created. Run create_grid() first.")

        if not hasattr(self, 'pipeline_outputs'):
            self.load_bases()

        # Get the parameters for this index
        params = self.grid_parameters[:, index]

        # Get the photometry for this index
        photometry = self.grid_photometry[:, index]

        # Get the filter codes
        filter_codes = [f'JWST/{i}' for i in self.grid_filter_codes]
        filterset = FilterCollection(filter_codes, verbose=False)

        # For each basis, look at which parameters for that basis and match to the spectra.

        combined_spectra = []
        total_wavelengths = []
        for i, base in enumerate(self.bases):
            # Get the varying parameters for this base
            base_params = {}
            for i, param_name in enumerate(self.grid_parameter_names):
                # Extract the original parameter name without the base prefix
                if '/' in param_name:
                    basis, orig_param = param_name.split('/')
                else:
                    basis = base.model_name
                    orig_param = param_name
                if basis == base.model_name:
                    base_params[orig_param] = params[i]
                    

            basis_params = list(self.pipeline_outputs[base.model_name]['properties'].keys())

            total_mask = np.ones(len(self.pipeline_outputs[base.model_name]['properties'][basis_params[0]]), dtype=bool)
            for key in base_params.keys():
                if key not in basis_params:
                    continue
                all_values = self.pipeline_outputs[base.model_name]['properties'][key]
                _i = all_values == base_params[key]
                total_mask = np.logical_and(total_mask, _i)

            assert np.sum(total_mask) == 1, f"Found {np.sum(total_mask)} matches for {base.model_name} with parameters {base_params}. Expected 1 match."
            j = np.where(total_mask)[0][0]

            # Get the spectra for this index
            spectra = self.pipeline_outputs[base.model_name]['observed_spectra'][j]

            flux_unit = spectra.units
            # get the mass of the spectra and the expected mass, and renormalise the spectra
            mass = self.pipeline_outputs[base.model_name]['properties']['mass'][j]
            expected_mass = 10**params[1] * Msun
            scaling_factor = expected_mass / mass
            spectra = spectra * scaling_factor
            # Append the spectra to the combined spectra
            combined_spectra.append(spectra)
            wavs = self.pipeline_outputs[base.model_name]['wavelengths']
            total_wavelengths.append(wavs)

        # Assert wavelengths for all bases are the same for now (could refactor this to resample later)
        for i, wavs in enumerate(total_wavelengths):
            if i == 0:
                continue
            assert np.all(wavs == total_wavelengths[0]), f"Wavelengths for base {i} do not match base 0. {wavs} != {total_wavelengths[0]}"

        weight_pos = 'weight_fraction' == self.grid_parameter_names
        weights = params[weight_pos]

        # Only works for combining 2 bases at the moment
        weights = np.array((weights[0], 1-weights[0]))
        
        # Stack the spectra according to the combination weights. Spectra has shape (wav, n_bases)
        combined_spectra = np.array(combined_spectra)

        combined_spectra = combined_spectra * weights[:, np.newaxis]
        
        combined_spectra_summed = np.sum(combined_spectra, axis=0)

        # apply redshift 

        combined_spectra_summed = combined_spectra_summed 

        fig, ax = plt.subplots(figsize=(10, 6))
        
        photwavs = filterset.pivot_lams

        ax.scatter(photwavs, photometry, label='Photometry', color='red', s=10, path_effects=[PathEffects.withStroke(linewidth=4, foreground='white')])

        ax.plot(wavs * (1 + params[0]), combined_spectra_summed, label='Combined Spectra', color='blue')

        for i, base in enumerate(self.bases):
            # Get the spectra for this index
            spectra = combined_spectra[i]
            
            ax.plot(wavs *(1+params[0]), spectra, label=f'{base.model_name} Spectra', alpha=0.5, linestyle='--')

        ax.set_xlabel('Wavelength (AA)')

        ax.set_xlim(0.8*np.min(photwavs), 1.2*np.max(photwavs))

        ax.set_yscale('log')
        ax.set_ylim(1e-2, None)
        ax.legend()

        def ab_to_jy(f):
            return 1e9 * 10**(f/(-2.5) -8.9)  

        def jy_to_ab(f):
            f = f/1e9
            return -2.5 * np.log10(f) + 8.9

        secax = ax.secondary_yaxis('right', functions=(jy_to_ab, ab_to_jy))

        secax.yaxis.set_major_formatter(ScalarFormatter())
        secax.yaxis.set_minor_formatter(ScalarFormatter())
        secax.set_ylabel('Flux Density [AB mag]')

        ax.set_xlabel(f'Wavelength ({wavs.units})')
        ax.set_ylabel(f'Flux Density ({flux_unit})')

        # Text box with parameters and values

        textstr = '\n'.join([f'{key}: {value:.2f}' for key, value in zip(self.grid_parameter_names, params)])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.5, 0.98, f'index: {index}\n' + textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props, horizontalalignment='center')

        if show:
            plt.show()
            
        if save:
            fig.savefig(f"{self.out_dir}/{self.out_name}_{index}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

        return fig
                         
    def load_grid_from_file(self, 
                        file_path: str,
                        ) -> dict:
        """
        Load the grid from a file.
        Parameters
        ----------
        file_path : str
            Path to the HDF5 file containing the grid data.
        Returns
        -------
        dict
            Dictionary containing the grid data.
        """
        with h5py.File(file_path, 'r') as f:
            grid_data = {
                'photometry': f['Grid']['Photometry'][()],
                'parameters': f['Grid']['Parameters'][()],
                'parameter_names': f.attrs['ParameterNames'],
                'filter_codes': f.attrs['FilterCodes'],
            }

        self.grid_photometry = grid_data['photometry']
        self.grid_parameters = grid_data['parameters']
        self.grid_parameter_names = grid_data['parameter_names']
        self.grid_filter_codes = grid_data['filter_codes']
        
        return grid_data
    
    def create_full_grid(self,
                      override_instrument: Union[Instrument, None] = None,
                      save: bool = True,
                      overload_out_name: str = '',
                      overwrite: bool = False) -> None:
        """
        Create a complete grid of spectral energy distributions (SEDs) by combining galaxy bases.
        
        This method handles both single-base and multi-base scenarios, properly scaling according to
        masses and weights. It constructs a grid without sampling, generating the full set of
        combinations for all specified parameters.
        
        Parameters
        ----------
        override_instrument : Instrument, optional
            If provided, use these filters instead of those in the bases
        save : bool, default=True
            Whether to save the grid to disk
        overload_out_name : str, default=''
            Custom filename for saving the grid
        overwrite : bool, default=False
            Whether to overwrite existing files
        """
        # Validate initialization conditions
        if self.draw_parameter_combinations:
            raise AssertionError("Cannot create full grid with draw_parameter_combinations set to True. Set to False to create full grid.")
        
        # Load base model data
        pipeline_outputs = self.load_bases()
        
        # ===== VALIDATION =====
        # Check all bases have the same number of galaxies
        ngal = len(pipeline_outputs[self.bases[0].model_name]['properties']['mass'])
        for i, base in enumerate(self.bases):
            model_name = base.model_name
            if len(pipeline_outputs[model_name]['properties']['mass']) != ngal:
                raise ValueError(f"Base {i} has different number of galaxies to base 0. Cannot combine bases with different number of galaxies.")
        
        # Validate input arrays
        for array_name, array in [('redshifts', self.redshifts), 
                                ('total_stellar_masses', self.total_stellar_masses),
                                ('combination_weights', self.combination_weights)]:
            if len(array) != ngal:
                raise ValueError(f"{array_name} length {len(array)} does not match number of galaxies {ngal}.")
        
        # Validate filters
        base_filters = self.bases[0].instrument.filters.filter_codes
        for i, base in enumerate(self.bases):
            if base.instrument.filters.filter_codes != base_filters:
                raise ValueError(f"Base {i} has different filters to base 0. Cannot combine bases with different filters.")
        
        # Determine which filters to use
        if override_instrument is not None:
            # Ensure all requested filters exist in the base
            for filter_code in override_instrument.filters.filter_codes:
                if filter_code not in base_filters:
                    raise ValueError(f"Filter {filter_code} not found in base filters. Cannot override instrument.")
            filter_codes = override_instrument.filters.filter_codes
        else:
            filter_codes = base_filters
        
        # Strip path info from filter codes
        filter_codes = [code.split('/')[-1] for code in filter_codes]
        
        # ===== PARAMETER SETUP =====
        # Set up parameter names
        ignore_keys = ['redshift']
        param_columns = ['redshift', 'log_mass']
        
        # Add weight fraction parameter for multiple bases
        is_multi_base = len(self.bases) > 1
        if is_multi_base:
            param_columns.append('weight_fraction')
        
        # Set up per-base parameter tracking
        total_property_names = {}
        for base in self.bases:
            model_name = base.model_name
            # Track parameters unique to this base
            varying_params = [f'{model_name}/{param}' for param in base.varying_param_names if param not in ignore_keys]
            total_property_names[model_name] = varying_params
            param_columns.extend(varying_params)
            
            # Rename parameter keys in the pipeline outputs
            params = pipeline_outputs[model_name]['properties']
            for param in base.varying_param_names:
                if param not in ignore_keys and param in params:
                    params[f'{model_name}/{param}'] = params[param]
        
        # ===== SUPPLEMENTARY PARAMETER SETUP =====
        # Get the list of supplementary parameters from the first base
        supp_param_keys = list(pipeline_outputs[self.bases[0].model_name]['supp_properties'].keys())
        
        # Validate all bases have the same supplementary parameters
        for key in supp_param_keys:
            for base in self.bases:
                if key not in pipeline_outputs[base.model_name]['supp_properties']:
                    raise ValueError(f"Supplementary parameter {key} not found in base {base.model_name}.")
        
        # Process supplementary parameters for each base
        supp_params = {}
        supp_param_units = {}
        
        for i, base in enumerate(self.bases):
            model_name = base.model_name
            supp_params[model_name] = {}
            
            for key in supp_param_keys:
                value = pipeline_outputs[model_name]['supp_properties'][key]
                
                # Handle nested dictionary case (typically for emission models)
                if isinstance(value, dict):
                    emission_key = self.base_emission_model_keys[i]
                    if emission_key not in value:
                        raise ValueError(f"Emission model key {emission_key} not found in supplementary parameter {key}.")
                    supp_params[model_name][key] = value[emission_key]
                else:
                    supp_params[model_name][key] = value
        
        # ===== PROCESS EACH galaxy =====
        all_outputs = []
        all_params = []
        all_supp_params = []
        
        for pos in range(ngal):
            redshift = self.redshifts[pos]
            total_mass = self.total_stellar_masses[pos]
            weights = self.combination_weights[pos]
            
            # Create per-base data structures to hold galaxy information at this redshift
            base_data = []
            
            # For each base, extract galaxies at this redshift and calculate scaling factors
            for i, base in enumerate(self.bases):
                model_name = base.model_name
                model_output = pipeline_outputs[model_name]
                
                masses = np.array([model_output['properties']['mass'][pos]*Msun]) 

                if is_multi_base:
                    # Multiple bases: scale by weight for this base
                    scaling_factors = weights[i] * total_mass / masses
                else:
                    # Single base: scale by total mass directly
                    scaling_factors = total_mass / masses
                
                # Get photometry for all filters and scale it
                photometry = np.array([model_output['observed_photometry'][code][pos] 
                                    for code in filter_codes], dtype=np.float32)
                scaled_phot = photometry * scaling_factors
                
                # Extract parameters for this base
                params_dict = {}
                for param_name in total_property_names.get(model_name, []):
                    original_param = param_name.split('/')[-1]
                    # Check both possible parameter locations
                    if f'{model_name}/{original_param}' in model_output['properties']:
                        params_dict[param_name] = model_output['properties'][f'{model_name}/{original_param}'][pos]
                    elif original_param in model_output['properties']:
                        params_dict[param_name] = model_output['properties'][original_param][pos]
                
                # Process supplementary parameters
                supp_dict = {}
                for key, value in supp_params[model_name].items():
                    if isinstance(value, dict):
                        supp_dict[key] = {}
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, unyt_array):
                                supp_dict[key][subkey] = subvalue[pos] * scaling_factors
                                supp_param_units[key] = str(subvalue.units)
                            else:
                                supp_dict[key][subkey] = subvalue[pos]
                                supp_param_units[key] = str(dimensionless)
                    else:
                        if isinstance(value, unyt_array):
                            supp_dict[key] = value[pos] * scaling_factors
                            supp_param_units[key] = str(value.units)
                        else:
                            supp_dict[key] = value[pos]
                            supp_param_units[key] = str(dimensionless)
                
                # Store all relevant data for this base
                base_data.append({
                    'photometry': scaled_phot,
                    'params': params_dict,
                    'supp_params': supp_dict,
                    'num_items': len(masses)
                })
            
            # Decide how to process based on number of bases
            if not is_multi_base:
                # SINGLE BASE CASE - just use the data directly
                base = base_data[0]
                num_items = base['num_items']

                # Use photometry directly
                output_array = base['photometry']
                
                # Create parameter array
                params_array = np.zeros((len(param_columns), num_items))
                param_idx = 0
                
                # Fill standard parameters
                params_array[param_idx, :] = redshift
                param_idx += 1
                params_array[param_idx, :] = np.log10(total_mass.value)
                param_idx += 1
                
                # Fill varying parameters
                for param_name in total_property_names.get(self.bases[0].model_name, []):
                    if param_name in base['params']:
                        params_array[param_idx, :] = base['params'][param_name]
                    param_idx += 1
                
                # Process supplementary parameters
                supp_array = np.zeros((len(supp_param_keys), num_items))
                for k, param_name in enumerate(supp_param_keys):
                    data = base['supp_params'][param_name]
                    if isinstance(data, unyt_array):
                        data = data.value
                    supp_array[k, :] = data
                    
            else:
                # MULTI-BASE CASE - create combinations
                
                # Get the number of items per base
                items_per_base = [base['num_items'] for base in base_data]
                
                # Create meshgrid of indices for combination
                mesh_indices = np.meshgrid(*[np.arange(n) for n in items_per_base], indexing='ij')
                
                # Reshape to get all combinations: array of shape (total_combinations, num_bases)
                combinations = np.array([indices.flatten() for indices in mesh_indices]).T
                num_combinations = len(combinations)

                # Create output arrays
                output_array = np.zeros((len(filter_codes), num_combinations))
                params_array = np.zeros((len(param_columns), num_combinations))
                supp_array = np.zeros((len(supp_param_keys), num_combinations))
                
                # Fill arrays
                for i, combo_indices in enumerate(combinations):
                    # Fill photometry - combine the weighted contributions
                    for j, base_idx in enumerate(combo_indices):
                        output_array[:, i] += base_data[j]['photometry'][:, base_idx]
                    
                    # Fill parameters
                    param_idx = 0
                    
                    # Standard parameters first
                    params_array[param_idx, i] = redshift
                    param_idx += 1
                    params_array[param_idx, i] = np.log10(total_mass.value)
                    param_idx += 1
                    if is_multi_base:
                        params_array[param_idx, i] = weights[0]  # weight fraction
                        param_idx += 1
                    
                    # Add all varying parameters from each base
                    for j, base in enumerate(self.bases):
                        model_name = base.model_name
                        for param_name in total_property_names.get(model_name, []):
                            if param_name in base_data[j]['params']:
                                params_array[param_idx, i] = base_data[j]['params'][param_name][combo_indices[j]]
                            param_idx += 1
                    
                    # Fill supplementary parameters
                    for k, param_name in enumerate(supp_param_keys):
                        total_value = 0
                        for j, base_idx in enumerate(combo_indices):
                            data = base_data[j]['supp_params'][param_name]
                            if isinstance(data, unyt_array):
                                total_value += data[base_idx].value
                            elif hasattr(data, '__len__') and len(data) > base_idx:
                                total_value += data[base_idx]
                            else:
                                total_value += data  # Scalar case
                        supp_array[k, i] = total_value
            
            # Append results for this redshift to the global arrays
            all_outputs.append(output_array)
            all_params.append(params_array)
            all_supp_params.append(supp_array)
        
        # ===== COMBINE RESULTS =====
        
        combined_outputs = np.hstack(all_outputs).T 
        combined_params = np.hstack(all_params)
        combined_supp_params = np.hstack(all_supp_params)
        
        # Convert units dict to list
        supp_param_units_list = [supp_param_units.get(name, str(dimensionless)) for name in supp_param_keys]
        
        # Print summary
        print(f"Combined outputs shape: {combined_outputs.shape}")
        print(f"Combined parameters shape: {combined_params.shape}")
        print(f"Combined supplementary parameters shape: {combined_supp_params.shape}")
        print(f"Filter codes: {filter_codes}")
        
        # Create output dictionary
        out = {
            'photometry': combined_outputs,
            'parameters': combined_params,
            'parameter_names': param_columns,
            'filter_codes': filter_codes,
            'supplementary_parameters': combined_supp_params,
            'supplementary_parameter_names': supp_param_keys,
            'supplementary_parameter_units': supp_param_units_list,
        }
        
        # Update object attributes
        self.grid_photometry = combined_outputs
        self.grid_parameters = combined_params
        self.grid_parameter_names = param_columns
        self.grid_filter_codes = filter_codes
        self.grid_supplementary_parameters = combined_supp_params
        self.grid_supplementary_parameter_names = supp_param_keys
        
        # Save results if requested
        if save:
            self.save_grid(out, overload_out_name=overload_out_name, overwrite=overwrite)
            
    def create_full_grid_old(self,
                override_instrument: Union[Instrument, None] = None,
                save: bool = True,
                overload_out_name: str = '',
                overwrite: bool = False,
                ) -> None:
        '''
        This is the annoying case where we have to create a grid of SEDs for a given set of parameters,
        but we're not sampling a grid so we have to generate the fullset of galaxies for both bases.
        '''

        assert self.draw_parameter_combinations == False, "Cannot create full grid with draw_parameter_combinations set to True. Set to False to create full grid."

        pipeline_outputs = self.load_bases()
        # Check all basis have the same number of galaxies
        ngal = len(pipeline_outputs[self.bases[0].model_name]['properties']['mass'])
        for i, base in enumerate(self.bases):
            if len(pipeline_outputs[base.model_name]['properties']['mass']) != ngal:
                raise ValueError(f"Base {i} has different number of galaxies to base 0. Cannot combine bases with different number of galaxies.")

        assert len(self.redshifts) == ngal, f"Redshift array length {len(self.redshifts)} does not match number of galaxies {ngal}. Cannot combine bases with different number of galaxies."
        assert len(self.total_stellar_masses) == ngal, f"Mass array length {len(self.total_stellar_masses)} does not match number of galaxies {ngal}. Cannot combine bases with different number of galaxies."
        assert len(self.combination_weights) == ngal, f"Combination weights array length {len(self.combination_weights)} does not match number of galaxies {ngal}. Cannot combine bases with different number of galaxies."
        
        base_filters = self.bases[0].instrument.filters.filter_codes
        for i, base in enumerate(self.bases):
            if base.instrument.filters.filter_codes != base_filters:
                raise ValueError(f"Base {i} has different filters to base 0. Cannot combine bases with different filters.")

        if override_instrument is not None:
            # Check all filters in override_instrument are in the base filters
            for filter_code in override_instrument.filters.filter_codes:
                if filter_code not in base_filters:
                    raise ValueError(f"Filter {filter_code} not found in base filters. Cannot override instrument.")
                
            filter_codes = override_instrument.filters.filter_codes
        else:
            filter_codes = base_filters

        filter_codes = [i.split('/')[-1] for i in filter_codes]

        all_outputs = []
        all_params = []
        all_supp_params = []
        ignore_keys = ['redshift']
        total_property_names = {}
        for j, base in enumerate(self.bases):
            total_property_names[base.model_name] = [f'{base.model_name}/{i}' for i in base.varying_param_names if i not in ignore_keys]
            params = pipeline_outputs[base.model_name]['properties']
            rename_keys = [g for g in base.varying_param_names if g not in ignore_keys]
            for key in list(params.keys()):
                if key in rename_keys:
                    # rename the key to be the base name + parameter name
                    params[f'{base.model_name}/{key}'] = params[key]
       
        # Check if any of the bases have the same varying parameters
        all_combined_param_names = []
        for key, value in total_property_names.items():
            all_combined_param_names.extend(value)
        
        # Add our standard parameters that are always included
        param_columns = ['redshift', 'log_mass']

        if len(self.bases) > 1:
            param_columns.append('weight_fraction')

        param_columns.extend(all_combined_param_names)

        # set supplementary parameters
        supp_param_keys = list(pipeline_outputs[self.bases[0].model_name]['supp_properties'].keys())
        assert all([i in pipeline_outputs[self.bases[0].model_name]['supp_properties'] for i in supp_param_keys]), f"Not all bases have the same supplementary parameters. {supp_param_keys} not found in {pipeline_outputs[self.bases[0].model_name]['supp_properties'].keys()}"
        supp_params = {}
        supp_param_units = {}
        for i, base in enumerate(self.bases):
            supp_params[base.model_name] = {}
            for key in supp_param_keys:
                if isinstance(pipeline_outputs[base.model_name]['supp_properties'][key], dict):
                    subkeys = list(pipeline_outputs[base.model_name]['supp_properties'][key].keys())
                    # Check if the emission model key is in the subkeys
                    if self.base_emission_model_keys[i] not in subkeys:
                        raise ValueError(f"Emission model key {self.base_emission_model_keys[i]} not found in {subkeys}. Don't know how to deal with dictionary supplementary parameters with other keys.")
                    value = pipeline_outputs[base.model_name]['supp_properties'][key][self.base_emission_model_keys[i]]
                else:
                    value = pipeline_outputs[base.model_name]['supp_properties'][key]
                
                supp_params[base.model_name][key] = value

        for pos, redshift in enumerate(self.redshifts):
            total_mass = self.total_stellar_masses[pos]
            combination = self.combination_weights[pos]
            
            # Handle both single and multi-base cases
            is_single_base = len(self.bases) == 1
            
            if is_single_base:
                # For single base, weight is always 1.0
                mass_weights = [total_mass]
            else:
                # For multiple bases, distribute according to weights
                mass_weights = np.array(combination) * total_mass

            # Process each base to get the scaled photometries and parameters
            scaled_photometries = []
            base_param_values = []
            supp_params_values = []

            for i, base in enumerate(self.bases):
                outputs = pipeline_outputs[base.model_name]
                z_base = outputs['properties']['redshift']
                mask = z_base == redshift
                mass = outputs['properties']['mass'][mask]
                if len(mass) == 0:
                    raise ValueError(f"No galaxies found for redshift {redshift} in base {base.model_name}. Check your redshift array.")

                # Calculate the scaling factor for each base
                if is_single_base:
                    scaling_factors = mass_weights[0] / mass
                else:
                    scaling_factors = mass_weights[i] / mass

                base_photometry = np.array([pipeline_outputs[base.model_name]['observed_photometry'][filter_code][mask] for filter_code in filter_codes], dtype=np.float32)

                # Scale the photometry by the scaling factor
                scaled_photometry = base_photometry * scaling_factors
                scaled_photometries.append(scaled_photometry)

                # Get the varying parameters for this base
                base_params = {}
                for param_name in total_property_names[base.model_name]:
                    # Extract the original parameter name without the base prefix
                    orig_param = param_name.split('/')[-1]
                    if f'{base.model_name}/{orig_param}' in outputs['properties']:
                        base_params[param_name] = outputs['properties'][f'{base.model_name}/{orig_param}'][mask]
                    elif orig_param in outputs['properties']:
                        base_params[param_name] = outputs['properties'][orig_param][mask]
                
                base_param_values.append(base_params)
                
                # Get the supplementary parameters for this base
                # For any supp params that are a flux or luminosity, scale them by the scaling factor
                scaled_supp_params = {}
                for key, value in supp_params[base.model_name].items():
                    if isinstance(value, dict):
                        scaled_supp_params[key] = {}
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, unyt_array):
                                scaled_supp_params[key][subkey] = subvalue[mask] * scaling_factors
                            else:
                                scaled_supp_params[key][subkey] = subvalue[mask]
                    else:
                        if isinstance(value, unyt_array):
                            scaled_supp_params[key] = value[mask] * scaling_factors
                        else:
                            scaled_supp_params[key] = value[mask]

                supp_params_values.append(scaled_supp_params)

            # For single base, we don't need to combine - just use all the data points from that base
            if is_single_base:
                num_items = scaled_photometries[0].shape[1]
                output_array = scaled_photometries[0]
                params_array = np.zeros((len(param_columns), num_items))
                supp_array = np.zeros((len(supp_param_keys), num_items))
                
                # Fill parameter values
                param_idx = 0
                
                # Add standard parameters first
                params_array[param_idx, :] = redshift
                param_idx += 1
                
                params_array[param_idx, :] = np.log10(total_mass.value)
                param_idx += 1
                
                # Add all varying parameters from the base
                base_idx = 0
                for param_name in total_property_names[self.bases[0].model_name]:
                    if param_name in base_param_values[0]:
                        params_array[param_idx, :] = base_param_values[0][param_name]
                    param_idx += 1

                # Add supplementary parameters
                for k, param_name in enumerate(supp_param_keys):
                    data = supp_params_values[0][param_name]
                    if isinstance(data, unyt_array):
                        supp_param_units[param_name] = str(data.units)
                        data = data.value
                    else:
                        supp_param_units[param_name] = str(dimensionless)
                    
                    supp_array[k, :] = data
                
            else:
                # For multiple bases, we need to create combinations
                # Calculate the number of items per base (not the total combinations)
                items_per_base = [photometry.shape[1] for photometry in scaled_photometries]
                
                # Create individual index arrays for each base
                indices_per_base = [np.arange(items) for items in items_per_base]
                
                # Create the combinations grid
                combinations = np.meshgrid(*indices_per_base, indexing='ij')
                # Reshape to get all combinations
                combinations = np.array(combinations).reshape(len(self.bases), -1).T
                
                # Now combinations has shape (total_combinations, num_bases)
                num_combinations = len(combinations)
                
                # Initialize output arrays
                output_array = np.zeros((scaled_photometries[0].shape[0], num_combinations))
                params_array = np.zeros((len(param_columns), num_combinations))
                supp_array = np.zeros((len(supp_param_keys), num_combinations))
                
                # Fill the arrays
                for i, combo_indices in enumerate(combinations):
                    # For photometry, combine the weighted contributions from each base
                    current_total = np.zeros_like(scaled_photometries[0][:, 0])
                    for j, base_idx in enumerate(combo_indices):
                        current_total += scaled_photometries[j][:, base_idx]
                    
                    output_array[:, i] = current_total
                    
                    # Fill parameter values
                    param_idx = 0
                    
                    # Add standard parameters
                    params_array[param_idx, i] = redshift
                    param_idx += 1
                    
                    params_array[param_idx, i] = np.log10(total_mass.value)
                    param_idx += 1
                    
                    if len(self.bases) > 1:
                        params_array[param_idx, i] = combination[0]  # weight fraction
                        param_idx += 1
                    
                    # Add all varying parameters from each base
                    for j, base in enumerate(self.bases):
                        for param_name in total_property_names[base.model_name]:
                            if param_name in base_param_values[j]:
                                params_array[param_idx, i] = base_param_values[j][param_name][combo_indices[j]]
                            param_idx += 1
                    
                    # Calculate supplementary parameters
                    for k, param_name in enumerate(supp_param_keys):
                        current_supp_value = 0
                        for j, base_idx in enumerate(combo_indices):
                            data = supp_params_values[j][param_name][base_idx]
                            if isinstance(data, unyt_array):
                                if j == 0:
                                    supp_param_units[param_name] = str(data.units)
                                current_supp_value += data.value
                            else:
                                if j == 0:
                                    supp_param_units[param_name] = str(dimensionless)
                                current_supp_value += data
                        
                        supp_array[k, i] = current_supp_value

            all_outputs.append(output_array)
            all_params.append(params_array)
            all_supp_params.append(supp_array)

        supp_param_units = [i for i in supp_param_units.values()]
        # Combine all outputs and parameters
        combined_outputs = np.hstack(all_outputs)
        combined_params = np.hstack(all_params)
        combined_supp_params = np.hstack(all_supp_params)
        
        print(f"Combined outputs shape: {combined_outputs.shape}")
        print(f"Combined parameters shape: {combined_params.shape}")
        print(f"Combined supplementary parameters shape: {combined_supp_params.shape}")
        print(f"Filter codes: {filter_codes}")

        out = {
            'photometry': combined_outputs,
            'parameters': combined_params,
            'parameter_names': param_columns,
            'filter_codes': filter_codes,
            'supplementary_parameters': combined_supp_params,
            'supplementary_parameter_names': supp_param_keys,
            'supplementary_parameter_units': supp_param_units,
        }

        self.grid_photometry = combined_outputs
        self.grid_parameters = combined_params
        self.grid_parameter_names = param_columns
        self.grid_filter_codes = filter_codes
        self.grid_supplementary_parameters = combined_supp_params
        self.grid_supplementary_parameter_names = supp_param_keys

        if save:
            self.save_grid(out, overload_out_name=overload_out_name, overwrite=overwrite)


    def create_full_grid_old2(self,
                override_instrument: Union[Instrument, None] = None,
                save: bool = True,
                overload_out_name: str = '',
                overwrite: bool = False,
                ) -> None:
        '''

        This is the annoying case where we have to create a grid of SEDs for a given set of parameters,
        but we're not sampling a grid so we have to generate the fullset of galaxies for both bases.

        '''

        assert self.draw_parameter_combinations == False, "Cannot create full grid with draw_parameter_combinations set to True. Set to False to create full grid."


        pipeline_outputs = self.load_bases()
        # Check all basis have the same number of galaxies
        ngal = len(pipeline_outputs[self.bases[0].model_name]['properties']['mass'])
        for i, base in enumerate(self.bases):
            if len(pipeline_outputs[base.model_name]['properties']['mass']) != ngal:
                raise ValueError(f"Base {i} has different number of galaxies to base 0. Cannot combine bases with different number of galaxies.")

        assert len(self.redshifts) == ngal, f"Redshift array length {len(self.redshifts)} does not match number of galaxies {ngal}. Cannot combine bases with different number of galaxies."
        assert len(self.total_stellar_masses) == ngal, f"Mass array length {len(self.total_stellar_masses)} does not match number of galaxies {ngal}. Cannot combine bases with different number of galaxies."
        assert len(self.combination_weights) == ngal, f"Combination weights array length {len(self.combination_weights)} does not match number of galaxies {ngal}. Cannot combine bases with different number of galaxies."
        
        base_filters = self.bases[0].instrument.filters.filter_codes
        for i, base in enumerate(self.bases):
            if base.instrument.filters.filter_codes != base_filters:
                raise ValueError(f"Base {i} has different filters to base 0. Cannot combine bases with different filters.")

        if override_instrument is not None:
            # Check all filters in override_instrument are in the base filters
            for filter_code in override_instrument.filters.filter_codes:
                if filter_code not in base_filters:
                    raise ValueError(f"Filter {filter_code} not found in base filters. Cannot override instrument.")
                
            filter_codes = override_instrument.filters.filter_codes
        else:
            filter_codes = base_filters

        filter_codes = [i.split('/')[-1] for i in filter_codes]

        all_outputs = []
        all_params = []
        all_supp_params = []
        ignore_keys = ['redshift']
        total_property_names = {}
        for j, base in enumerate(self.bases):
            total_property_names[base.model_name] = [f'{base.model_name}/{i}' for i in base.varying_param_names if i not in ignore_keys]
            params = pipeline_outputs[base.model_name]['properties']
            rename_keys = [g for g in base.varying_param_names if g not in ignore_keys]
            for key in list(params.keys()):
                if key in rename_keys:
                    # rename the key to be the base name + parameter name
                    params[f'{base.model_name}/{key}'] = params[key]
       
        # Check if any of the bases have the same varying parameters
        all_combined_param_names = []
        for key, value in total_property_names.items():
            all_combined_param_names.extend(value)
        
        # Add our standard parameters that are always included
        param_columns = ['redshift', 'log_mass']

        if len(self.bases) > 1:
            param_columns.append('weight_fraction')

        param_columns.extend(all_combined_param_names)

        # set supplementary parameters

        supp_param_keys = list(pipeline_outputs[self.bases[0].model_name]['supp_properties'].keys())
        assert all([i in pipeline_outputs[self.bases[0].model_name]['supp_properties'] for i in supp_param_keys]), f"Not all bases have the same supplementary parameters. {supp_param_keys} not found in {pipeline_outputs[self.bases[0].model_name]['supp_properties'].keys()}"
        supp_params = {}
        supp_param_units = {}
        for i, base in enumerate(self.bases):
            supp_params[base.model_name] = {}
            for key in supp_param_keys:
                if isinstance(pipeline_outputs[base.model_name]['supp_properties'][key], dict):
                    subkeys = list(pipeline_outputs[base.model_name]['supp_properties'][key].keys())
                    # Check if the emission model key is in the subkeys
                    if self.base_emission_model_keys[i] not in subkeys:
                        raise ValueError(f"Emission model key {self.base_emission_model_keys[i]} not found in {subkeys}. Don't know how to deal with dictionary supplementary parameters with other keys.")
                    value = pipeline_outputs[base.model_name]['supp_properties'][key][self.base_emission_model_keys[i]]
                else:
                    value = pipeline_outputs[base.model_name]['supp_properties'][key]
                
                supp_params[base.model_name][key] = value

        for pos, redshift in enumerate(self.redshifts):
            total_mass = self.total_stellar_masses[pos]
            combination = self.combination_weights[pos]

            mass_weights = np.array(combination) * total_mass

            scaled_photometries = []
            base_param_values = []
            supp_params_values = []

            for i, base in enumerate(self.bases):
                outputs = pipeline_outputs[base.model_name]
                z_base = outputs['properties']['redshift']
                mask = z_base == redshift
                mass = outputs['properties']['mass'][mask]
                if len(mass) == 0:
                    raise ValueError(f"No galaxies found for redshift {redshift} in base {base.model_name}. Check your redshift array.")

                # Calculate the scaling factor for each base
                if combination == 1.0:
                    scaling_factors = mass_weights / mass
                else:
                    scaling_factors = mass_weights[i] / mass

                #print(f"Scaling factors: {scaling_factors}")

                base_photometry = np.array([pipeline_outputs[base.model_name]['observed_photometry'][filter_code][mask] for filter_code in filter_codes], dtype=np.float32)

                # Scale the photometry by the scaling factor
                scaled_photometry = base_photometry * scaling_factors

                scaled_photometries.append(scaled_photometry)

                # Get the varying parameters for this base
                base_params = {}
                for param_name in total_property_names[base.model_name]:
                    # Extract the original parameter name without the base prefix
                    orig_param = param_name.split('/')[-1]
                    if f'{base.model_name}/{orig_param}' in outputs['properties']:
                        base_params[param_name] = outputs['properties'][f'{base.model_name}/{orig_param}'][mask]
                    elif orig_param in outputs['properties']:
                        base_params[param_name] = outputs['properties'][orig_param][mask]
                
                base_param_values.append(base_params)
                # Get the supplementary parameters for this base
                # For any supp params that are a flux or luminosity, scale them by the scaling factor
                scaled_supp_params = {}
                for key, value in supp_params[base.model_name].items():
                    #print(key, value)
                    if isinstance(value, dict):
                        scaled_supp_params[key] = {}
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, unyt_array):
                                scaled_supp_params[key][subkey] = subvalue[mask] * scaling_factors
                            else:
                                scaled_supp_params[key][subkey] = subvalue[mask]
                    else:
                        if isinstance(value, unyt_array):
                            scaled_supp_params[key] = value[mask] * scaling_factors
                        else:
                            scaled_supp_params[key] = value[mask]

                supp_params_values.append(scaled_supp_params)

            # Calculate the total number of combinations
            dimension = np.prod([i.shape[-1] for i in scaled_photometries])
            
            output_array = np.zeros((scaled_photometries[0].shape[0], dimension))
            params_array = np.zeros((len(param_columns), dimension))
            supp_array = np.zeros((len(supp_param_keys), dimension))

            # Create all combinations of indices
            combinations = np.meshgrid(*[np.arange(i.shape[-1]) for i in scaled_photometries], indexing='ij')
            combinations = np.array(combinations).T.reshape(-1, len(scaled_photometries))
            
            # Fill the output and parameter array.out_name
            for i, combo_indices in enumerate(combinations):
                # Add the scaled photometries for each base
                for j, base in enumerate(self.bases):
                    output_array[:, i] += scaled_photometries[j][:, combo_indices[j]]
                
                # Fill parameter values
                param_idx = 0
                
                # Add standard parameters first
                params_array[param_idx, i] = redshift
                param_idx += 1
                
                params_array[param_idx, i] = np.log10(total_mass.value)
                param_idx += 1
                
                if len(self.bases) > 1:
                    params_array[param_idx, i] = combination[0]  # assuming this is weight fraction
                    param_idx += 1
                
                # Add all varying parameters from each base
                for j, base in enumerate(self.bases):
                    for param_name in total_property_names[base.model_name]:
                        if param_name in base_param_values[j]:
                            params_array[param_idx, i] = base_param_values[j][param_name][combo_indices[j]]
                        param_idx += 1

                # Add supplementary parameters. Sum parameters from all bases
                for j, base in enumerate(self.bases):
                    for k, param_name in enumerate(supp_param_keys):
                        data = supp_params_values[j][param_name][combo_indices[j]]
                        if isinstance(data, unyt_array):
                            if j == 0:
                                supp_param_units[param_name] = str(data.units)
                            data = data.value
                        else:
                            if j == 0:
                                supp_param_units[param_name] = str(dimensionless)
                        
                        supp_array[k, i] += data

            all_outputs.append(output_array)
            all_params.append(params_array)
            all_supp_params.append(supp_array)

        supp_param_units = [i for i in supp_param_units.values()]
        # Combine all outputs and parameters
        combined_outputs = np.hstack(all_outputs)
        combined_params = np.hstack(all_params)
        combined_supp_params = np.hstack(all_supp_params)
        print(f"Combined outputs shape: {combined_outputs.shape}")
        print(f"Combined parameters shape: {combined_params.shape}")
        print(f"Combined supplementary parameters shape: {combined_supp_params.shape}")
        print(f"Combined parameters: {combined_params}")
        print(f"Filter codes: {filter_codes}")

        out = {
            'photometry': combined_outputs,
            'parameters': combined_params,
            'parameter_names': param_columns,
            'filter_codes': filter_codes,
            'supplementary_parameters': combined_supp_params,
            'supplementary_parameter_names': supp_param_keys,
            'supplementary_parameter_units': supp_param_units,
        }

        self.grid_photometry = combined_outputs
        self.grid_parameters = combined_params
        self.grid_parameter_names = param_columns
        self.grid_filter_codes = filter_codes
        self.grid_supplementary_parameters = combined_supp_params
        self.grid_supplementary_parameter_names = supp_param_keys

        if save:
            self.save_grid(out, overload_out_name=overload_out_name, overwrite=overwrite) 


class GalaxySimulator(object):
    '''
    Photometry Simulator class. Initialize with required model,
    then take an array or dictionary of parameters and return a dictionary of photometry.
    A seperate function will handle normalization and scaling of the photometry.

    '''


    def __init__(self,
                sfh_model:Type[SFH.Common],
                zdist_model:Type[ZDist.Common],
                grid: Grid,
                instrument: Instrument,
                emission_model: EmissionModel,
                emission_model_key: str,
                emitter_params: dict = {'stellar': [], 'galaxy':[]},
                cosmo: Cosmology = Planck18,
                param_order: Union[None, list] = None,
                param_units: dict = {},
                param_transforms: dict[callable] = {},
                out_flux_unit: str = 'nJy',
                required_keys = ['redshift', 'log_mass'],
                extra_functions: List[callable] = None,
                normalize_method: str = None,
                output_type: str = 'photo_fnu',
                set_self=False,
                depths: Union[np.ndarray, unyt_array] = None, 
                depth_sigma: int = 5,
                ) -> None:

        """
        Parameters

        ----------

        sfh_model : Type[SFH.Common]
            SFH model to use. Must be a subclass of SFH.Common.
        zdist_model : Type[ZDist.Common]
            ZDist model to use. Must be a subclass of ZDist.Common.
        grid : Grid
            Grid object to use. Must be a subclass of Grid.
        instrument : Instrument
            Instrument object to use for photometry/spectra. Must be a subclass of Instrument.
        emission_model : EmissionModel
            Emission model to use. Must be a subclass of EmissionModel.
        emission_model_key : str
            Emission model key to use e.g. 'total', 'intrinsic'.
        emitter_params : dict
            Dictionary of parameters to pass to the emitter. Keys are 'stellar' and 'galaxy'.
        cosmo : Cosmology = Planck18
            Cosmology object to use. Must be a subclass of Cosmology.
        param_order : Union[None, list] = None
            Order of parameters to use. If None, will use the order of the keys in the params dictionary.
        param_units : dict
            Dictionary of parameter units to use. Keys are the parameter names and values are the units.
        param_transforms : dict
            Dictionary of parameter transforms to use. Keys are the parameter names and values are the transforms functions.
            Can be used to fit say Av when model requires tau_v, or for unit conversion etc.
        out_flux_unit : str
            Output flux unit to use. Default is 'nJy'. Can be 'ABmag', 'Jy', 'nJy', 'uJy', 'mJy'.
        required_keys : list
            List of required keys to use. Default is ['redshift', 'log_mass']. Typically this can be ignored.
        extra_functions : List[callable]
            List of extra functions to call after the simulation.
            Inputs to each function are determined by the function signature and can be any of the following:
            galaxy, spec, fluxes, sed, stars, emission_model, cosmo.
            The output of each function is appended to an array and returned in the output tuple.
        normalize_method : str
            Normalization method to use. If None, no normalization is applied. Only works on photo_fnu currently.
            Currently can be either the name of a filter, or a callable function with the same rules as extra_functions.
        output_type : str
            Output type to use. Default is 'photo_fnu'. Can be 'photo_fnu', 'photo_lnu', 'fnu', 'lnu', or a list of any of these.     
        depths : Union[np.ndarray, unyt_array] = None
            Depths to use for the photometry. If None, no depths are applied. Default is None.
        depth_sigma : int
            Sigma to use for the depths. Default is 5. This is used to calculate the 1 sigma error in nJy.
            If depths is None, this is ignored.

        """
        
        assert isinstance(grid, Grid), f"Grid must be a subclass of Grid. Got {type(grid)} instead."
        self.grid = grid
        
        assert isinstance(instrument, Instrument), f"Instrument must be a subclass of Instrument. Got {type(instrument)} instead."
        assert isinstance(emission_model, EmissionModel), f"Emission model must be a subclass of EmissionModel. Got {type(emission_model)} instead."

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

        if depths is not None: assert len(depths) == len(instrument.filters.filter_codes), f"Depths array length {len(depths)} does not match number of filters {len(instrument.filters.filter_codes)}. Cannot create photometry."
        self.depths = depths
        self.depth_sigma = depth_sigma
        
        assert isinstance(output_type, list) or output_type in ['photo_fnu', 'photo_lnu', 'fnu', 'lnu'], f"Output type {output_type} not recognised. Must be one of ['photo_fnu', 'photo_lnu', 'fnu', 'lnu']"
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


        self.total_possible_keys = self.sfh_params + self.zdist_params + self.optional_sfh_params + self.optional_zdist_params + required_keys

    def simulate(self, params):

        params = copy.deepcopy(params)

        if not isinstance(params, dict):
            if self.param_order is None:
                raise ValueError('simulate() input requires a dictionary unless param_order is set. Cannot create photometry.')
            assert len(params) == len(self.param_order), f"Parameter array length {len(params)} does not match parameter order length {len(self.param_order)}. Cannot create photometry."
            params = {i:j for i, j in zip(self.param_order, params)}

        
        for key in self.required_keys:
            if key not in params:
                raise ValueError(f"Missing required parameter {key}. Cannot create photometry.")

        mass = 10**params['log_mass'] * Msun

        # Check if we have sfh_params and zdist_params

        for key in params:
            if key in self.param_units:
                params[key] = params[key] * self.param_units[key]

        for key in self.param_transforms:
            if key in params:
                params[key] = self.param_transforms[key](params[key])


        sfh = self.sfh_model(**{i: params[i] for i in self.sfh_params}, 
                              **{i: params[i] for i in self.optional_sfh_params if i in params})
        zdist = self.zdist_model(**{i: params[i] for i in self.zdist_params},
                                    **{i: params[i] for i in self.optional_zdist_params if i in params})

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
                raise ValueError(f"Parameter {key} not found in emitter params. Cannot create photometry.")
        
            else:
                found_params.append(key)

        # Check we understand all the parameters

        assert len(found_params) == len(self.emitter_params), f"Found {len(found_params)} parameters but expected {len(self.emitter_params)}. Cannot create photometry."

        stellar_keys = {}
        if 'stellar' in self.emitter_params:
            for key in found_params:
                stellar_keys[key] = params[key]

        stars = Stars(
            log10ages=self.grid.log10ages,
            metallicities=self.grid.metallicity,
            sf_hist=sfh,
            metal_dist=zdist,
            initial_mass=mass,
            **stellar_keys,
        )

        galaxy_keys = {}
        if 'galaxy' in self.emitter_params:
            for key in found_params:
                galaxy_keys[key] = params[key]

        galaxy = Galaxy(
            stars=stars,
            redshift = params['redshift'],
            **galaxy_keys,
        )

        # Get the spectra for the galaxy
        spec = galaxy.stars.get_spectra(self.emission_model)
        outputs = {}

        if 'sfh' in self.output_type:

            stars_sfh = stars.get_sfh()
            stars_sfh = stars_sfh / np.diff(10**(self.grid.log10age), prepend=0) / yr
            time = 10**(self.grid.log10age) * yr
            time = time.to('Myr')

            # Check if any NANS in SFH. If there are, print sfh
            #if np.isnan(sfr).any():
            #    print(f"SFH has NANS: {sfh.parameters}")
                
            outputs['sfh'] = stars_sfh
            outputs['sfh_time'] = time
            outputs['redshift'] = galaxy.redshift
            # Put an absolute SFH time here as well given self.cosmo and redshift
            outputs['sfh_time_abs'] = self.cosmo.age(galaxy.redshift).to('Myr').value * Myr
            outputs['sfh_time_abs'] = outputs['sfh_time_abs'] - time

        if 'lnu' in self.output_type: 
            fluxes = spec.lnu
            outputs['lnu'] = fluxes

        if 'photo_lnu' in self.output_type:
            fluxes = galaxy.stars.spectra[self.emission_model_key].get_photo_lnu(self.instrument.filters)
            fluxes = fluxes.photo_lnu
            outputs['photo_lnu'] = fluxes

        if 'fnu' in self.output_type or 'photo_fnu' in self.output_type:
            # Apply IGM and distance
            galaxy.get_observed_spectra(self.cosmo)

            if 'photo_fnu' in self.output_type:
                fluxes = galaxy.stars.spectra[self.emission_model_key].get_photo_fnu(self.instrument.filters)
                outputs['photo_fnu'] =  fluxes.photo_fnu
                outputs['photo_wav'] = fluxes.filters.pivot_lams

                fluxes = galaxy.stars.spectra[self.emission_model_key].fnu
                outputs['fnu'] = fluxes
                outputs['fnu_wav'] = galaxy.stars.spectra[self.emission_model_key].lam * (1+galaxy.redshift)

        if self.out_flux_unit == 'ABmag':
            convert = lambda f: -2.5 * np.log10(f.to(Jy).value) + 8.9
            if 'photo_fnu' in self.output_type:
                fluxes = convert(outputs['photo_fnu'])
                outputs['photo_fnu'] = fluxes
            if 'fnu' in self.output_type:
                fluxes = convert(outputs['fnu'])
                outputs['fnu'] = fluxes
        else:
            if 'photo_fnu' in self.output_type:
                outputs['photo_fnu'] = fluxes.to(self.out_flux_unit).value
            if 'fnu' in self.output_type:
                outputs['fnu'] = fluxes.to(self.out_flux_unit).value

        if len(self.output_type) == 1:
            fluxes = outputs[self.output_type[0]]
        else:
            outputs['filters'] = self.instrument.filters
            return outputs

        def inspect_func(func, locals):
            parameters = inspect.signature(func).parameters
            inputs = {}
            possible_inputs = ['galaxy', 'spec', 'fluxes', 'sed', 'stars', 'emission_model', 'cosmo']
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
            # possible arguments are galaxy, spec, fluxes, sed, stars, emission_model, cosmo
            
            output = []
            for func in self.extra_functions:
                if isinstance(func, tuple):
                    func, args = func
                    output.append(func(galaxy, *args))
                else:
                    inputs = inspect_func(func, locals())
                    output.append(func(**inputs))

        
        fluxes = self._scatter(fluxes, flux_units=self.out_flux_unit)

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

        return fluxes

    def _normalize(self, fluxes, method=None, norm_unit='ABmag', add_norm_pos=-1):
        if method is None:
            return fluxes

        if norm_unit == 'ABmag':
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
                raise ValueError(f"Filter {method} not found in filter codes. Cannot normalize photometry.")
        elif isinstance(method, (unyt_array, unyt_quantity)):
            if norm_unit == 'ABmag':
                # Convert to ABmag
                method = -2.5 * np.log10(method.to(Jy).value) + 8.9

            norm = method
            fluxes = func(fluxes, norm)
        elif callable(method):
            norm = method(fluxes)
            fluxes = func(fluxes, norm)

        if add_norm_pos is not None:
            # Insert the normalization value at this position, increasing the size of the array
            if add_norm_pos == -1:
                fluxes = np.append(fluxes, norm)
            else:
                fluxes = np.insert(fluxes, add_norm_pos, norm)

        return fluxes

    def _scatter(self,
                fluxes: np.ndarray,
                flux_units: str = 'nJy',
            ):
        '''
        Given a set of depths, convert to photometry unit and use as standard deviations
        to generate a scattered photometry array of shape (m*n_scatters, n) from photometry array
        '''

        depths = self.depths

        if depths is None:
            return fluxes

        depths = np.array(depths)

        m = fluxes.shape

        if type(fluxes) == unyt_array:
            assert fluxes.units == flux_units, f"Fluxes units {fluxes.units} do not match flux units {flux_units}. Cannot scatter photometry."
        
        if flux_units == 'ABmag':
            fluxes = (10**((fluxes-23.9)/-2.5)) * uJy
    
        # Convert depths based on units
        if self.out_flux_unit == 'ABmag' and not hasattr(depths, 'unit'):
            # Convert from AB magnitudes to microjanskys
            depths_converted = (10**((depths-23.9)/-2.5)) * uJy
            depths_std = depths_converted.to(uJy).value / self.depth_sigma
        else:
            depths_std = depths.to(self.out_flux_unit).value / self.depth_sigma
        # Pre-allocate output array with correct dimensions
        output_arr = np.zeros(m)
        
        # Generate all random values at once for better performance
        random_values = np.random.normal(loc=0, scale=depths_std, size=(m)) * uJy
        # Add the random values to the fluxes        
        output_arr = fluxes + random_values

        
        if flux_units == 'ABmag':
            # Convert back to AB magnitudes
            output_arr = -2.5 * np.log10(output_arr.to(Jy).value) + 8.9

        return output_arr

    def __call__(self, params):
        return self.simulate(params)

    def __repr__(self):
        return f"PhotometrySimulator({self.sfh_model}, {self.zdist_model}, {self.grid}, {self.instrument}, {self.emission_model}, {self.emission_model_key})"

    def __str__(self):
        return f"PhotometrySimulator({self.sfh_model}, {self.zdist_model}, {self.grid}, {self.instrument}, {self.emission_model}, {self.emission_model_key})"


def calculate_min_max_wav_grid(filterset, max_redshift, min_redshift=0):
    """
    Calculate the minimum and maximum wavelengths for a given redshift.
    """
    # Get the filter limits
    filter_lims = filterset.get_non_zero_lam_lims()
    
    # Calculate the maximum wavelength for a given redshift
    max_wav = filter_lims[1] / (1 + min_redshift)
    
    # Calculate the minimum wavelength for a given redshift
    min_wav = filter_lims[0] / (1 + max_redshift)
    
    return min_wav, max_wav



                


    



if __name__ == "__main__":
    print("This is a module, not a script.")

# TODO
# Add saving photometry unit to the output.hdf5 file - DONE
# Need to use weights when combining supplemental parameters like MUV - DONE
# Switch to using estimated MUV for normalisation, which is closer to the observed MUV
# Make weight fraction clearer as to which base it is for.
# Use Latin Hypercube sampling for the grid generation DONE