import sys
import os
import h5py
import scipy.stats
import warnings
import numpy as np
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
from synthesizer.conversions import fnu_to_lnu
from typing import Dict, Any, List, Tuple, Union, Optional, Type
from abc import ABC, abstractmethod
import copy
from scipy.stats import uniform, loguniform
from unyt import unyt_array, unyt_quantity, erg, cm, s, Angstrom, um, Hz, m, nJy, K, Msun, Myr, yr, Unit, kg, Mpc
from astropy.cosmology import Planck18, Cosmology, z_at_value
from synthesizer.emission_models.attenuation import Inoue14
import astropy.units as u
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from tqdm import tqdm
from synthesizer.pipeline import Pipeline
import matplotlib.patheffects as PathEffects

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
        phot = fnu_to_lnu(lnu, cosmo=cosmo, redshift=z)
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

def generate_sfh_basis(
        sfh_type: Type[SFH.Common],
        sfh_param_names: List[str],
        sfh_param_arrays: List[np.ndarray],
        sfh_param_units: List[Union[None, Unit]],
        redshifts: Union[Dict[str, Any], float, np.ndarray],
        max_redshift: float = 15,
        calculate_min_age: bool = True,
        min_age_frac = 0.001, 
        cosmo: Type[Cosmology] = Planck18,
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

def generate_constant_R(R=300, start=1*Angstrom, end=9e5*Angstrom):
    
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
        
        if self.redshift_dependent_sfh:
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

        hashes = []
        for gal in self.galaxies:
            hash_i = 0
            for key in varying_param_names:
                if key in gal.all_params:
                    if isinstance(gal.all_params[key], unyt_array) or isinstance(gal.all_params[key], unyt_quantity):
                        hash_i += hash(float(gal.all_params[key].value))
                    else:
                        hash_i += hash(gal.all_params[key])
            hashes.append(hash_i)
        if len(hashes) != len(np.unique(hashes)):
            raise ValueError("Varying parameters are not unique across galaxies. Check your input parameters.")

        self.varying_param_names = varying_param_names
        self.fixed_param_names = fixed_param_names
        self.all_parameters = all_parameters
        

        for key in to_remove:
            all_parameters.pop(key)

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

        # Define the number of stellar particles we want
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

    def process_galaxies(self, 
                        galaxies: List[Type[Galaxy]],
                        out_name: str = 'auto',
                        out_dir: str = 'self',
                        n_proc: int = 4,
                        verbose: int = 1,
                        save: bool = True,
                        **extra_analysis_functions
                        ) -> Pipeline:

        self.emission_model.set_per_particle(self.per_particle)

        pipeline = Pipeline(
            emission_model=self.emission_model,
            instruments=self.instrument,
            nthreads=n_proc,
            verbose=verbose,
        )


        for key in self.all_parameters.keys():
            pipeline.add_analysis_func(lambda gal, key=key: gal.all_params[key], result_key=key)
            
        pipeline.add_analysis_func(lambda gal: gal.stars.initial_mass, result_key='mass')

        # Add any extra analysis functions requested by the user. 

        for key, params in extra_analysis_functions.items():
            if callable(params):
                func = params
            else:
                func = params[0]
                params = params[1:]
            
            pipeline.add_analysis_func(func, f'supp_{key}', *params)

        pipeline.add_galaxies(galaxies)
        pipeline.get_spectra()
        pipeline.get_observed_spectra(self.cosmo)
        pipeline.get_photometry_luminosities()
        pipeline.get_photometry_fluxes()

        pipeline.run()

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

        return pipeline

class CombinedBasis:
    def __init__(self, 
                bases: List[Type[GalaxyBasis]],
                total_stellar_masses: unyt_array,
                redshifts: np.ndarray,
                base_emission_model_keys: List[str],
                combination_weights: np.ndarray,
                out_name='combined_basis',
                out_dir='../output/',
                base_mass=1e9 * Msun,
                ) -> None:
        

        self.bases = bases
        self.total_stellar_masses = total_stellar_masses
        self.redshifts = redshifts
        self.combination_weights = combination_weights
        self.out_name = out_name
        self.out_dir = out_dir
        self.base_mass = base_mass
        self.base_emission_model_keys = base_emission_model_keys

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
            full_out_path = f"{self.out_dir}/{self.out_name}_{base.model_name}.hdf5"

            if os.path.exists(full_out_path) and not overwrite[i]:
                print(f"File {full_out_path} already exists. Skipping.")
                continue
            elif os.path.exists(full_out_path) and overwrite[i]:
                print(f"File {full_out_path} already exists. Overwriting.")
                os.remove(full_out_path)
            elif not os.path.exists(self.out_dir):
                os.makedirs(self.out_dir)

            # Create galaxies for each base. Only create for one mass. We will scale later.
            galaxies = base.create_galaxies(self.base_mass)

            print(f'Created {len(galaxies)} galaxies for base {base.model_name}')
            # Process the galaxies
            pipeline = base.process_galaxies(galaxies,  
                                            f"{self.out_name}_{base.model_name}.hdf5",
                                            out_dir=self.out_dir,
                                            n_proc=n_proc, verbose=1, save=True, 
                                            **extra_analysis_functions
                                            )
           
    def load_bases(self, indexes: List[int] = None) -> dict:
        # Load the bases from the files

        outputs = {}
        for i, base in enumerate(self.bases):

            print(f'Emission model key for base {base.model_name}: {self.base_emission_model_keys[i]}')

            full_out_path = f"{self.out_dir}/{self.out_name}_{base.model_name}.hdf5"
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
                out_name: str = 'output.hdf5',
                overwrite: bool = False,
                ) -> dict:    
        """
        Create a grid of SEDs for the given bases and stellar masses from the pipeline outputs.
        Can override the instrument used to compute the SEDs if you want to subset which filters are used in the grid.
        E.g. the base model is run for all NIRCam wide and medium filters, but you want to create a grid for just the wide filters.
        """

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
        assert all([i in pipeline_outputs[self.bases[0].model_name]['supp_properties'] for i in supp_param_keys]), f"Not all bases have the same supplementary parameters. {supp_params} not found in {pipeline_outputs[bases[0].model_name]['supp_properties'].keys()}"


        # Deal with any supplementary model parameters. Currently we require that all bases have the same supplementary parameters and add them 
        supp_params = {}
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
        param_columns = ['redshift', 'log_mass', 'weight_fraction']
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
                        supp_params_values.append(supp_params[base.model_name])

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
                                    data = data.value

                                supp_array[k, i] += data


                    all_outputs.append(output_array)
                    all_params.append(params_array)
                    all_supp_params.append(supp_array)
        
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
                  out_name: str = 'grid.hdf5',
                  overwrite: bool = False,
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
            # Create a dataset for the parameter names
            f.attrs['ParameterNames'] = grid_dict['parameter_names']
            f.attrs['FilterCodes'] = grid_dict['filter_codes']

            # Add anything else as a dataset
            for key, value in grid_dict.items():
                if key not in ['photometry', 'parameters', 'parameter_names', 'filter_codes']:
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

class sed_grid_generator:
    """

    OBSELETE

    Abstract base class for generating grids of data.
    """
    
    def __init__(self, 
                sps_grid:Grid, 
                instrument:Instrument,
                cosmo: Type[Cosmology] = Planck18,
                redshifts: np.ndarray = np.arange(5, 15, 0.1),
                emission_models: List[Type[EmissionModel]] = None,
                sfhs: List = None,
                metal_dists: List[Type[ZDist.Common]] = None,
                logm_range: tuple = (4, 12, 0.1),
                truncate_sfh_at_z: float = 30,
                popIII_sps_grid: Grid = None,
                popIII_sfhs: List[Type[SFH.Common]] = None,
                popIII_metal_dists: List[Type[ZDist.Common]] = None,
                popIII_emission_models: List[Type[EmissionModel]] = None,
                mass_fraction_range: np.ndarray = None,
                igm: Type[ABC] = Inoue14,
                emission_keys: dict = {'Pop II': 'total', 'Pop III': 'intrinsic'},
                ) -> None:
                
        '''

        Parameters
        ----------
        sps_grid : Grid
            Grid object containing the SPS grid.
        instrument : Instrument
            Instrument object containing the filters.
        cosmo : Type[Cosmology], optional
            Cosmology object, by default Planck18
        redshifts : np.ndarray, optional
            Redshift array, by default np.arange(5, 15, 0.1)
        emission_model : Type[EmissionModel], optional
            Emission model object, by default None
        sfh : Type[Common], optional
            Star formation history object, by default None
        metal_dists : Type[ZDist.Common], optional
            Metallicity distribution objects, by default None
        logm_range : tuple, optional
            Log mass range (min, max, step), by default (4, 12, 0.1)
        truncate_sfh_at_z : float, optional
            Redshift at which to truncate the SFH, by default 30
        popIII_sps_grid : Grid, optional
            Extra SPS grid object, by default None
        mass_fraction_range : np.ndarray, optional
            Split mass between pop III grid and main grid, by default None

            
        Now the goal is basically to create a grid of SEDs for a given set of parameters.
        The parameters are:
            - logm_range: range of stellar masses to sample
            - redshift: range of redshifts to sample
            - sfh: star formation history object
            - metal_dists: metallicity distribution objects
            - emission_model: emission model object (typically fixed.)

        May need custom code to handle having a sensible SFH cutoff (e.g. no 10 Gyr old stars at z=10). Done. 

        Typical use atm has number of SFHs match number of redshifts - e.g. redshifts are not unique. 

        '''


        self.grid = sps_grid
        self.instrument = instrument
        self.cosmo = cosmo
        self.redshifts = redshifts
        self.emission_models = emission_models
        self.sfhs = sfhs
        self.metal_dists = metal_dists
        self.logm_range = logm_range
        self.truncate_sfh_at_z = truncate_sfh_at_z

        # Pop III
        self.popIII_sps_grid = popIII_sps_grid
        self.popIII_sfhs = popIII_sfhs
        self.popIII_metal_dists = popIII_metal_dists
        self.popIII_emission_models = popIII_emission_models
        self.mass_fraction_range = mass_fraction_range

        if self.popIII_sps_grid is not None:
            self.has_popIII = True
        else:
            self.has_popIII = False
        
        self.igm = igm
        self.emission_keys = emission_keys



        self.masses = unyt_array(np.power(np.arange(self.logm_range[0], self.logm_range[1], self.logm_range[2]), 10), Msun)

        if mass_fraction_range is not None:
            self.mass_fractions = np.arange(mass_fraction_range[0], mass_fraction_range[1], mass_fraction_range[2])
        else:
            self.mass_fractions = np.array([0.0])

        # Calculate total number of models. e.g. len(sfh) * len(metal_dists) * len(emission_models) * len(logm_range)

        self.N_models = len(self.sfhs) * len(self.metal_dists) * len(self.emission_models) * len(self.masses) * len(self.mass_fractions) 

        if self.has_popIII:
            self.N_models *= len(self.popIII_sfhs) * len(self.popIII_metal_dists) * len(self.popIII_emission_models)
        
        self.N_models = int(self.N_models)

         
    def draw_random_sed(self, plot=True, seed=None):
        """
        Draw a random SED from the grid of SEDs (for plotting etc.)

        TODO: Plot SFH, metallicity, emission model, etc. as well.
        """

        if seed is not None:
            np.random.seed(seed)

        # Randomly select parameters from the grid
        sfh_index = np.random.randint(0, len(self.sfhs))
        metal_dist_index = np.random.randint(0, len(self.metal_dists))
        emission_model_index = np.random.randint(0, len(self.emission_models))
        logm_index = np.random.randint(0, len(self.masses))

        sfh = self.sfhs[sfh_index]
        metal_dist = self.metal_dists[metal_dist_index]
        emission_model = self.emission_models[emission_model_index]
        logm = self.masses[logm_index]

        if self.has_popIII:
            popIII_sfh_index = np.random.randint(0, len(self.popIII_sfhs))
            popIII_metal_dist_index = np.random.randint(0, len(self.popIII_metal_dists))
            popIII_emission_model_index = np.random.randint(0, len(self.popIII_emission_models))
            popIII_sfh = self.popIII_sfhs[popIII_sfh_index]
            popIII_metal_dist = self.popIII_metal_dists[popIII_metal_dist_index]
            popIII_emission_model = self.popIII_emission_models[popIII_emission_model_index]
            pop_III_mass_fraction = self.mass_fractions[np.random.randint(0, len(self.mass_fractions))]

        else:
            popIII_sfh = None
            popIII_metal_dist = None
            popIII_emission_model = None
            pop_III_mass_fraction = 0.0

        # Get the redshift

        # Create a new SED object with the selected parameters
        galaxy = self.create_galaxy(sfh, metal_dist, emission_model, logm, redshift=sfh.redshift, return_combined_sed=False, 
                                    popIII_sfh=popIII_sfh, popIII_metal_dist=popIII_metal_dist, 
                                    popIII_emission_model=popIII_emission_model, pop_III_mass_fraction=pop_III_mass_fraction)


        if plot:

            fig, ax = plt.subplots(1, 2, figsize=(10, 5), layout='constrained') 

            if not isinstance(galaxy, dict):
                galaxy = {'Galaxy': galaxy}
           
            plot_dict = {key: gal.stars.spectra[self.emission_keys.get(key, 'total')] for key, gal in galaxy.items()}


            if len(galaxy.keys()) > 1:
                plot_dict['Combined'] = np.sum(list(plot_dict.values()), axis=0)
                galaxy['Combined'] = plot_dict['Combined']

            colors = {key: plt.cm.viridis(i/len(galaxy)) for i, key in enumerate(plot_dict.keys())}
            
     

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
            for key, gal in galaxy.items():
                if isinstance(gal, Galaxy):
                    emission_model = self.emission_keys.get(key, 'total')
                    sed = gal.stars.spectra[emission_model]
                else:
                    sed = gal

                # Plot photometry
                phot = sed.get_photo_fnu(filters=self.instrument)

                min_x = min(min_x, np.nanmin(phot.filters.pivot_lams))
                max_x = max(max_x, np.nanmax(phot.filters.pivot_lams))
                min_y = min(min_y, np.nanmin(phot.photo_fnu))
                max_y = max(max_y, np.nanmax(phot.photo_fnu))

                ax[0].plot(phot.filters.pivot_lams, phot.photo_fnu, '+', color=colors[key], path_effects=[PathEffects.withStroke(linewidth=4, foreground='white')])

                if not isinstance(gal, Galaxy):
                    continue

                # Get the redshift
                redshift = gal.redshift

                if key == 'Pop III':
                    grid = self.popIII_sps_grid
                else:
                    grid = self.grid
                    
                # Get the SFH
                stars_sfh = gal.stars.get_sfh()
                stars_sfh = stars_sfh / np.diff(10**(grid.log10age), prepend=0) / yr
                t, sfh = gal.stars.sf_hist_func.calculate_sfh()

                ax[1].plot(10**(grid.log10age-6), stars_sfh, label=f'{key} SFH', color=colors[key])
                ax[1].plot(t/1e6, sfh/np.max(sfh) * np.max(stars_sfh), label=f'Requested {key} SFH', color=colors[key], linestyle='--')

                mass = gal.stars.initial_mass
                if mass == 0:
                   text_gal[key] = f'{key}     \nNo stars'
                else:
                    age = gal.stars.calculate_mean_age()
                    zmet = gal.stars.calculate_mean_metallicity()

                    text_gal[key] = f'{key}     \nAge {age.to(Myr):.0f}\n$\log_{{10}}$Z: {np.log10(zmet.value):.2f}\nM$_\star$: {np.log10(mass):.1f} M$_\odot$'

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

            max_age = self.cosmo.age(redshift) - self.cosmo.age(self.truncate_sfh_at_z)
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

            return fig

        return galaxy

    def generate_grid(self):

        """
        Generate a grid of SEDs based on the specified parameters.
        """

        # Create a list to store the generated SEDs
        galaxies = []
        params = []
        # sfhs, metal_dists, emission_models, masses, ratios, popIII_sfhs, popIII_metal_dists, popIII_emission_models
        # Iterate over all combinations of parameters
        for sfh in tqdm(self.sfhs):
            for metal_dist in self.metal_dists:
                for emission_model in self.emission_models:
                    for logm in self.masses:
                        if self.has_popIII:
                            for popIII_sfh in self.popIII_sfhs:
                                for popIII_metal_dist in self.popIII_metal_dists:
                                    for popIII_emission_model in self.popIII_emission_models:
                                        for pop_III_mass_fraction in self.mass_fractions:
                                            # Create a new SED object with the current parameters
                                            galaxy = self.create_galaxy(sfh, metal_dist, emission_model, logm, sfh.redshift,
                                                                    popIII_sfh=popIII_sfh,
                                                                    pop_III_mass_fraction=pop_III_mass_fraction,
                                                                    popIII_metal_dist=popIII_metal_dist,
                                                                    popIII_emission_model=popIII_emission_model)
                                            galaxies.append(galaxy)
                                            params.append([sfh.redshift, sfh, metal_dist, emission_model, logm,
                                                            popIII_sfh, pop_III_mass_fraction, popIII_metal_dist, popIII_emission_model])
                            else:
                                        
                                # Create a new SED object with the current parameters
                                galaxy = self.create_galaxy(sfh, metal_dist, emission_model, logm, sfh.redshift)
                                galaxies.append(galaxy)
                                params.append([sfh.redshift, sfh, metal_dist, emission_model, logm])
                                

        # Group into grid/emission model, and run Pipeline to generate SEDs/photometry.

        # Store photometry, redshift and parameters of model in HDF5 file. 

        return grid
    
    def create_galaxy(self,
                        sfh: SFH,
                        metal_dist: ZDist.Common,
                        emission_model: Type[EmissionModel],
                        mstar: float,
                        redshift: float,
                        popIII_sfh: SFH = None,
                        pop_III_mass_fraction: float = 0.0,
                        popIII_metal_dist: ZDist.Common = None,
                        popIII_emission_model: Type[EmissionModel] = None,
                        get_spectra: bool = True,
                        return_combined_sed: bool = True,
                        ) -> Union[Galaxy, dict]:
                      
        """
        Create a new SED object with the specified parameters.
        """

        total_mass = mstar

        # Calculate the mass of the Pop III stars
        popIII_mass = total_mass * pop_III_mass_fraction
        # Calculate the mass of the Pop II stars
        popII_mass = total_mass * (1 - pop_III_mass_fraction)
        
        # Create the Pop II stars
        stars = Stars(
            self.grid.log10age,
            self.grid.metallicity,
            sf_hist=sfh,
            metal_dist=metal_dist,
            initial_mass=popII_mass,
        )

        galaxy = Galaxy(
            stars=stars,
            redshift=redshift,
        )

        # Create the Pop III stars if specified

        if self.has_popIII and popIII_sfh is not None:
            popIII_stars = Stars(
                self.popIII_sps_grid.log10age,
                self.popIII_sps_grid.metallicity,
                sf_hist=popIII_sfh,
                metal_dist=popIII_metal_dist,
                initial_mass=popIII_mass,
            )

            popIII_galaxy = Galaxy(
                stars=popIII_stars,
                redshift=redshift, #hmmm 
            )


            if get_spectra:
                popIII_sed = popIII_galaxy.stars.get_spectra(popIII_emission_model)
                #popIII_sed.lnu = np.squeeze(popIII_sed.lnu)
                popIII_galaxy.get_observed_spectra(cosmo=self.cosmo, igm=self.igm)


        if get_spectra:
            sed = galaxy.stars.get_spectra(emission_model)
            galaxy.get_observed_spectra(cosmo=self.cosmo, igm=self.igm)

            if popIII_sfh is not None:
                combined_galaxy_sed = galaxy.stars.spectra[self.emission_keys['Pop II']] + popIII_galaxy.stars.spectra[self.emission_keys['Pop III']]
            else:
                combined_galaxy_sed = galaxy.stars.spectra[self.emission_keys['Pop II']]
                 
            phot = combined_galaxy_sed.get_photo_fnu(filters=self.instrument)
            

        if return_combined_sed:
            return combined_galaxy_sed
        
        if popIII_sfh is not None:
            output = {'Pop II': galaxy, 'Pop III': popIII_galaxy}
        else:
            output = galaxy

        return output

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


if __name__ == "__main__":
    print("This is a module, not a script.")

# TODO
# Add saving photometry unit to the output.hdf5 file.
# Need to use weights when combining supplemental parameters like MUV. 