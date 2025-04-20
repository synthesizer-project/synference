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
from synthesizer.instruments import Instrument
from typing import Dict, Any, List, Tuple, Union, Optional, Type
from abc import ABC, abstractmethod
import copy
from scipy.stats import uniform, loguniform
from unyt import unyt_array, unyt_quantity, erg, cm, s, Angstrom, um, Hz, m, nJy, K, Msun, Myr, yr, Unit, kg
from astropy.cosmology import Planck18, Cosmology, z_at_value
from synthesizer.emission_models.attenuation import Inoue14
import astropy.units as u
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from tqdm import tqdm

warnings.filterwarnings('ignore')

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
    for i, params in tqdm(enumerate(varying_param_combinations)):
        # Create parameter dictionary for emission model constructor
        emission_params = {
            key: params[j] for j, key in enumerate(varying_params.keys())
        }


        # Apply units if not None
        for key, value in emission_params.items():
            if 'units' in varying_params[key] and varying_params[key]['units'] is not None:
                emission_params[key] = value * varying_params[key]['units']
        
        emission_params.update(fixed_params)
        
        # Create and append emission model instance
        emission_model_instance = emission_model(grid=grid, **emission_params)
        emission_models.append(emission_model_instance)


    return emission_models

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


class sed_grid_generator():
    """
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

                import matplotlib.patheffects as PathEffects
                ax[0].plot(phot.filters.pivot_lams, phot.photo_fnu, '+', color=colors[key], path_effects=[PathEffects.withStroke(linewidth=4, foreground='white')])

                if not isinstance(gal, Galaxy):
                    continue

                # Get the redshift
                redshift = gal.redshift

                # Get the SFH
                stars_sfh = gal.stars.get_sfh()
                stars_sfh = stars_sfh / np.diff(10**(self.grid.log10age), prepend=0) / yr
                t, sfh = gal.stars.sf_hist_func.calculate_sfh()

                ax[1].plot(10**(self.grid.log10age-6), stars_sfh, label=f'{key} SFH', color=colors[key])
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
        seds = []

        # Iterate over all combinations of parameters
        for sfh in tqdm(self.sfhs):
            for metal_dist in self.metal_dists:
                for emission_model in self.emission_models:
                    for logm in self.masses:
                        # Create a new SED object with the current parameters
                        sed = self.create_galaxy(sfh, metal_dist, emission_model, logm, sfh.redshift)
                        seds.append(sed)

        return seds
    
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
                popIII_galaxy.get_observed_spectra(cosmo=self.cosmo, igm=self.igm)


        if get_spectra:
            sed = galaxy.stars.get_spectra(emission_model)
            galaxy.get_observed_spectra(cosmo=self.cosmo, igm=self.igm)

            if popIII_sfh is not None:
                combined_galaxy_sed = galaxy.stars.spectra['total'] + popIII_galaxy.stars.spectra['intrinsic']
            else:
                combined_galaxy_sed = galaxy.stars.spectra['total']
                 
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
