"""File containing Synthesizer grid generation and galaxy creation utilities."""

import copy
import inspect
import os
import threading
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import astropy.units as u
import h5py
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import Cosmology, Planck18, z_at_value
from dill.source import getsource
from joblib import Parallel, delayed, parallel_config
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from scipy.linalg import inv
from scipy.stats import qmc

from . import logger

try:
    from synthesizer.conversions import lnu_to_fnu
    from synthesizer.emission_models import EmissionModel
    from synthesizer.emission_models.attenuation import Inoue14
    from synthesizer.emissions import plot_spectra
    from synthesizer.grid import Grid
    from synthesizer.instruments import UVJ, FilterCollection, Instrument
    from synthesizer.parametric import SFH, Galaxy, Stars, ZDist
    from synthesizer.particle.stars import sample_sfzh
    from synthesizer.pipeline import Pipeline

    synthesizer_available = True
except Exception:
    logger.warning(
        "Synthesizer dependencies not installed. Only the SBI functions will be available."
    )
    synthesizer_available = False

from tqdm import tqdm
from unyt import (
    Angstrom,
    Jy,
    Mpc,
    Msun,
    Myr,
    Unit,
    define_unit,
    dimensionless,
    nJy,
    uJy,
    um,
    unyt_array,
    unyt_quantity,
    yr,
)

from .noise_models import (
    EmpiricalUncertaintyModel,
)
from .utils import check_log_scaling, check_scaling, list_parameters, save_emission_model

file_path = os.path.dirname(os.path.realpath(__file__))
grid_folder = os.path.join(os.path.dirname(os.path.dirname(file_path)), "grids")
# Global variables for thread-shared data (initialized once per process)
_thread_local = threading.local()


UNIT_DICT = {
    "log10metallicity": "log10(Zmet)",
    "metallicity": "Zmet",
    "av": "mag",
    "tau_v": "mag",
    "tau_v_ism": "mag",
    "tau_v_birth": "mag",
    "weight_fraction": "dimensionless",
    "log_sfr": "log10(Msun/yr)",
    "sfr": "Msun/yr",
    "log_stellar_mass": "log10(Msun)",
    "log_surviving_mass": "log10(Msun)",
    "stellar_mass": "Msun",
}


uvj = UVJ()
uvj = {
    "U": FilterCollection(filters=[uvj["U"]]),
    "V": FilterCollection(filters=[uvj["V"]]),
    "J": FilterCollection(filters=[uvj["J"]]),
}

tophats = {
    "MUV": {"lam_eff": 1500 * Angstrom, "lam_fwhm": 100 * Angstrom},
}

muv_filter = FilterCollection(tophat_dict=tophats, verbose=False)


try:
    define_unit("log10_Msun", 1 * dimensionless)
except RuntimeError:
    pass

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

except ImportError:
    rank = 0
    size = 1
    comm = None
# ------------------------------------------
# Functions for galaxy parameters
# ------------------------------------------


if not synthesizer_available:
    # Make dummy classes for type checking
    class SFH:
        """Dummy class for SFH to allow type checking without synthesizer installed."""

        class Common:
            """Dummy class."""

            pass

    class ZDist:
        """Dummy class."""

        class Common:
            """Dummy class."""

            pass

    class EmissionModel:
        """Dummy class."""

        pass

    class Galaxy:
        """Dummy class."""

        pass

    class Grid:
        """Dummy class."""

        pass

    class Instrument:
        """Dummy class."""

        pass

    class Pipeline:
        """Dummy class."""

        pass


def calculate_muv(galaxy, cosmo=Planck18):
    """Calculate the apparent mUV magnitude of a galaxy.

    Parameters
    ----------
    galaxy : Galaxy
        The galaxy object containing stellar spectra.
    cosmo : Cosmology, optional
        The cosmology to use for redshift calculations, by default Planck18.

    Returns:
    -------
    dict
        Dictionary containing the MUV malogginggnitude for each stellar spectrum in the galaxy.
    """
    z = galaxy.redshift

    phots = {}

    for key in list(galaxy.stars.spectra.keys()):
        lnu = galaxy.stars.spectra[key].get_photo_lnu(muv_filter).photo_lnu[0]
        phot = lnu_to_fnu(lnu, cosmo=cosmo, redshift=z)
        phots[key] = phot

    return phots


def calculate_MUV(galaxy, cosmo=Planck18):
    """Calculate the absolute MUV magnitude of a galaxy.

    Parameters
    ----------
    galaxy : Galaxy
        The galaxy object containing stellar spectra.
    cosmo : Cosmology, optional
        The cosmology to use for redshift calculations, by default Planck18.

    Returns:
    -------
    dict
        Dictionary containing the MUV magnitude for each stellar spectrum in the galaxy.
    """
    phots = {}

    for key in list(galaxy.stars.spectra.keys()):
        lnu = galaxy.stars.spectra[key].get_photo_lnu(muv_filter).photo_lnu[0]
        phots[key] = lnu

    return phots


def calculate_sfr(galaxy, timescale=10 * Myr):
    """Calculate the star formation rate (SFR) of a galaxy over a specified timescale.

    Args:
        galaxy: An instance of a synthesizer.parametric.Galaxy object.
        timescale: The timescale over which to calculate the SFR (default is 10 Myr).

    Returns:
        The star formation rate as a float.
    """
    timescale = (0, timescale.to("yr").value)  # Convert timescale to years
    sfr = galaxy.stars.calculate_average_sfr(t_range=timescale)
    return sfr


def calculate_mass_weighted_age(galaxy):
    """Calculate the mass-weighted age of the stars in the galaxy."""
    return galaxy.stars.get_mass_weighted_age().to("Myr")


def calculate_lum_weighted_age(galaxy, spectra_type="total", filter_code="V"):
    """Calculate the luminosity-weighted age of the stars in the galaxy."""
    return galaxy.stars.get_lum_weighted_age(spectra_type=spectra_type, filter_code=filter_code).to(
        "Myr"
    )


def calculate_flux_weighted_age(galaxy, spectra_type="total", filter_code="JWST/NIRCam.F444W"):
    """Calculate the flux-weighted age of the stars in the galaxy."""
    return galaxy.stars.get_flux_weighted_age(
        spectra_type=spectra_type, filter_code=filter_code
    ).to("Myr")


def calculate_colour(
    galaxy: Galaxy,
    filter1: str,
    filter2: str,
    emission_model_key: str = "total",
    rest_frame: bool = False,
) -> float:
    """Measures the colour of a galaxy between two filters (filter1 - filter2).

    Args:
        galaxy: An instance of a synthesizer.parametric.Galaxy object.
        filter1: The first filter code (e.g., 'JWST/NIRCam.F444W').
        filter2: The second filter code (e.g., 'JWST/NIRCam.F115W').
        emission_model_key: The key for the emission model to use (default is 'total').
        rest_frame: Whether to use the rest frame (default is False).

    Returns:
        The colour of the galaxy as a float.
    """
    if (filter1 in ["U", "V", "J"] or filter2 in ["U", "V", "J"]) and not rest_frame:
        logger.warning(
            "Warning: Using 'U', 'V', or 'J' filters in the observed frame is not recommended. "
            "Set 'rest_frame=True' to use these filters in the rest frame."
        )
    for i, filter_code in enumerate([filter1, filter2]):
        if (
            galaxy.stars.spectra[emission_model_key].photo_fnu is None
            or filter_code not in galaxy.stars.spectra[emission_model_key].photo_fnu
        ):
            try:
                if filter_code in ["U", "V", "J"]:
                    filters = uvj[filter_code]

                else:
                    filters = FilterCollection(filter_codes=[filter_code])
                if rest_frame:
                    galaxy.stars.get_photo_lnu(filters)
                else:
                    galaxy.stars.get_photo_fnu(filters)
            except ValueError:
                raise ValueError(
                    "Filter '{filter_code}' is not available in the "
                    f"emission model '{emission_model_key}'."
                )
        if i == 0:
            flux1 = galaxy.stars.spectra[emission_model_key]
            if rest_frame:
                flux1 = flux1.photo_lnu[filter_code]
            else:
                flux1 = flux1.photo_fnu[filter_code]
        else:
            flux2 = galaxy.stars.spectra[emission_model_key]
            if rest_frame:
                flux2 = flux2.photo_lnu[filter_code]
            else:
                flux2 = flux2.photo_fnu[filter_code]

    colour = 2.5 * np.log10(flux2 / flux1)

    return colour


def calculate_d4000(galaxy: Galaxy, emission_model_key: str = "total") -> float:
    """Measures the D4000 index of a galaxy.

    Args:
        galaxy: An instance of a synthesizer.parametric.Galaxy object.
        emission_model_key: The key for the emission model to use (default is 'total').

    Returns:
        The D4000 index as a float.
    """
    d4000 = galaxy.stars.spectra[emission_model_key].measure_d4000()
    return d4000


def calculate_beta(galaxy: Galaxy, emission_model_key: str = "total") -> float:
    """Measures the beta index of a galaxy.

    Args:
        galaxy: An instance of a synthesizer.parametric.Galaxy object.
        emission_model_key: The key for the emission model to use (default is 'total').

    Returns:
        The beta index as a float.
    """
    beta = galaxy.stars.spectra[emission_model_key].measure_beta()
    return beta


def calculate_balmer_decrement(galaxy: Galaxy, emission_model_key: str = "total") -> float:
    """Measures the Balmer decrement oemission_modelf a galaxy.

    Args:
        galaxy: An instance of a synthesizer.parametric.Galaxy object.
        emission_model_key: The key for the emission model to use (default is 'total').

    Returns:
        The Balmer decrement as a float.
    """
    balmer_decrement = galaxy.stars.spectra[emission_model_key].measure_balmer_break(
        integration_method="simps"
    )
    return balmer_decrement


def calculate_line_flux(
    galaxy: Galaxy, emission_model, line="Ha", emission_model_key="total", cosmo=Planck18
):
    """Measures the equivalent widths of specific emission lines in a galaxy.

    Args:
        galaxy: An instance of a synthesizer.parametric.Galaxy object.
        emission_model: An instance of a synthesizer.emission_models.EmissionModel.
        line: The name of the emission line to measure (default is 'Ha').
        emission_model_key: The key for the emission model to use (default is 'total').
        cosmo: An instance of astropy.cosmology.Cosmology (default is Planck18).

    Returns:
        A dictionary with line names as keys and their equivalent widths as values.
    """
    from synthesizer.emissions.utils import aliases

    line = aliases.get(line, line)  # Handle aliases for line names

    line = galaxy.stars.get_lines(([line]), emission_model)
    flux = line.get_flux(cosmo=cosmo, z=galaxy.redshift)[0]

    return flux


def calculate_line_ew(galaxy: Galaxy, emission_model, line="Ha", emission_model_key="total"):
    """Measures the rest-frame equivalent widths of specific emission lines in a galaxy.

    Args:
        galaxy: An instance of a synthesizer.parametric.Galaxy object.
        emission_model: An instance of a synthesizer.emission_models.EmissionModel.
        line: The name of the emission line to measure (default is 'Ha').
        emission_model_key: The key for the emission model to use (default is 'total').

    Returns:
        A dictionary with line names as keys and their equivalent widths as values.
    """
    from synthesizer.emissions.utils import aliases

    line = aliases.get(line, line)  # Handle aliases for line names

    galaxy.stars.get_lines(([line]), emission_model)

    line = galaxy.stars.lines[emission_model_key]

    return line.equivalent_width[0]


def calculate_line_luminosity(
    galaxy: Galaxy, emission_model, line="Ha", emission_model_key="total"
):
    """Measures the luminosity of specific emission lines in a galaxy.

    Args:
        galaxy: An instance of a synthesizer.parametric.Galaxy object.
        emission_model: An instance of a synthesizer.emission_models.EmissionModel.
        line: The name of the emission line to measure (default is 'Ha').
        emission_model_key: The key for the emission model to use (default is 'total').

    Returns:
        A dictionary with line names as keys and their luminosities as values.
    """
    from synthesizer.emissions.utils import aliases

    line = aliases.get(line, line)  # Handle aliases for line names

    galaxy.stars.get_lines(([line]), emission_model)

    return galaxy.stars.lines[emission_model_key].luminosity[0]


def calculate_sfh_quantile(galaxy, quantile=0.5, norm=False, cosmo=Planck18):
    """Calculate the lookback time at which a certain fraction of the total mass is formed.

    Args:
        galaxy: An instance of a synthesizer.parametric.Galaxy object.
        quantile: The fraction of total mass formed (default is 0.5 for median).
        norm: If True then the age is as a fraction of the age of the universe at
            the redshift of the galaxy.
        cosmo: Cosmology object to use for age calculations, default is Planck18.

    Returns:
        The lookback time in Myr at which the specified fraction of total mass is formed.
    """
    if not isinstance(galaxy, Galaxy):
        raise TypeError("galaxy must be an instance of synthesizer.parametric.Galaxy")

    if not hasattr(galaxy.stars, "sfh"):
        raise AttributeError("galaxy.stars must have a 'sfh' attribute")

    assert 0 <= quantile <= 1, "quantile must be between 0 and 1."

    mass_bins = galaxy.stars.sf_hist
    ages = galaxy.stars.ages
    # from young to old

    cumulative_mass = np.cumsum(mass_bins[::-1])

    # Find the time at which the specified quantile of total mass is formed
    total_mass = cumulative_mass[-1]
    target_mass = quantile * total_mass

    # Find the index where cumulative mass exceeds target mass
    index = np.searchsorted(cumulative_mass, target_mass)

    lookback_time = ages[::-1][index].to("Myr")  # Get the corresponding age from the ages array

    if norm:
        # Normalize the lookback time to the age of the universe at the galaxy's redshift
        age_of_universe = cosmo.age(galaxy.redshift).to("Myr").value
        lookback_time = lookback_time.value / age_of_universe

    return lookback_time


def calculate_surviving_mass(galaxy, grid: Grid):
    """Calculate the surviving mass of the stellar population in a galaxy.

    Args:
        galaxy: An instance of a synthesizer.parametric.Galaxy object.
        grid: An instance of synthesizer.grid.Grid containing the SPS grid.

    Returns:
        The surviving mass as a unyt_quantity in Msun.
    """
    mass = galaxy.get_surviving_mass(grid)
    mass = np.log10(mass.to_value("Msun"))
    mass = unyt_array(mass, "log10_Msun")

    return mass


class SUPP_FUNCTIONS:
    """A class to hold supplementary functions for galaxy analysis."""

    calculate_muv = calculate_muv
    calculate_MUV = calculate_MUV
    calculate_sfr = calculate_sfr
    calculate_mass_weighted_age = calculate_mass_weighted_age
    calculate_lum_weighted_age = calculate_lum_weighted_age
    calculate_flux_weighted_age = calculate_flux_weighted_age
    calculate_colour = calculate_colour
    calculate_d4000 = calculate_d4000
    calculate_beta = calculate_beta
    calculate_balmer_decrement = calculate_balmer_decrement
    calculate_line_flux = calculate_line_flux
    calculate_line_ew = calculate_line_ew
    calculate_sfh_quantile = calculate_sfh_quantile
    calculate_surviving_mass = calculate_surviving_mass


# ------------------------------------------


def generate_random_DB_sfh(
    Nparam=3,
    tx_alpha=5,
    redshift=6,
    logmass=8,
    logsfr=-1,
    seed: Optional[int] = None,
):
    """Generate a random Dense Basis SFH.

    This function generates a random star formation history (SFH) using a dense basis
    representation. It creates a star formation history with a specified number of
    parameters, a Dirichlet distribution for the time series, and allows for customization
    of the log mass and log SFR.

    Parameters
    ----------
    size : int, optional
        Number of SFHs to generate, by default 1
    Nparam : int, optional
        Number of parameters for the SFH, by default 3
    tx_alpha : float, optional
        Concentration parameter for the Dirichlet distribution, by default 5
    redshift : float, optional
        Redshift of the SFH, by default 6
    logmass : float, optional
        Logarithm of the stellar mass in solar masses, by default 8
    logsfr : float, optional
        Logarithm of the star formation rate in solar masses per year, by default -1

    Returns:
    -------
    SFH.DenseBasis
        A DenseBasis SFH object with the generated parameters.
    np.ndarray
        An array of time series parameters for the SFH.
    """
    if seed is not None:
        np.random.seed(seed)
    txs = np.cumsum(
        np.random.dirichlet(np.ones((Nparam + 1,)) * tx_alpha, size=1),
        axis=1,
    )[0:, 0:-1][0]

    db_tuple = (logmass, logsfr, Nparam, *txs)
    sfh = SFH.DenseBasis(db_tuple, redshift)

    return sfh, txs


def generate_sfh_grid(
    sfh_type: Type[SFH.Common],
    sfh_priors: Dict[str, Dict[str, Any]],
    redshift: Union[Dict[str, Any], float],
    max_redshift: float = 15,
    cosmo: Type[Cosmology] = Planck18,
) -> Tuple[List[Type[SFH.Common]], np.ndarray]:
    """Generate a grid of SFHs based on prior distributions.

    This function creates a grid of SFH models by combining possible parameter
    values, which can depend explicitly on the redshift. It first draws
    redshifts, calculates maximum stellar ages at each
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

    Returns:
    -------
    Tuple[List[SFH], np.ndarray]
        - List of SFH objects with parameters drawn from the priors
        - Array of parameter combinations, where the first column is redshift followed
            by SFH parameters

    Notes:
    -----
    For parameters that depend on redshift (marked with 'depends_on': 'max_redshift'),
    the maximum allowed value will be dynamically adjusted based on
    the age of the universe at that redshift.
    """
    # Draw redshifts
    if isinstance(redshift, dict):
        redshifts = redshift["prior"].rvs(
            size=int(redshift["size"]),
            loc=redshift["min"],
            scale=redshift["max"] - redshift["min"],
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
        param_size = int(param_data["size"])
        param_min = param_data["min"]
        param_max = param_data["max"]

        # If parameter depends on redshift, adjust for each redshift
        if "depends_on" in param_data and param_data["depends_on"] == "max_redshift":
            # Create parameter values for each redshift
            all_values = []
            for z, max_age in zip(redshifts, max_ages):
                # Adjust maximum value based on the age of the universe at this redshift
                adjusted_max = min(param_max, max_age)
                if "units" in param_data and param_data["units"] is not None:
                    adjusted_max = adjusted_max * param_data["units"]
                    if hasattr(adjusted_max, "value"):
                        adjusted_max = adjusted_max.value

                # Draw values for this parameter at this redshift
                values = param_data["prior"].rvs(
                    size=param_size // len(redshifts),  # Distribute samples across redshifts
                    loc=param_min,
                    scale=adjusted_max - param_min,
                )
                all_values.append(values)

            # Combine values from all redshifts
            param_values = np.concatenate(all_values)
        else:
            # Parameter doesn't depend on redshift, draw from the same distribution
            param_values = param_data["prior"].rvs(
                size=param_size, loc=param_min, scale=param_max - param_min
            )

        param_arrays.append(param_values)
        param_names.append(param_data.get("name", key))  # Use specified name or key

    # Create parameter combinations using meshgrid
    mesh_arrays = np.meshgrid(*param_arrays, indexing="ij")
    param_combinations = np.stack([arr.flatten() for arr in mesh_arrays], axis=1)

    # Create SFH objects for each parameter combination
    sfhs = []
    for params in tqdm(param_combinations, desc="Generating SFHs", disable=(rank != 0)):
        z = params[0]  # First parameter is always redshift
        max_age = (cosmo.age(z) - cosmo.age(max_redshift)).to(u.Myr).value

        # Create parameter dictionary for SFH constructor
        sfh_params = {param_names[i + 1]: params[i + 1] for i in range(len(param_names) - 1)}

        # Apply units if not None
        for key, value in sfh_params.items():
            if "units" in sfh_priors[key] and sfh_priors[key]["units"] is not None:
                sfh_params[key] = value * sfh_priors[key]["units"]

        # Add max_age parameter
        sfh_params["max_age"] = max_age * Myr

        # Create and append SFH instance
        sfh = sfh_type(**sfh_params)
        sfhs.append(sfh)

    return sfhs, param_combinations


def generate_metallicity_distribution(
    zmet_dist: ZDist.Common,
    zmet: dict = {
        "prior": "loguniform",
        "min": -3,
        "max": 0.3,
        "size": 6,
        "units": None,
    },
):
    """Generate a grid of metallicity distributions based on prior distributions.

    Parameters
    ----------
    zmet_dist : Type[ZDist]
        The metallicity distribution class to instantiate.
        E.g., ZDist.DeltaConstant or ZDist.Normal
    z : dict
        Dictionary of prior distributions for zmet parameters. Each parameter should have:
        - 'prior': scipy.stats distribution
        - 'min': minimum value
        - 'max': maximum value
        - 'size': number of samples to draw
        - 'units': unyt unit (optional)
        - 'name': parameter name for constructor (optional, defaults to the key)
    """
    if isinstance(zmet, dict):
        zmet_values = zmet["prior"].rvs(
            size=int(zmet["size"]),
            loc=zmet["min"],
            scale=zmet["max"] - zmet["min"],
        )

    else:
        zmet_values = np.array([zmet])

    # Create parameter combinations using meshgrid
    zmet_array = np.meshgrid(zmet_values, indexing="ij")
    zmet_combinations = np.stack([arr.flatten() for arr in zmet_array], axis=1)

    # Create ZDist objects for each parameter combination
    zmet_dists = []
    for params in tqdm(zmet_combinations, desc="Generating ZDist", disable=(rank != 0)):
        # Create parameter dictionary for ZDist constructor
        zmet_params = {"zmet": params[0]}

        # Apply units if not None
        if "units" in zmet and zmet["units"] is not None:
            zmet_params["zmet"] = params[0] * zmet["units"]

        # Create and append ZDist instance
        zdist = zmet_dist(**zmet_params)
        zmet_dists.append(zdist)


def generate_emission_models(
    emission_model: Type[EmissionModel],
    varying_params: dict,
    grid: Grid,
    fixed_params: dict = None,
):
    """Generate a grid of emission models based on varying parameters.

    Parameters
    ----------

    emission_model : Type[EmissionModel]
        The emission model class to instantiate. E.g., TotalEmission or PacmanEmission

    varying_params : dict
        Dictionary of varying parameters for emission model. Each parameter should have:
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
        if "min" in args:
            args["loc"] = param_data["min"]
        if "max" in args:
            args["scale"] = param_data["max"] - param_data["min"]

        if "units" in args:
            args.pop("units")
        if "name" in args:
            args.pop("name")
        if "prior" in args:
            args.pop("prior")

        # Draw values for this parameter
        param_values = param_data["prior"].rvs(
            **args,
        )

        varying_param_arrays.append(param_values)

    # Create parameter combinations using meshgrid
    varying_param_mesh = np.meshgrid(*varying_param_arrays, indexing="ij")
    varying_param_combinations = np.stack([arr.flatten() for arr in varying_param_mesh], axis=1)
    # Create emission model objects for each parameter combination

    emission_models = []
    out_params = {}
    for i, params in tqdm(
        enumerate(varying_param_combinations),
        desc="Generating Emission Models",
        disable=(rank != 0),
    ):
        # Create parameter dictionary for emission model constructor
        emission_params = {key: params[j] for j, key in enumerate(varying_params.keys())}

        # Apply units if not None
        for key, value in emission_params.items():
            if "units" in varying_params[key] and varying_params[key]["units"] is not None:
                emission_params[key] = value * varying_params[key]["units"]
            if key not in out_params:
                out_params[key] = []
            out_params[key].append(emission_params[key])

        # store value of varying parameter(s) in dictionary for this emission model

        emission_params.update(fixed_params)

        # Create and append emission model instance
        emission_model_instance = emission_model(grid=grid, **emission_params)
        emission_models.append(emission_model_instance)

    return emission_models, out_params


def draw_from_hypercube(
    param_ranges,
    N: int = 1e6,
    model: Type[qmc.QMCEngine] = qmc.LatinHypercube,
    rng: Optional[np.random.Generator] = None,
    unlog_keys: Optional[List[str]] = None,
):
    """Draw N samples from a hypercube defined by the parameter ranges.

    Parameters
    ----------
    N : int
        Number of samples to draw.
    param_ranges : dict, optional
        Dictionary where keys are parameter names and values are tuples (min, max).
        Can be unyt_quantities for units.
    model : Type[qmc], optional
        The sampling model to use, by default LatinHypercube.
    rng : Optional[np.random.Generator], optional
        Random number generator to use for sampling, by default None.
    unlog_keys : Optional[List[str]], optional
        List of keys in param_ranges that should be unlogged
        (i.e., raised to power of 10). Units will be preserved
        even if this doesn't really make sense (e.g. Msun).

    Returns:
    -------
    dict
        Dictionary where keys are parameter names and values are arrays of sampled values.
    -------
    """
    if unlog_keys is None:
        unlog_keys = []

    # check if model takes 'rng' or 'seed' as an argument
    if "rng" in inspect.signature(model).parameters:
        key = "rng"
    elif "seed" in inspect.signature(model).parameters:
        key = "seed"
    else:
        raise ValueError("The model must accept either 'rng' or 'seed' as an argument.")

    rng_ = {key: rng} if rng is not None else {}

    # Create a Latin Hypercube sampler
    sampler = model(d=len(param_ranges), **rng_)

    # Generate samples in the unit hypercube
    sample = sampler.random(int(N))

    low = [param_ranges[key][0] for key in param_ranges.keys()]
    high = [param_ranges[key][1] for key in param_ranges.keys()]

    units = []

    for pos, (i, j) in enumerate(zip(low, high)):
        assert i < j, f"Parameter range {i} must be less than {j}"
        if isinstance(i, unyt_quantity):
            # If the parameter is a unyt_quantity, extract the value and unit
            units.append(i.units)
            i = i.value
            j = j.value
            low[pos] = i
            high[pos] = j
        else:
            # Otherwise, just use the value
            units.append(None)

    # Scale samples to the specified ranges
    scaled_samples = qmc.scale(
        sample,
        np.array(low),
        np.array(high),
    )

    all_param_dict = {}
    for i, key in enumerate(param_ranges.keys()):
        samples = scaled_samples[:, i].astype(np.float32)
        if key in unlog_keys:
            samples = 10**samples

            # If the key is in unlog_keys, raise to power of 10
            key = key.replace("log_", "")  # Remove 'log_' prefix if present
        if units[i] is not None:
            # If the parameter has units, convert samples to unyt_quantity
            samples = unyt_array(samples, units=units[i])

        if np.any(~np.isfinite(samples)):
            raise ValueError(
                f"Non-finite values found in samples for parameter '{key}'. "
                "Check the parameter ranges and ensure they are valid."
            )
        all_param_dict[key] = samples

    return all_param_dict


def load_hypercube_from_npy(file_path: str):
    """Load a hypercube from a .npy file.

    Parameters
    ----------
    file_path : str
        Path to the .npy file containing the hypercube data.

    Returns:
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
    redshifts: Union[Dict[str, Any], float, np.ndarray],
    sfh_param_units: List[Union[None, Unit]] = None,
    max_redshift: float = 20,
    calculate_min_age: bool = False,
    min_age_frac=0.001,
    cosmo: Type[Cosmology] = Planck18,
    iterate_redshifts: bool = False,
) -> Tuple[List[Type[SFH.Common]], np.ndarray]:
    """Generate a grid of SFHs based on parameter arrays and redshifts.

    Parameters
    ----------
    sfh_type : Type[SFH]
        The star formation history class to instantiate
    sfh_param_names : List[str]
        List of parameter names for SFH constructor
    sfh_param_arrays : List[np.ndarray]
        List of parameter arrays for SFH constructor.
        Should have the same length in the first dimension as sfh_param_names.
        if values are lambda functions the input will be the max age given max_redshift.
    redshifts : Union[Dict[str, Any], float]
        Either a single redshift value, an array of redshifts, or a dictionary with:
        'prior': scipy.stats distribution
        'min': minimum redshift
        'max': maximum redshift
        'size': number of redshift samples
    max_redshift : float, optional
        Maximum possible redshift to consider for age calculations, by default 15
    cosmo : Type[Cosmology], optional
        Cosmology to use for age calculations, by default Planck18
    calculate_min_age : bool, optional
        If True, calculate the lookback time at which only min_age_frac of total mass
        is formed, by default True
    min_age_frac : float, optional
        Fraction of total mass formed to calculate the minimum age, by default 0.001
    iterate_redshifts : bool, optional
        If True, iterate over redshifts and create SFH for each, by default True
        If False, assume input redshift SFH param array is a 1:1 mapping of
        redshift to SFH parameters.

    Returns:
    -------
    Tuple[List[SFH], np.ndarray]
        List of SFH objects with parameters drawn from the priors
        Array of parameter combinations, where the first column is redshift
        followed by SFH parameters
    """
    if isinstance(redshifts, dict):
        redshifts = redshifts["prior"].rvs(
            size=int(redshifts["size"]),
            loc=redshifts["min"],
            scale=redshifts["max"] - redshifts["min"],
        )
    elif isinstance(redshifts, (float, int)):
        redshifts = np.array([redshifts])
        if not iterate_redshifts:
            # extend redshifts to match the length of sfh_param_arrays
            redshifts = np.full(len(sfh_param_arrays[0]), redshifts)
    elif isinstance(redshifts, np.ndarray):
        pass
    else:
        raise ValueError("redshifts must be a dictionary, float/int, or numpy array")

    # Calculate maximum ages at each redshift

    max_ages = (cosmo.age(redshifts) - cosmo.age(max_redshift)).to(u.Myr).value

    sfhs = []

    if sfh_param_units is None:
        # If no units are provided, assume all parameters are dimensionless
        sfh_param_units = [None] * len(sfh_param_names)

    all_redshifts = []
    param_names_i = [i.replace("_norm", "") for i in sfh_param_names]

    if isinstance(sfh_param_arrays, tuple):
        sfh_param_arrays = list(sfh_param_arrays)

    if isinstance(sfh_param_arrays, (list, np.ndarray)):
        for pos, (set_unit, param) in enumerate(zip(sfh_param_units, sfh_param_arrays)):
            if isinstance(param, unyt_array):
                # If the parameter is a unyt_array, extract the unit
                sfh_param_units[pos] = param.units
                # Convert to numpy array if needed
                sfh_param_arrays[pos] = param.value
            elif isinstance(param, (list, tuple)):
                # If it's a list or tuple, convert to numpy array
                sfh_param_arrays[pos] = np.array(param)

            assert isinstance(sfh_param_units[pos], (Unit, type(None)))

    if len(sfh_param_names) == len(sfh_param_arrays):
        sfh_param_arrays = np.vstack(sfh_param_arrays).T

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
                    sfh_param_names[sf]: row_params[sf] for sf in range(len(sfh_param_names))
                }

                if "sfh_timescale" in sfh_param_names:
                    sfh_params["max_age"] = sfh_params["min_age"] + sfh_params["sfh_timescale"]
                    sfh_params.pop("sfh_timescale")

                # Add max_age parameter
                if "max_age" in sfh_param_names:
                    sfh_params["max_age"] = min(max_ages[i] * Myr, sfh_params["max_age"])
                else:
                    sfh_params["max_age"] = max_ages[i] * Myr
                # Create and append SFH instance

                sfh = sfh_type(**sfh_params)
                sfh.redshift = redshift
                sfhs.append(sfh)
                all_redshifts.append(redshift)
    else:
        assert len(redshifts) == len(
            sfh_param_arrays
        ), """If iterate_redshifts is False, len(redshifts)
            must equal len(sfh_param_arrays)"""
        for i, redshift in tqdm(enumerate(redshifts), desc="Creating SFHs"):
            params = copy.deepcopy(sfh_param_arrays[i])
            row_params = {}
            for j, param in enumerate(params):
                # Check if the parameter is a function
                if callable(param):
                    # Call the function with the maximum age
                    row_params[j] = param(max_ages[i])
                elif sfh_param_names[j].endswith("_norm"):
                    # If the parameter is normalized to max age, multiply by max_age
                    row_params[j] = param * max_ages[i] * Myr
                else:
                    # Otherwise, just use the parameter value
                    row_params[j] = param

                if sfh_param_units[j] is not None:
                    # Apply units if not None
                    row_params[j] = row_params[j] * sfh_param_units[j]

            # Create parameter dictionary for SFH constructor

            sfh_params = {param_names_i[sf]: row_params[sf] for sf in range(len(param_names_i))}

            # remove _norm from parameter names
            if "sfh_timescale" in sfh_param_names:
                sfh_params["max_age"] = sfh_params["min_age"] + sfh_params["sfh_timescale"]
                sfh_params.pop("sfh_timescale")

            # Add max_age parameter
            if "max_age" in sfh_param_names:
                sfh_params["max_age"] = min(max_ages[i] * Myr, sfh_params["max_age"])
            else:
                sfh_params["max_age"] = max_ages[i] * Myr
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
            age, sfr = sfh.calculate_sfh(t_range=(0, 1.1 * max_age), dt=1e6 * yr)
            sfr = sfr / sfr.max()
            mass_formed = np.cumsum(sfr[::-1])[::-1] / np.sum(sfr)
            total_mass = mass_formed[0]

            # Find the age at which min_age_frac of total mass is formed
            # interpolate
            min_age = np.interp(min_age_frac * total_mass, mass_formed, age)
            min_ages.append(min_age)

    return np.array(sfhs), redshifts


# from joblib import delayed, Parallel, wrap_non_picklable_objects, parallel_config


def create_galaxy(
    sfh: Type[SFH.Common],
    redshift: float,
    metal_dist: Type[ZDist.Common],
    grid: Grid,
    log_stellar_masses: Union[float, list] = 9,
    bh_kwargs=None,
    gas_kwargs=None,
    **galaxy_kwargs,
) -> Type[Galaxy]:
    """Create a new galaxy with the specified parameters."""
    # Initialise the parametric Stars object

    assert not isinstance(log_stellar_masses, (unyt_array, unyt_quantity)), (
        "log_stellar_masses must be a float or list of floats, not a unyt array"
    )

    single_mass = (
        log_stellar_masses[0] if isinstance(log_stellar_masses, (list)) else log_stellar_masses
    )

    single_mass = 10**single_mass * Msun

    param_stars = Stars(
        log10ages=grid.log10ages,
        metallicities=grid.metallicity,
        sf_hist=sfh,
        metal_dist=metal_dist,
        initial_mass=single_mass,
        **galaxy_kwargs,  # most parameters want to be on the emitter
    )

    # Define the number of stellar particles
    n = len(log_stellar_masses) if isinstance(log_stellar_masses, list) else 1
    if n > 1:
        # Sample the parametric SFZH to create "fake" stellar particles
        part_stars = sample_sfzh(
            sfzh=param_stars.sfzh,
            log10ages=np.log10(param_stars.ages),
            log10metallicities=np.log10(param_stars.metallicities),
            nstar=n,
            current_masses=10**log_stellar_masses * Msun,
            redshift=redshift,
            coordinates=np.random.normal(0, 0.01, (n, 3)) * Mpc,
            centre=np.zeros(3) * Mpc,
        )

        part_stars.__dict__.update(
            galaxy_kwargs
        )  # Add any additional parameters to the stars object
        from synthesizer.particle import Galaxy
    else:
        from synthesizer.parametric import Galaxy

        part_stars = param_stars

    if bh_kwargs is not None:
        from synthesizer.particle import BlackHoles

        bh = BlackHoles(**bh_kwargs)
    else:
        bh = None

    if gas_kwargs is not None:
        from synthesizer.particle import Gas

        gas = Gas(**gas_kwargs)
    else:
        gas = None

    # And create the galaxy
    galaxy = Galaxy(stars=part_stars, redshift=redshift, gas=gas, black_holes=bh)

    return galaxy


def _init_worker(grid, alt_parametrizations, fixed_params):
    """Initialize worker process with shared data."""
    _thread_local.grid = grid
    _thread_local.alt_parametrizations = alt_parametrizations
    _thread_local.fixed_params = fixed_params


def _process_galaxy_batch(galaxy_indices_and_data):
    """Process a batch of galaxies in a single worker."""
    # Access shared data from thread-local storage
    grid = _thread_local.grid
    alt_parametrizations = _thread_local.alt_parametrizations
    base_params = _thread_local.fixed_params

    galaxies = []

    for galaxy_idx, galaxy_data in galaxy_indices_and_data:
        # Reconstruct minimal parameters for this galaxy
        params = base_params.copy()  # Only copy once per batch item
        params.update(galaxy_data["varying_params"])

        # Create galaxy
        gal = create_galaxy(
            sfh=galaxy_data["sfh"],
            redshift=galaxy_data["redshift"],
            metal_dist=galaxy_data["metal_dist"],
            log_stellar_masses=galaxy_data["log_stellar_mass"],
            grid=grid,  # Use shared reference
            **params,
        )

        # Process parameters (reuse logic from original)
        save_params = copy.deepcopy(params)
        save_params["redshift"] = galaxy_data["redshift"]
        save_params.update(galaxy_data["sfh"].parameters)
        save_params.update(galaxy_data["metal_dist"].parameters)

        # Apply alternative parametrizations
        if len(alt_parametrizations) > 0:
            to_remove = set()
            for key, (new_key, func) in alt_parametrizations.items():
                if key in save_params:
                    if isinstance(new_key, str):
                        save_params[new_key] = func(save_params)
                        to_remove.add(key)
                    elif isinstance(new_key, (list, tuple)):
                        for k in new_key:
                            save_params[k] = func(k, save_params)
                        to_remove.add(key)

            for key in to_remove:
                save_params.pop(key, None)

        gal.all_params = save_params
        galaxies.append((galaxy_idx, gal))

    return galaxies


class GalaxyBasis:
    """Class to create a basis of galaxies with different SFHs, redshifts, and parameters.

    It can support two modes of operation.

    The first case in the simplest - you have some set of priors controlling e.g. mass,
    redshift, SFH parameters, and metallicity distribution parameters, which you draw
    randomly or from a Latin hypercube, and then you create a galaxy for each set of
    parameters. So if you are drawing 1000 galaxies, you would pass in 1000 redshifts,
    1000 SFHs, 1000 metallicity distributions, and 1000 sets of galaxy parameters.

    If however you are sampling on a grid, or want dependent priors, you can instead
    pass in a few basis SFHs, redshifts, and metallicity distributions, and then
    the class will generate a grid of galaxies by combining every combination of
    SFH, redshift, and metallicity distributions. This can reduce the number of
    galaxies created.
    """

    def __init__(
        self,
        model_name: str,
        redshifts: np.ndarray,
        grid: Grid,
        emission_model: Type[EmissionModel],
        sfhs: List[Type[SFH.Common]],
        metal_dists: List[Type[ZDist.Common]],
        log_stellar_masses: Optional[np.ndarray] = None,
        galaxy_params: dict = None,
        alt_parametrizations: Dict[str, Tuple[str, callable]] = None,
        cosmo: Type[Cosmology] = Planck18,
        instrument: Instrument = None,
        redshift_dependent_sfh: bool = False,
        params_to_ignore: List[str] = None,
        build_grid: bool = False,
    ) -> None:
        """Initialize the GalaxyBasis object with SFHs, redshifts, and other parameters.

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
        alt_parametrizations : dict
            Dictionary of alternative parametrizations for the galaxy parameters -
            for parametrizing differently to Synthesizer if wanted. Should be a dictionary
            with keys as the parameter names and values as tuples of the new parameter
            name and a function which takes the parameter dictionary and returns the new
            parameter value (so it can be calculated from the other parameters if needed).
        metal_dists : List[Type[ZDist.Common]], optional
            List of metallicity distribution objects, by default None
        log_stellar_masses : Optional[np.ndarray], optional
            Array of logarithmic stellar masses in solar masses, by default None.
        cosmo : Type[Cosmology], optional
            Cosmology object, by default Planck18
        instrument : Instrument, optional
            Instrument object containing the filters, by default None
        redshift_dependent_sfh : bool, optional
            If True, the SFH will depend on redshift, by default False. If True, expect
            each SFH to have a redshift attribute.
        params_to_ignore : List[str], optional
            List of parameters to ignore as being different when calculating
            which parameters are varying.
            E.g. max_age may be dependent on redshift, so we don't want to include it
                in the varying parameters as the model can learn this.
        build_grid : bool, optional
            If True, build the grid of galaxies, by default True.
            If False, assume all dimensions of parameters are the same size and
            build the grid from the parameters. I.e don't generate combinations of
            parameters, just use the parameters as they are.
        """
        if galaxy_params is None:
            galaxy_params = {}
        if alt_parametrizations is None:
            alt_parametrizations = {}
        if params_to_ignore is None:
            params_to_ignore = []

        if isinstance(redshifts, (float, int)) and not build_grid:
            redshifts = np.full(len(sfhs), redshifts)

        self.model_name = model_name
        self.sfhs = sfhs
        self.redshifts = redshifts
        self.grid = grid
        self.emission_model = emission_model
        self.galaxy_params = galaxy_params
        self.alt_parametrizations = alt_parametrizations
        self.metal_dists = metal_dists
        self.cosmo = cosmo
        self.instrument = instrument
        self.redshift_dependent_sfh = redshift_dependent_sfh
        self.log_stellar_masses = log_stellar_masses
        self.params_to_ignore = params_to_ignore
        self.build_grid = build_grid

        self.galaxies = []

        if isinstance(self.metal_dists, ZDist.Common):
            self.metal_dists = [self.metal_dists]

        if isinstance(self.sfhs, SFH.Common):
            self.sfhs = [self.sfhs]

        # if self.stellar_masses is not None:
        #    assert isinstance(self.stellar_masses, (unyt_array, unyt_quantity)), (
        #        "stellar_masses must be a unyt array or quantity"
        #   )

        self.per_particle = False

        # Check if any galaxy parameters are dictionaries with keys like 'prior', 'min'
        for key, value in galaxy_params.items():
            if isinstance(value, dict):
                # If the value is a dictionary, process it as a prior
                self.galaxy_params[key] = self.process_priors(value)

        if not build_grid:
            logger.info("Generating grid directly from provided parameter samples.")
        elif self.redshift_dependent_sfh:
            # Check if the SFHs have a redshift attribute
            for sfh in self.sfhs:
                if not hasattr(sfh, "redshift"):
                    raise ValueError(
                        "SFH must have a redshift attr if redshift_dependent_sfh==True"
                    )

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

    def process_priors(
        self,
        prior_dict: Dict[str, Any],
    ) -> unyt_array:
        """Process priors from dictionary.

        Parameters
        ----------
        prior_dict : Dict[str, Any]
            Dictionary containing prior information. Must contain:
            - 'prior': scipy.stats distribution
            - 'size': number of samples to draw
            - 'units': unyt unit (optional)
            - other parameters required by the distribution

        """
        assert isinstance(prior_dict, dict), "prior_dict must be a dictionary"

        assert "prior" in prior_dict, "prior_dict must contain a 'prior' key"
        assert "size" in prior_dict, "prior_dict must contain a 'size' key"

        stats_params = list_parameters(prior_dict["prior"])

        # Check required parameters are present
        params = {}
        for param in stats_params:
            assert param in prior_dict, f"prior_dict must contain a '{param}' key"
            params[param] = prior_dict[param]

        # draw values for this parameter

        values = prior_dict["prior"].rvs(size=int(prior_dict["size"]), **params)

        if "units" in prior_dict and prior_dict["units"] is not None:
            values = unyt_array(values, units=prior_dict["units"])

        return values

    def _create_galaxies(
        self,
        log_base_masses: 9,
    ) -> List[Type[Galaxy]]:
        """Create galaxies with the specified SFHs, redshifts, and other parameters.

        Parameters
        ----------
        log_base_masses  : Union[float, np.ndarray], optional
            Base mass (or array of base masses) to use for the galaxies.
            Units of log10 M sun.
            Default mass (or mass array) to use for the galaxies.

        Returns:
        -------
        List[Type[Galaxy]]
            List of Galaxy objects.
        """
        if not self.build_grid:
            raise ValueError("You probably meant to call_create_matched_galaxies instead.")

        varying_param_values = [
            i for i in self.galaxy_params.values() if type(i) in [list, np.ndarray]
        ]

        if isinstance(log_base_masses, (list, np.ndarray)):
            self.per_particle = True

        # generate all combinations of the varying parameters
        if len(varying_param_values) == 0:
            param_list = [{}]
            fixed_params = self.galaxy_params

        else:
            varying_param_combinations = np.array(np.meshgrid(*varying_param_values)).T.reshape(
                -1, len(varying_param_values)
            )
            column_names = [
                i
                for i, j in zip(self.galaxy_params.keys(), varying_param_values)
                if type(j) in [list, np.ndarray]
            ]
            fixed_params = {
                key: value
                for key, value in self.galaxy_params.items()
                if type(value) not in [list, np.ndarray]
            }
            param_list = [
                {column_names[i]: j for i, j in enumerate(row)}
                for row in varying_param_combinations
            ]

        galaxies = []
        all_parameters = {}
        for i, redshift in tqdm(
            enumerate(self.redshifts),
            desc=f"Creating {self.model_name} galaxies",
            total=len(self.redshifts),
            disable=(rank != 0),
        ):
            # get the sfh for this redshift
            sfh_models = self.sfhs[redshift]
            for sfh_model in sfh_models:
                sfh_parameters = sfh_model.parameters
                for k, Z_dist in enumerate(self.metal_dists):
                    Z_parameters = Z_dist.parameters

                    # Create a new galaxy with the specified parameters
                    for params in param_list:
                        params.update(fixed_params)
                        gal = create_galaxy(
                            sfh=sfh_model,
                            redshift=redshift,
                            metal_dist=Z_dist,
                            log_stellar_masses=log_base_masses,
                            grid=self.grid,
                            **params,
                        )
                        save_params = copy.deepcopy(params)
                        save_params["redshift"] = redshift
                        save_params.update(sfh_parameters)
                        save_params.update(Z_parameters)

                        if len(self.alt_parametrizations) > 0:
                            to_remove = []
                            # Apply alternative parametrizations if provided
                            for key, (
                                new_key,
                                func,
                            ) in self.alt_parametrizations.items():
                                if isinstance(new_key, str):
                                    save_params[new_key] = func(save_params)
                                    to_remove.append(key)
                                elif isinstance(new_key, (list, tuple)):
                                    for k in new_key:
                                        save_params[k] = func(k, save_params)
                                to_remove.append(key)
                            for key in to_remove:
                                save_params.pop(key)

                        # This stores all input parameters for the galaxy
                        # so we can work out which parameters
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
        fixed_param_values = []
        fixed_param_units = []
        varying_param_names = []

        for key, value in all_parameters.items():
            if len(value) == 1 and value[0] is None:
                to_remove.append(key)
                continue
            # check if all values are the same.
            if len(np.unique(value)) == 1:
                fixed_param_names.append(key)
                fixed_param_units.append(
                    str(value[0].units) if isinstance(value[0], unyt_quantity) else ""
                )
                fixed_param_values.append(value[0])
            else:
                varying_param_names.append(key)

        for param in self.params_to_ignore:
            if param in varying_param_names:
                varying_param_names.remove(param)

        # Sanity check all varying parameters combinations on self.galaxies are unique
        logger.info("Checking parameters are unique.")
        hashes = []
        for gal in self.galaxies:
            relevant_params = {
                key: gal.all_params[key] for key in varying_param_names if key in gal.all_params
            }
            # Calculate hash for each parameter and sum them
            hash_i = sum(
                hash(float(param.value))
                if isinstance(param, (unyt_array, unyt_quantity))
                else hash(param)
                for param in relevant_params.values()
            )
            hashes.append(hash_i)

        if len(hashes) != len(set(hashes)):
            raise ValueError(
                """Varying parameters are not unique across galaxies.
                Check your input parameters."""
            )

        self.varying_param_names = varying_param_names
        self.fixed_param_names = fixed_param_names
        self.fixed_param_values = fixed_param_values
        self.fixed_param_units = fixed_param_units
        self.all_parameters = all_parameters

        for key in to_remove:
            all_parameters.pop(key)

        logger.info("Finished creating galaxies.")

        return self.galaxies

    def _check_model_simplicity(self, parameter_transforms_to_save=None, verbose=True) -> bool:
        """Check if the model is simple enough to be stored in a file.

        Checks include:
            All SFHs are the same underlying SFH class.
            All metallicity distributions are the same underlying ZDist class.
            Single emission model class.
            We understand emitter parameters and can store them.
            We can store the grid path and filter names.
            We can serialize alt_parametrizations.

        If the model is simple enough, we can store in HDF5 and
        allow creation of a GalaxySimulator from it.
        """
        accept = True

        sfh_classes = set(type(sfh) for sfh in self.sfhs)
        if len(sfh_classes) > 1:
            if verbose:
                logger.warning(
                    f"SFH classes are not all the same: {sfh_classes}. Cannot store model."
                )
            accept = False

        if type(self.sfhs[0]).__name__ not in SFH.parametrisations:
            if verbose:
                logger.warning(SFH.parametrisations)
                logger.warning(
                    f"SFH class {type(self.sfhs[0]).__name__} is not in SFH.parametrisations. "  # noqa: E501
                    "Cannot store model."
                )
            accept = False

        metal_dist_classes = set(type(metal_dist) for metal_dist in self.metal_dists)
        if len(metal_dist_classes) > 1:
            if verbose:
                logger.warning(
                    f"Metallicity distribution classes are not all the same: {metal_dist_classes}. "  # noqa: E501
                    "Cannot store model."
                )
            accept = False

        if type(self.metal_dists[0]).__name__ not in ZDist.parametrisations:
            if verbose:
                logger.warning(
                    f"Metallicity distribution class {type(self.metal_dists[0]).__name__} "  # noqa: E501
                    "is not in ZDist.parametrisations. Cannot store model."
                )
            accept = False

        if not isinstance(self.emission_model, EmissionModel):
            if verbose:
                logger.warning(
                    f"Emission model is not an instance of EmissionModel: {self.emission_model}. "  # noqa: E501
                    "Cannot store model."
                )
            accept = False

        # check emission model in synthesizer.emission_models.PREMADE_MODELS

        from synthesizer.emission_models import PREMADE_MODELS

        if type(self.emission_model).__name__ not in PREMADE_MODELS:
            if verbose:
                logger.warning(
                    f"Emission model {type(self.emission_model).__name__} is not in PREMADE_MODELS. "  # noqa: E501
                    "Cannot store model."
                )
            accept = False

        em_args = inspect.signature(type(self.emission_model)).parameters
        forbidden_args = [
            "nlr_grid",
            "blr_grid",
            "covering_fraction",
            "covering_fraction_nlr",
            "covering_fraction_blr",
            "torus_emission_model",
            "dust_curve_ism",
            "dust_curve_birth",
            "dust_emission_ism",
            "dust_emission_birth",
        ]

        for arg in forbidden_args:
            if arg in em_args:
                if verbose:
                    logger.warning(
                        f"Emission model {type(self.emission_model).__name__} has forbidden argument '{arg}'. "  # noqa: E501
                        "Cannot store model."
                    )
                accept = False
        # can we we convert functions in alt_parametrizations to strings?
        # and load with ast.literal_eval?

        if parameter_transforms_to_save is not None:
            for key, value in parameter_transforms_to_save.items():
                # value should be (str, callable) or (list/tuple, callable)
                if isinstance(value, tuple) or callable(value):
                    if callable(value):
                        value = (key, value)

                    if len(value) != 2 or not callable(value[1]):
                        accept = False
                    else:
                        try:
                            # Check if we can get the source code of the function
                            getsource(value[1])
                        except Exception:
                            if verbose:
                                logger.warning(
                                    f"Cannot serialize function for alt_parametrization '{key}': {value[1]}"  # noqa: E501
                                )
                            accept = False
                else:
                    accept = False

        return accept

    def _store_model(
        self,
        model_path: str,
        overwrite=False,
        group: str = "Model",
        other_info: dict = None,
        parameter_transforms_to_save=None,
    ) -> bool:
        if not self._check_model_simplicity(parameter_transforms_to_save):
            logger.warning("Model is too complex to be stored in a file.")
            return False

        # if not overwrite, append to existing file
        if os.path.exists(model_path) and not overwrite:
            open_mode = "a"
        else:
            open_mode = "w"
        with h5py.File(model_path, open_mode) as f:
            if group in f:
                if overwrite:
                    del f[group]
                else:
                    logger.warning(f"Group {group} already exists in {model_path}.")
                    return False

            base = f.create_group(group)

            # store grid_name and grid_dir
            base.attrs["grid_name"] = self.grid.grid_name
            base.attrs["grid_dir"] = self.grid.grid_dir

            # store emission model class name
            em_group = base.create_group("EmissionModel")

            em_group.attrs["name"] = type(self.emission_model).__name__

            # store emission model parameters
            em_model_params = save_emission_model(self.emission_model)

            em_group.attrs["parameter_keys"] = em_model_params["fixed_parameter_keys"]
            em_group.attrs["parameter_values"] = em_model_params["fixed_parameter_values"]
            # if it can be an int or float, store as such
            em_group.attrs["parameter_units"] = em_model_params["fixed_parameter_units"]

            if em_model_params["dust_law"] is not None:
                em_group.attrs["dust_law"] = em_model_params["dust_law"]
                em_group.attrs["dust_attenuation_keys"] = em_model_params["dust_attenuation_keys"]
                em_group.attrs["dust_attenuation_values"] = em_model_params[
                    "dust_attenuation_values"
                ]
                em_group.attrs["dust_attenuation_units"] = em_model_params["dust_attenuation_units"]

            if em_model_params["dust_emission"] is not None:
                em_group.attrs["dust_emission"] = em_model_params["dust_emission"]
                em_group.attrs["dust_emission_keys"] = em_model_params["dust_emission_keys"]
                em_group.attrs["dust_emission_values"] = em_model_params["dust_emission_values"]
                em_group.attrs["dust_emission_units"] = em_model_params["dust_emission_units"]

            # Store a version of astropy cosmo
            cosmo_yaml = self.cosmo.to_format("yaml")
            base.attrs["cosmology"] = cosmo_yaml

            # store instrument.filter_codes

            base.attrs["instrument"] = self.instrument.label if self.instrument else None
            if self.instrument:
                base.attrs["filters"] = self.instrument.filters.filter_codes

            instrument_group = base.create_group("Instrument")

            self.instrument.to_hdf5(instrument_group)

            # store sfh class
            base.attrs["sfh_class"] = type(self.sfhs[0]).__name__
            # store metallicity distribution class
            base.attrs["metallicity_distribution_class"] = type(self.metal_dists[0]).__name__

            base.attrs["model_name"] = self.model_name

            if other_info is None:
                other_info = {}

            for key, value in other_info.items():
                if isinstance(value, (list, np.ndarray)):
                    # Store as a dataset
                    base.create_dataset(key, data=np.array(value))
                    # if a unyt quantity, store the units as an attribute
                    if isinstance(value, unyt_array):
                        base[key].attrs["units"] = str(value.units)
                else:
                    # Store as an attribute
                    base.attrs[key] = value

            base.attrs["stellar_params"] = list(self.galaxy_params.keys())

            if parameter_transforms_to_save is not None:
                transforms_group = base.create_group("Transforms")
                for key, value in parameter_transforms_to_save.items():
                    if isinstance(value, tuple) or callable(value):
                        if callable(value):
                            value = (key, value)
                        # Store the new parameter name and the function as a string
                        transforms_group.create_dataset(
                            key, data=getsource(value[1]).encode("utf-8")
                        )
                        transforms_group[key].attrs["new_parameter_name"] = value[0]

            # Store param_order
            base.attrs["varying_param_names"] = self.varying_param_names
            base.attrs["fixed_param_names"] = self.fixed_param_names
            base.attrs["fixed_param_values"] = self.fixed_param_values
            base.attrs["fixed_param_units"] = self.fixed_param_units

    def create_galaxy(
        self,
        sfh: Type[SFH.Common],
        redshift: float,
        metal_dist: Type[ZDist.Common],
        log_stellar_masses: Union[float, list] = 9,
        grid: Optional[Grid] = None,
        **galaxy_kwargs,
    ) -> Type[Galaxy]:
        """Create a galaxy from parameters.

        Parameters:
        -----------
        sfh: SFH model class, e.g. SFH.LogNormal
        redshift: redshift of the galaxy
        metal_dist: metallicity distriution of the galaxy
        log_stellar_masses: float or list of floats, log10 stellar mass in solar masses
        grid: Grid object, if None use self.grid
        galaxy_kwargs: additional keyword arguments to pass to the Galaxy class

        Returns:
        -------
        Type[Galaxy]
            Galaxy object created with the specified parameters.
        """
        return create_galaxy(
            sfh=sfh,
            redshift=redshift,
            metal_dist=metal_dist,
            log_stellar_masses=log_stellar_masses,
            grid=grid if grid is not None else self.grid,
            **galaxy_kwargs,
        )

    def create_galaxies_optimized(
        self,
        fixed_params,
        varying_param_names,
        log_base_masses,
        galaxies_mask=None,
        n_proc=28,
        batch_size=None,
    ):
        """Optimized version with reduced serialization overhead."""
        # Determine optimal batch size if not provided
        if batch_size is None:
            total_galaxies = len(self.sfhs)
            if galaxies_mask is not None and len(galaxies_mask) > 0:
                total_galaxies = np.sum(galaxies_mask)
            batch_size = max(1, total_galaxies // (n_proc))  # 1 batch per thread
            logger.info(total_galaxies)

        # Prepare lightweight job inputs (grouped into batches)
        job_batches = []
        current_batch = []

        for i in tqdm(range(len(self.sfhs)), desc="Preparing galaxy batches", disable=(rank != 0)):
            # Skip this galaxy if the mask is provided and is False
            if galaxies_mask is not None and len(galaxies_mask) > 0 and not galaxies_mask[i]:
                continue

            # Create minimal data structure (no large objects)
            varying_params = {}
            for key in varying_param_names:
                varying_params[key] = self.galaxy_params[key][i]

            # Determine the stellar mass for this galaxy
            try:
                mass = log_base_masses[i]
            except (IndexError, TypeError):
                mass = log_base_masses

            galaxy_data = {
                "sfh": self.sfhs[i],
                "redshift": self.redshifts[i],
                "metal_dist": self.metal_dists[i],
                "log_stellar_mass": mass,
                "varying_params": varying_params,  # Only varying parameters
            }

            current_batch.append((i, galaxy_data))

            # Create batch when it reaches desired size
            if len(current_batch) >= batch_size:
                job_batches.append(current_batch)
                current_batch = []

        # Add remaining galaxies to final batch
        if current_batch:
            job_batches.append(current_batch)

        if n_proc > 1:
            logger.info(f"Creating {len(job_batches)} batches across {n_proc} processes.")
            logger.info(
                f"Average batch size: {sum(len(batch) for batch in job_batches) / len(job_batches):.1f}"  # noqa: E501
            )

            # Create a wrapper function that includes the shared data
            def process_batch_with_shared_data(batch):
                # Initialize thread-local data for this worker
                _init_worker(self.grid, self.alt_parametrizations, fixed_params)
                return _process_galaxy_batch(batch)

            # Use threading backend
            with parallel_config("threading"):
                tasks = [delayed(process_batch_with_shared_data)(batch) for batch in job_batches]

                # Process batches
                batch_results = Parallel(n_jobs=n_proc)(
                    tqdm(tasks, desc="Creating galaxy batches", disable=(rank != 0))
                )

            # Flatten results and sort by original index
            all_results = []
            for batch_result in batch_results:
                all_results.extend(batch_result)

            # Sort by original galaxy index to maintain order
            all_results.sort(key=lambda x: x[0])
            self.galaxies = [galaxy for _, galaxy in all_results]

        else:
            # Sequential processing - initialize once for the main thread
            _init_worker(self.grid, self.alt_parametrizations, fixed_params)
            self.galaxies = []
            for batch in tqdm(job_batches, desc="Creating galaxies", disable=(rank != 0)):
                batch_galaxies = _process_galaxy_batch(batch)
                self.galaxies.extend([galaxy for _, galaxy in batch_galaxies])

    def _create_matched_galaxies(
        self,
        log_base_masses: Union[float, np.ndarray] = 9,
        galaxies_mask: Optional[np.ndarray] = None,
        n_proc: int = 1,
    ) -> List[Type[Galaxy]]:
        """Creates galaxies where all parameters have been sampled.

        A galaxy is created for each SFH, redshift, and metallicity distribution
        supplied.

        Parameters
        ----------
        log_base_masses : Union[float, np.ndarray], optional
            Base mass (or array of base masses) to use for the galaxies.
        n_procs : int, optional
            Number of processes to use for parallel processing, by default 1.

        Returns:
        -------
        List[Type[Galaxy]]
            List of Galaxy objects created with the specified parameters.
        """
        if len(self.metal_dists) == 1:
            # Just reference the first one
            self.metal_dists = [self.metal_dists[0]] * len(self.sfhs)

        assert len(self.sfhs) == len(
            self.redshifts
        ), f"""If iterate_redshifts is False, sfhs and redshifts must be the same length,
            got {len(self.sfhs)} and {len(self.redshifts)}"""
        assert len(self.sfhs) == len(
            self.metal_dists
        ), f"""If iterate_redshifts is False, sfhs and metal_dists must be the same
            length, got {len(self.sfhs)} and {len(self.metal_dists)}"""

        if galaxies_mask is not None:
            assert len(galaxies_mask) == len(self.sfhs), (
                "galaxies_mask must be the same length as sfhs, redshifts, and metal_dists"
            )

        logger.info("Checking parameters inside create_matched_galaxies.")
        varying_param_values = [
            i for i in self.galaxy_params.values() if type(i) in [list, np.ndarray]
        ]

        # generate all combinations of the varying parameters
        if len(varying_param_values) == 0:
            fixed_params = self.galaxy_params
            varying_param_names = []

        else:
            fixed_params = {
                key: value
                for key, value in self.galaxy_params.items()
                if type(value) not in [list, np.ndarray]
            }
            varying_param_names = [
                i for i in self.galaxy_params.keys() if i not in fixed_params.keys()
            ]
            assert all(
                len(self.galaxy_params[i]) == len(self.sfhs) for i in varying_param_names
            ), f"""All varying parameters must be the same length,
                got {len(self.sfhs)} and {len(self.galaxy_params)}"""

        # This was a side-effect in the original loop. We can detect it here
        # before running the jobs.
        if isinstance(log_base_masses, (list, np.ndarray)) and isinstance(
            log_base_masses[0], (list, np.ndarray)
        ):
            self.per_particle = True

        """
        job_inputs: List[Dict[str, Any]] = []
        for i in tqdm(range(len(self.sfhs)), desc="Batching galaxy inputs", disable=(rank != 0)):
            # Skip this galaxy if the mask is provided and is False
            if galaxies_mask is not None and len(galaxies_mask) > 0 and not galaxies_mask[i]:
                continue

            # Assemble the parameters for this specific galaxy
            params = fixed_params.copy()
            for key in varying_param_names:
                params[key] = self.galaxy_params[key][i]

            # Determine the stellar mass for this galaxy
            try:
                mass = log_base_masses[i]
            except (IndexError, TypeError):
                mass = log_base_masses

            job_inputs.append(
                {
                    "sfh": self.sfhs[i],
                    "redshift": self.redshifts[i],
                    "metal_dist": self.metal_dists[i],
                    "log_stellar_mass": mass,
                    "params": params,
                    "grid": self.grid,
                    "alt_parametrizations": self.alt_parametrizations,
                }
            )
        """

        # Use the optimized version
        self.create_galaxies_optimized(
            galaxies_mask=galaxies_mask,
            varying_param_names=varying_param_names,
            log_base_masses=log_base_masses,
            fixed_params=fixed_params,
            n_proc=1,
        )

        logger.info(f"Created {len(self.galaxies)} galaxies.")

        # Use sets instead of lists for faster lookups and unique values
        self.all_parameters = defaultdict(set)
        self.all_params = {}

        param_units = {}
        # Process all galaxies in one pass
        for i, gal in enumerate(self.galaxies):
            # Store the galaxy parameters directly
            self.all_params[i] = gal.all_params

            # Add unique values to sets using update operation
            for key, value in gal.all_params.items():
                if key not in self.params_to_ignore:
                    if isinstance(value, (unyt_quantity, unyt_array)):
                        unit = str(value.units)
                        param_units[key] = unit
                        value = value.value
                    if isinstance(value, np.ndarray):
                        value = value.tolist()

                    if isinstance(value, list):
                        # If value is a list, add each element to the set
                        # for u, v in enumerate(value):
                        #    self.all_parameters[f'{key}_{u}'].add(v)
                        pass
                    else:
                        self.all_parameters[key].add(value)

        # Convert sets to lists at the end if needed
        self.all_parameters = {k: list(v) for k, v in self.all_parameters.items()}

        # Remove any paremters which are just [None]
        to_remove = []
        fixed_param_names = []
        fixed_param_values = []
        varying_param_names = []

        for key, value in tqdm(
            self.all_parameters.items(), desc="Processing parameters", disable=(rank != 0)
        ):
            if len(value) == 1 and value[0] is None:
                to_remove.append(key)
                continue
            # check if all values are the same.
            if len(np.unique(value)) == 1:
                fixed_param_names.append(key)
                fixed_param_values.append(value[0])
            else:
                varying_param_names.append(key)

        for param in self.params_to_ignore:
            if param in varying_param_names:
                varying_param_names.remove(param)

        self.varying_param_names = varying_param_names
        self.fixed_param_names = fixed_param_names
        self.fixed_param_values = fixed_param_values
        self.fixed_param_units = []
        for key in fixed_param_names:
            if key in param_units:
                self.fixed_param_units.append(param_units[key])
            else:
                self.fixed_param_units.append("")

        for key in to_remove:
            self.all_parameters.pop(key)

        logger.info("Finished creating galaxies.")

        return self.galaxies

    def process_galaxies(
        self,
        galaxies: List[Type[Galaxy]],
        out_name: str = "auto",
        out_dir: str = "internal",
        n_proc: int = 4,
        verbose: int = 1,
        save: bool = True,
        emission_model_keys=None,
        batch_galaxies: bool = True,
        batch_size: int = 40_000,
        overwrite: bool = False,
        multi_node: bool = False,
        **extra_analysis_functions,
    ) -> Pipeline:
        """Processes galaxies through Synthesizer pipeline.

        Parameters
        ----------
        galaxies : List[Type[Galaxy]]
            List of Galaxy objects to process.
        out_name : str, optional
            Name of the output file to save the pipeline results, by default "auto".
            If "auto", uses the model_name.
        out_dir : str, optional
            Directory to save the output file, by default "internal".
            If "internal", saves to the Synthesizer grids directory.
        n_proc : int, optional
            Number of processes to use for parallel processing, by default 4.
        verbose : int, optional
            Verbosity level for the pipeline, by default 1.
        save : bool, optional
            If True, saves the pipeline results to disk, by default True.
        emission_model_keys : List[str], optional
            List of emission model keys to save spectra for, by default None.
            If None, saves all spectra.
        batch_galaxies : bool, optional
            If True, processes galaxies in batches, by default True.
            If False, processes all galaxies in a single batch.
        batch_size : int, optional
            Size of each batch of galaxies to process, by default 40,000.
        extra_analysis_functions : dict, optional
            Additional analysis functions to add to the pipeline.
            Should be a dictionary where keys are function names and values are tuples
            of the function and its parameters.

        Returns:
        -------
        Pipeline
            The pipeline object after processing the galaxies.

        """
        self.emission_model.set_per_particle(self.per_particle)

        if emission_model_keys is not None:
            self.emission_model.save_spectra(emission_model_keys)

        logger.info("Creating pipeline.")

        if not batch_galaxies:
            batch_size = len(galaxies)

        n_batches = int(np.ceil(len(galaxies) / batch_size))

        if n_batches > 1:
            logger.info(f"Splitting galaxies into {n_batches} batches of size {batch_size}.")
            galaxies = [galaxies[i * batch_size : (i + 1) * batch_size] for i in range(n_batches)]
        else:
            logger.info("Processing all galaxies in a single batch.")
            galaxies = [galaxies]

        for batch_i, batch_gals in enumerate(galaxies):
            skip = False
            if save:
                if out_dir == "internal":
                    out_dir = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        "grids/",
                    )

                if out_name == "auto":
                    out_name = self.model_name

                fullpath = os.path.join(out_dir, out_name)

                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

                final_fullpath = fullpath.replace(".hdf5", f"_{batch_i + 1}.hdf5")
                init_fullpath = fullpath.replace(".hdf5", "_0.hdf5")
                if os.path.exists(final_fullpath) and not overwrite:
                    logger.warning(
                        f"Skipping batch {batch_i + 1} as {final_fullpath} already exists."
                    )
                    galaxies[batch_i] = None  # Clear the batch to free memory
                    skip = True

            if not skip:
                if multi_node:
                    logger.info("Running pipeline in multi-node mode with MPI.")
                    logger.debug(f"SIZE: {size}, RANK: {rank}")
                else:
                    logger.info("Running in single-node mode.")

                pipeline = Pipeline(
                    emission_model=self.emission_model,
                    nthreads=n_proc,
                    verbose=verbose,
                    comm=comm,
                )

                for key in self.all_parameters.keys():
                    pipeline.add_analysis_func(
                        lambda gal, key=key: gal.all_params[key],
                        result_key=key,
                    )

                pipeline.add_analysis_func(lambda gal: gal.stars.initial_mass, result_key="mass")

                logger.info("Added analysis functions to pipeline.")

                if multi_node:
                    logger.info(f"Pipeline MPI: {pipeline.using_mpi}")

                # Add any extra analysis functions requested by the user.

                for key, params in extra_analysis_functions.items():
                    if callable(params):
                        func = copy.deepcopy(params)
                        params = []
                    else:
                        func = params[0]
                        params = params[1:]

                    pipeline.add_analysis_func(func, f"supp_{key}", *params)

                # pipeline.get_spectra() # Switch off so they aren't saved
                pipeline.get_observed_spectra(self.cosmo)

                if self.instrument.can_do_photometry:
                    pipeline.get_photometry_fluxes(self.instrument)

                if False:
                    pipeline.get_lines(line_ids=["H 1 6562.80A", "O 3 5006.84A", "H 1 4861.32A"])
                    pipeline.get_observed_lines(self.cosmo)

                pipeline.add_galaxies(batch_gals)

                ngal = len(batch_gals)
                start = datetime.now()
                logger.info(f"Running pipeline at {start} for {ngal} galaxies")
                pipeline.run()
                elapsed = datetime.now() - start
                logger.info(f"Finished running pipeline at {datetime.now()} for {ngal} galaxies")
                logger.info(f"Pipeline took {elapsed} to run.")
                if save:
                    # Save the pipeline to a file
                    pipeline.write(fullpath, verbose=0)

                    if multi_node:
                        logger.info("Combining HDF5 files across nodes.")
                        pipeline.combine_files()  # virtual needs work

            if save:
                wav = self.grid.lam.to(Angstrom).value

                if n_batches == 1:
                    final_fullpath = fullpath

                # IF MPI, only do this on rank 0
                add = True
                if multi_node:
                    if rank != 0:
                        add = False
                        logger.debug(f"Skipping adding attributes on rank {rank}.")

                if add:
                    try:
                        os.rename(init_fullpath, final_fullpath)
                    except FileNotFoundError:
                        pass

                    with h5py.File(final_fullpath, "r+") as f:
                        # Add the varying and fixed parameters to the file
                        f.attrs["varying_param_names"] = self.varying_param_names
                        f.attrs["fixed_param_names"] = self.fixed_param_names
                        f.attrs["fixed_param_values"] = self.fixed_param_values
                        f.attrs["fixed_param_units"] = self.fixed_param_units

                        # Write some metadata about the model
                        f.attrs["model_name"] = self.model_name
                        f.attrs["grid_name"] = self.grid.grid_name
                        f.attrs["grid_dir"] = self.grid.grid_dir

                        f.attrs["date_created"] = str(datetime.now())
                        f.attrs["pipeline_time"] = str(elapsed)

                        f.create_dataset("Wavelengths", data=wav)
                        f["Wavelengths"].attrs["Units"] = "Angstrom"

                    logger.info(f"Written pipeline to disk at {final_fullpath}.")

            if not skip:
                del pipeline  # Clean up the pipeline object to free memory
                galaxies[batch_i] = None  # Clear the batch to free memory
                import gc

                gc.collect()  # Force garbage collection to free memory

    def plot_random_galaxy(self, masses, **kwargs):
        """Plot a random galaxy from the list of galaxies."""
        if not self.build_grid:
            idx = np.random.randint(0, len(self.redshifts))
            mass = masses[idx]
            return self.plot_galaxy(idx, log_stellar_mass=mass, **kwargs)

    def plot_galaxy(
        self,
        idx,
        save: bool = True,
        log_stellar_mass: float = 9,
        emission_model_keys: List[str] = ["total"],
        out_dir="./",
    ):
        """Plot the galaxy with the given index."""
        galaxy_params = {}
        for param in self.galaxy_params.keys():
            if isinstance(self.galaxy_params[param], dict):
                galaxy_params[param] = self.process_priors(self.galaxy_params[param])
            elif isinstance(self.galaxy_params[param], (list, np.ndarray)):
                galaxy_params[param] = self.galaxy_params[param][idx]
            else:
                galaxy_params[param] = self.galaxy_params[param]

        if not self.build_grid and len(self.galaxies) == 0:
            # Get idx's from requirements and build galaxy directly
            galaxy = create_galaxy(
                sfh=self.sfhs[idx],
                redshift=self.redshifts[idx],
                metal_dist=self.metal_dists[idx],
                log_stellar_masses=log_stellar_mass,
                grid=self.grid,
                **galaxy_params,
            )
        else:
            if idx >= len(self.galaxies):
                raise ValueError(f"Index {idx} out of range for galaxies.")

            galaxy = self.galaxies[idx]

        # Generate spectra

        if isinstance(emission_model_keys, str):
            emission_model_keys = [emission_model_keys]

        galaxy.stars.get_spectra(self.emission_model)
        galaxy.get_observed_spectra(cosmo=self.cosmo, igm=Inoue14)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")

        plot_dict = {key: galaxy.stars.spectra[key] for key in emission_model_keys}

        colors = {
            key: plt.cm.viridis(i / len(emission_model_keys))
            for i, key in enumerate(plot_dict.keys())
        }

        plot_spectra(
            plot_dict,
            show=False,
            fig=fig,
            ax=ax[0],
            x_units=um,
            quantity_to_plot="fnu",
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

        ax[0].set_yscale("log")

        def custom_xticks(x, pos):
            if x == 0:
                return "0"
            else:
                return f"{x / 1e4:.1f}"

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

            ax[0].plot(
                phot.filters.pivot_lams,
                phot.photo_fnu,
                "+",
                color=colors[emission_model],
                path_effects=[PathEffects.withStroke(linewidth=4, foreground="white")],
            )

            # Get the redshift
            redshift = galaxy.redshift

            # Get the SFH
            stars_sfh = galaxy.stars.get_sfh()
            stars_sfh = stars_sfh / np.diff(10 ** (self.grid.log10age), prepend=0) / yr
            t, sfh = galaxy.stars.sf_hist_func.calculate_sfh()

            ax[1].plot(
                10 ** (self.grid.log10age - 6),
                stars_sfh,
                label=f"{emission_model} SFH",
                color=colors[emission_model],
            )
            ax[1].plot(
                t / 1e6,
                sfh / np.max(sfh) * np.max(stars_sfh),
                label=f"Requested {emission_model} SFH",
                color=colors[emission_model],
                linestyle="--",
            )
            mass = galaxy.stars.initial_mass
            if mass == 0:
                text_gal[emission_model] = f"**{emission_model}**\nNo stars"
            else:
                age = galaxy.stars.calculate_mean_age()
                zmet = galaxy.stars.calculate_mean_metallicity()

            text_gal[emission_model] = f"""{emission_model}
Age: {age.to(Myr):.0f}
$\\log_{{10}}(Z)$: {np.log10(zmet):.2f}
$\\log_{{10}}(M_\\star/M_\\odot)$: {np.log10(mass):.1f}"""

        info_blocks = []
        info_blocks.append(f"$z = {redshift:.2f}$")
        info_blocks.extend(text_gal.values())

        # Format and add galaxy parameters if they exist
        if galaxy_params:
            param_lines = [f"{key}: {value:.2f}" for key, value in galaxy_params.items()]
            info_blocks.append("\n".join(param_lines))

        # Join the blocks with double newlines for clear separation
        textstr = "\n\n".join(info_blocks)

        # Define properties for the text box
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

        # Place the text box on the axes
        ax[1].text(
            0.95,
            0.95,  # Adjusted y-position for better placement with verticalalignment='top'
            textstr,
            transform=ax[1].transAxes,
            fontsize=10,  # Adjusted for better fit
            horizontalalignment="right",
            verticalalignment="top",
            bbox=props,
        )

        # add a secondary axis with AB magnitudes

        # 31.4

        def ab_to_jy(f):
            return 1e9 * 10 ** (f / (-2.5) - 8.9)

        def jy_to_ab(f):
            f = f / 1e9
            return -2.5 * np.log10(f) + 8.9

        secax = ax[0].secondary_yaxis("right", functions=(jy_to_ab, ab_to_jy))
        # set scalar formatter

        max_age = self.cosmo.age(redshift)  # - self.cosmo.age(20)
        max_age = max_age.to(u.Myr).value

        ax[1].set_xlim(0, 2 * max_age)
        # vline at max_age
        ax[1].axvline(max_age, color="red", linestyle="--", linewidth=0.5)

        secax.yaxis.set_major_formatter(ScalarFormatter())
        secax.yaxis.set_minor_formatter(ScalarFormatter())
        secax.set_ylabel("Flux Density [AB mag]")

        ax[1].set_xlabel("Time [Myr]")
        ax[1].set_ylabel(r"SFH [M$_\odot$ yr$^{-1}$]")
        # ax[1].set_yscale('log')
        ax[1].legend()

        self.tmp_redshift = redshift
        self.tmp_time_unit = u.Myr
        # secax = ax[1].secondary_xaxis("top",
        # functions=(self._time_convert, self._z_convert))
        # secax.set_xlabel("Redshift")

        # secax.set_xticks([6, 7, 8, 10, 12, 14, 15, 20])

        ax[0].set_xlim(min_x, max_x)
        ax[0].set_ylim(min_y, max_y)

        print(min_x, max_x)

        if save:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            fig.savefig(f"{out_dir}/{self.model_name}_{idx}.png", dpi=300)
            plt.close(fig)

        return fig

    def _time_convert(self, lookback_time):
        time_unit = getattr(self, "tmp_time_unit", u.yr)
        lookback_time = lookback_time * time_unit
        return z_at_value(
            self.cosmo.lookback_time,
            self.cosmo.lookback_time(self.tmp_redshift) + lookback_time,
        ).value

    def _z_convert(self, z):
        if type(z) in [list, np.ndarray] and len(z) == 0:
            return np.array([])

        time_unit = getattr(self, "tmp_time_unit", u.yr)

        return (
            (self.cosmo.lookback_time(z) - self.cosmo.lookback_time(self.tmp_redshift))
            .to(time_unit)
            .value
        )

    def process_base(
        self,
        out_name,
        log_stellar_masses: unyt_array = None,
        emission_model_key: str = "total",
        out_dir: str = grid_folder,
        n_proc: int = 6,
        overwrite: Union[bool, List[bool]] = False,
        verbose=False,
        batch_size: int = 40_000,
        multi_node: bool = False,
        **extra_analysis_functions,
    ):
        """Run pipeline for this base.

        Implements functionality of CombinedBasis.process_bases for
        a single base. This is a convenience method to allow the
        GalaxyBasis to be run seperately.
        """
        if log_stellar_masses is None:
            assert self.log_stellar_masses is not None, (
                "log_stellar_masses must be provided or set in the GalaxyBasis"
            )
            log_stellar_masses = self.log_stellar_masses

        assert not isinstance(log_stellar_masses, unyt_array), (
            "log_stellar_masses must be a unyt_array"
        )
        assert len(log_stellar_masses) == len(
            self.redshifts
        ), f"""log_stellar_masses must be the same length as redshifts,
            got {len(log_stellar_masses)} and {len(self.redshifts)},
            Calling this method on GalaxyBasis only supports
            the case where all samples have been provided, not
            the case where samples are drawn from a prior and
            combined directly.
            """

        if not isinstance(overwrite, (tuple, list, np.ndarray)):
            overwrite = [overwrite] * 1
        else:
            if len(overwrite) != 1:
                raise ValueError(
                    """overwrite must be a boolean or a
                    list of booleans with the same length as bases"""
                )

        full_out_path = f"{out_dir}/{out_name}.hdf5"
        ngalaxies = len(log_stellar_masses)
        total_batches = int(np.ceil(ngalaxies / batch_size))

        if (
            os.path.exists(full_out_path)
            or os.path.exists(f"{out_dir}/{out_name}_{total_batches}.hdf5")
            and not overwrite[0]
        ):
            logger.info(f"File {full_out_path} already exists. Skipping loading.")
            return
        if os.path.exists(full_out_path) and overwrite[0]:
            logger.info(f"Overwriting {full_out_path}.")

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        galaxies = self._create_galaxies(log_base_masses=log_stellar_masses)

        self.process_galaxies(
            galaxies,
            f"{out_name}.hdf5",
            out_dir=out_dir,
            n_proc=n_proc,
            verbose=verbose,
            save=True,
            emission_model_keys=emission_model_key,
            batch_size=batch_size,
            overwrite=overwrite[0],
            multi_node=multi_node,
            **extra_analysis_functions,
        )

    def create_mock_cat(
        self,
        out_name,
        log_stellar_masses: np.ndarray = None,
        emission_model_key: str = "total",
        out_dir: str = grid_folder,
        n_proc: int = 6,
        overwrite: Union[bool, List[bool]] = False,
        verbose=False,
        batch_size: int = 40_000,
        parameter_transforms_to_save: dict[str, (str, callable)] = None,
        cat_type="photometry",
        compile_grid: bool = True,
        multi_node: bool = False,
        **extra_analysis_functions,
    ):
        """Convenience method which calls CombinedBasis.

        This is a convenience method which allows
        you to not have to pass a GalaxyBasis into
        CombinedBasis, and instead just call
        this method which will run the components for you.

        Parameters
        ----------
        out_name : str
            Name of the output file to save the mock catalog.
        log_stellar_masses : np.ndarray, optional
            Array of log stellar masses to use for the mock catalog,
            Units of log10(M sun), by default None.
            If None, uses the stellar_masses set in the GalaxyBasis.
        emission_model_key : str, optional
            Emission model key to use for the mock catalog,
            by default "total".
        out_dir : str, optional
            Directory to save the output file, by default "grid_folder".
            If "grid_folder", saves to the Synthesizer grids directory.
        n_proc : int, optional
            Number of processes to use for the pipeline, by default 6.
        overwrite : Union[bool, List[bool]], optional
            If True, overwrites the output file if it exists,
            by default False. If a list, must be the same length as bases.
        verbose : bool, optional
            If True, prints verbose output during processing,
            by default False.
        batch_size : int, optional
            Size of each batch of galaxies to process,
            by default 40,000.
        parameter_transforms_to_save : Dict[str: (str, callable)], optional
            Dictionary of parameter transforms to save in the output file.
            Only used for for saving with the simulator to allow
            reconstruction of the model later.
            Should be a dictionary where keys are the parameter names
            in the model, and the values are a tuple.
            The tuple should be (str, callable), where the str is the
            new parameter name to save, and the callable is the function
            which takes the model parameters and returns the new parameter value.
            It can also be (List[str], callable) if the function returns multiple values.
            (e.g. converting one parameter to many.)
            Finally, if you are adding a new parameter which is not in the
            model, you can a direct str: callable pair, which will add a new
            parameter to the model based on the callable function.
        compile_grid : bool, optional
            If True, compiles the grid after processing,
            by default True.
        multi_node : bool, optional
            If True, runs the processing in parallel across multiple nodes,
            by default False. Will only enable this, script still needs to be run
            with slurm or similar.
        cat_type : str, optional
            Type of catalog to create, either "photometry" or "spectra",
            by default "photometry".

        """
        # make a CombinedBasis object with the current GalaxyBasis

        if log_stellar_masses is None:
            assert self.log_stellar_masses is not None, (
                "log_stellar_masses must be provided or set in the GalaxyBasis"
            )
            log_stellar_masses = self.log_stellar_masses

        assert not isinstance(log_stellar_masses, unyt_array), (
            "log_stellar_masses must be not be a unyt_array"
        )

        combined_basis = CombinedBasis(
            bases=[self],
            log_stellar_masses=log_stellar_masses,
            redshifts=self.redshifts,
            base_emission_model_keys=[emission_model_key],
            combination_weights=None,
            out_name=out_name,
            out_dir=out_dir,
            draw_parameter_combinations=False,
        )

        if multi_node:
            logger.info("Running in multi-node mode. Using MPI for parallel processing.")

            galaxy_mask = np.zeros(len(combined_basis.redshifts), dtype=bool)
            total_galaxies = len(combined_basis.redshifts)
            galaxies_per_node = total_galaxies // size
            start_idx = rank * galaxies_per_node
            end_idx = start_idx + galaxies_per_node
            if rank == size - 1:  # Last node gets the remainder
                end_idx = total_galaxies
            galaxy_mask[start_idx:end_idx] = True
            logger.info(f"Node {rank} processing galaxies from {start_idx} to {end_idx}.")
        else:
            galaxy_mask = None

        combined_basis.process_bases(
            n_proc=n_proc,
            overwrite=overwrite,
            verbose=verbose,
            batch_size=batch_size,
            multi_node=multi_node,
            galaxies_mask=galaxy_mask,
            **extra_analysis_functions,
        )

        if compile_grid:
            # Make code wait until all bases are processed
            logger.info("Compiling the grid after processing bases.")

            if cat_type == "photometry":
                combined_basis.create_grid(overwrite=overwrite)
            elif cat_type == "spectra":
                combined_basis.create_spectral_grid(overwrite=overwrite)
            else:
                raise ValueError(
                    f"Unknown catalog type: {cat_type}. Use 'photometry' or 'spectra'."
                )

            out_path = f"{combined_basis.out_dir}/{combined_basis.out_name}"
            if not out_path.endswith(".hdf5"):
                out_path += ".hdf5"

            self._store_model(
                out_path,
                other_info={
                    "emission_model_key": emission_model_key,
                    "timestamp": datetime.now().isoformat(),
                    "cat_type": cat_type,
                },
                parameter_transforms_to_save=parameter_transforms_to_save,
            )

            logger.info("Processed the bases and saved the output.")

            return combined_basis


class CombinedBasis:
    """Class to create a photometry array from Synthesizer pipeline outputs.

    This class combines multiple GalaxyBasis objects, processes them,
    and saves the output to HDF5 files. The simplest operation is a
    single base, where we would provide a single GalaxyBasis object
    and this class will handle running the Synthesizer pipeline,
    processing the output and saving the results. The reason this
    is seperated is because we can combine multiple bases.

    So when the GalaxyBasis is run, we determine photometry/spectra
    with a filler mass (normalization). Then this class will renormalize
    the photometry/spectra to the total stellar masses provided,
    and optionally weight them by the combination of bases. This is done
    so that you can have a case where there is e.g. two bases with
    different SPS grids, and we can build a galaxy where 15% of the mass
    is from the first base, and 85% is from the second base. It can
    also allow flexiblity when generating galaxies in the pipeline,
    as the stellar mass dimension could be neglected in e.g. the LHC
    and applied afterwards.
    """

    def __init__(
        self,
        bases: List[Type[GalaxyBasis]],
        log_stellar_masses: list,
        redshifts: np.ndarray,
        base_emission_model_keys: List[str],
        combination_weights: np.ndarray,
        out_name: str = "combined_basis",
        out_dir: str = grid_folder,
        log_base_masses: Union[float, np.ndarray] = 9,
        draw_parameter_combinations: bool = False,
    ) -> None:
        """Initialize the CombinedBasis object.

        Parameters
        ----------
        bases : List[Type[GalaxyBasis]]
            List of GalaxyBasis objects to combine.
        log_stellar_masses : list
            Array of total stellar masses to renormalize fluxes for.
            in log10(M sun) units.
        redshifts : np.ndarray
            Array of redshifts for the bases.
        base_emission_model_keys : List[str]
            List of emission model keys for each base.
        combination_weights : np.ndarray
            Array of combination weights for the bases.
        out_name : str
            Name of the output file.
        out_dir : str
            Directory to save the output files.
        log_base_masses : unyt_array
            Default mass (or mass array) to use for the galaxies.
            Units of log10(M sun).
        draw_parameter_combinations : bool
            If True, draw parameter combinations for the galaxies.
            If False, create matched galaxies with the same parameters.
        """
        self.bases = bases
        self.log_stellar_masses = log_stellar_masses
        self.redshifts = redshifts
        self.combination_weights = combination_weights
        self.out_name = out_name
        self.out_dir = out_dir
        self.log_base_masses = log_base_masses
        self.base_emission_model_keys = base_emission_model_keys
        self.draw_parameter_combinations = draw_parameter_combinations

        if isinstance(redshifts, (int, float)):
            redshifts = np.full(len(self.log_stellar_masses), redshifts)

        if self.combination_weights is None:
            assert len(self.bases) == 1
            self.combination_weights = [1.0] * len(redshifts)

    def process_bases(
        self,
        n_proc: int = 6,
        overwrite: Union[bool, List[bool]] = False,
        verbose=False,
        batch_size: int = 40_000,
        multi_node: bool = False,
        galaxies_mask: Optional[np.ndarray] = None,
        **extra_analysis_functions,
    ) -> None:
        """Process the bases and save the output to files.

        Parameters
        ----------
        n_proc : int
            Number of processes to use for the pipeline.
        overwrite : bool or list of bools
            If True, overwrite the existing files.
            If False, skip the files that already exist.
            If a list of bools is provided,
            it should have the same length as the number of bases.
        extra_analysis_functions : dict
            Extra analysis functions to add to the pipeline.
            The keys should be the names of the functions,
            and the values should be the functions themselves,
            or a tuple of (function, args). The function
            should take a Galaxy object as the first argument,
            and the args should be the arguments to pass to the function.
            The function should return a single value, an array of values,
            or a dictionary of values (with the same keys for all galaxies).
        """
        if not isinstance(overwrite, (tuple, list, np.ndarray)):
            overwrite = [overwrite] * len(self.bases)
        else:
            if len(overwrite) != len(self.bases):
                raise ValueError(
                    """overwrite must be a boolean or a
                    list of booleans with the same length as bases"""
                )

        for i, base in enumerate(self.bases):
            full_out_path = f"{self.out_dir}/{base.model_name}.hdf5"
            ngalaxies = len(self.log_stellar_masses)
            if galaxies_mask is not None:
                ngalaxies = np.sum(galaxies_mask)
                if ngalaxies == 0:
                    logger.warning(f"No galaxies to process for base {base.model_name}. Skipping.")
                    continue
            total_batches = int(np.ceil(ngalaxies / batch_size))

            if (
                os.path.exists(full_out_path)
                or os.path.exists(f"{self.out_dir}/{base.model_name}_{total_batches}.hdf5")
            ) and not overwrite[i]:
                logger.warning(f"File {full_out_path} already exists. Skipping.")
                continue
            elif os.path.exists(full_out_path) and overwrite[i]:
                logger.warning(f"File {full_out_path} already exists. Overwriting..")
                os.remove(full_out_path)
            elif not os.path.exists(self.out_dir):
                if rank == 0:
                    logger.warning(f"Creating output directory {self.out_dir}.")
                os.makedirs(self.out_dir)

            if self.draw_parameter_combinations:
                galaxies = base._create_galaxies(log_base_masses=self.log_base_masses)
                if galaxies_mask is not None:
                    raise NotImplementedError(
                        "galaxies_mask is not implemented for draw_parameter_combinations=False."
                    )
            else:
                galaxies = base._create_matched_galaxies(
                    log_base_masses=self.log_base_masses, galaxies_mask=galaxies_mask, n_proc=n_proc
                )

            logger.info(f"Created {len(galaxies)} galaxies for base {base.model_name}")
            # Process the galaxies
            base.process_galaxies(
                galaxies,
                f"{base.model_name}.hdf5",
                out_dir=self.out_dir,
                n_proc=n_proc,
                verbose=verbose,
                save=True,
                emission_model_keys=self.base_emission_model_keys[i],
                batch_size=batch_size,
                overwrite=overwrite[i],
                multi_node=multi_node,
                **extra_analysis_functions,
            )

    def load_bases(self, load_spectra=False) -> dict:
        """Load the processed bases from the output directory.

        Parameters
        ----------
        load_spectra : bool
            If True, load the observed spectra from the output files.
            If False, only load the properties and photometry.

        Returns:
        -------
        dict
            A dictionary with the base model names as keys and a dictionary
            of properties, observed spectra, wavelengths, and observed photometry.
        """
        outputs = {}
        for i, base in enumerate(self.bases):
            logger.info(
                f"Emission model key for base {base.model_name}:{self.base_emission_model_keys[i]}"
            )

            full_out_path = f"{self.out_dir}/{base.model_name}.hdf5"
            if not os.path.exists(full_out_path):
                if os.path.exists(f"{self.out_dir}/{base.model_name}_1.hdf5"):
                    import glob

                    # Check if there are multiple files for this base
                    full_out_paths = glob.glob(f"{self.out_dir}/{base.model_name}_*.hdf5")
                    logger.info(f"Found {len(full_out_paths)} files for base {base.model_name}.")

                    full_out_paths = sorted(
                        full_out_paths,
                        key=lambda x: int(x.split("_")[-1].split(".")[0]),
                    )
                else:
                    raise ValueError(
                        f"Synthesizer pipeline output {full_out_path} does not exist. "
                        "Have you run the pipeline using `combined_basis.process_bases` first?"
                    )  # noqa E501
            else:
                full_out_paths = [full_out_path]

            for j, path in tqdm(enumerate(full_out_paths), desc="Loading galaxy properties"):
                properties = {}
                supp_properties = {}
                with h5py.File(path, "r") as f:
                    # Load in which parameters are varying and fixed
                    base.varying_param_names = f.attrs["varying_param_names"]
                    base.fixed_param_names = f.attrs["fixed_param_names"]
                    base.fixed_param_units = f.attrs["fixed_param_units"]
                    base.fixed_param_values = f.attrs["fixed_param_values"]

                    galaxies = f["Galaxies"]

                    property_keys = list(galaxies.keys())
                    property_keys.remove("Stars")

                    for key in property_keys:
                        if key.startswith("supp_"):
                            dic = supp_properties
                            use_key = key[5:]
                        else:
                            dic = properties
                            use_key = key

                        if isinstance(galaxies[key], h5py.Group):
                            dic[use_key] = {}
                            for subkey in galaxies[key].keys():
                                dic[use_key][subkey] = galaxies[key][subkey][()]
                                if hasattr(galaxies[key][subkey], "attrs"):
                                    if "Units" in galaxies[key][subkey].attrs:
                                        unit = galaxies[key][subkey].attrs["Units"]
                                        dic[use_key][subkey] = unyt_array(
                                            dic[use_key][subkey], unit
                                        )
                        else:
                            dic[use_key] = galaxies[key][()]
                            if hasattr(galaxies[key], "attrs"):
                                if "Units" in galaxies[key].attrs:
                                    unit = galaxies[key].attrs["Units"]
                                    dic[use_key] = unyt_array(dic[use_key], unit)

                    if load_spectra:
                        # Get the spectra
                        spec = galaxies["Stars"]["Spectra"]["SpectralFluxDensities"]
                        assert (
                            self.base_emission_model_keys[i] in spec.keys()
                        ), f"""Emission model key {self.base_emission_model_keys[i]}
                            not found in {spec.keys()}"""
                        observed_spectra = spec[self.base_emission_model_keys[i]]
                        observed_spectra = unyt_array(
                            observed_spectra,
                            units=observed_spectra.attrs["Units"],
                        )
                    else:
                        observed_spectra = {}

                    observed_photometry = galaxies["Stars"]["Photometry"]["Fluxes"][
                        self.base_emission_model_keys[i]
                    ]

                    phot = {}
                    for observatory in observed_photometry:
                        phot_inst = observed_photometry[observatory]
                        if isinstance(phot_inst, h5py.Dataset):
                            # THIS IS A HACK TO AVOID LOADING
                            # REST-FRAME FLUXES
                            continue

                        for key in phot_inst.keys():
                            full_key = f"{observatory}/{key}"
                            phot[full_key] = phot_inst[key][()]

                    if j == 0:
                        outputs[base.model_name] = {
                            "properties": properties,
                            "observed_spectra": observed_spectra,
                            "wavelengths": unyt_array(
                                f["Wavelengths"][()],
                                units=f["Wavelengths"].attrs["Units"],
                            ),
                            "observed_photometry": phot,
                            "supp_properties": supp_properties,
                        }

                    else:
                        # Combine the outputs for this base with the previous ones
                        for key in properties.keys():
                            outputs[base.model_name]["properties"][key] = np.concatenate(
                                (
                                    outputs[base.model_name]["properties"][key],
                                    properties[key],
                                )
                            )

                        for key in phot.keys():
                            if key not in outputs[base.model_name]["observed_photometry"]:
                                outputs[base.model_name]["observed_photometry"][key] = []
                            outputs[base.model_name]["observed_photometry"][key] = np.concatenate(
                                (
                                    outputs[base.model_name]["observed_photometry"][key],
                                    phot[key],
                                )
                            )
                        if load_spectra:
                            # Combine the observed spectra (from different files)
                            if "observed_spectra" not in outputs[base.model_name]:
                                outputs[base.model_name]["observed_spectra"] = []

                            outputs[base.model_name]["observed_spectra"] = np.concatenate(
                                (outputs[base.model_name]["observed_spectra"], observed_spectra)
                            )

                        # Combine supplementary properties
                        for key in supp_properties.keys():
                            if key not in outputs[base.model_name]["supp_properties"]:
                                outputs[base.model_name]["supp_properties"][key] = {}
                            if not isinstance(
                                supp_properties[key],
                                dict,
                            ):
                                val = supp_properties[key]
                                supp_properties[key] = {self.base_emission_model_keys[i]: val}
                            if not isinstance(
                                outputs[base.model_name]["supp_properties"][key],
                                dict,
                            ):
                                outputs[base.model_name]["supp_properties"][key] = {
                                    self.base_emission_model_keys[i]: outputs[base.model_name][
                                        "supp_properties"
                                    ][key]
                                }

                            for subkey in supp_properties[key].keys():
                                if subkey not in outputs[base.model_name]["supp_properties"][key]:
                                    outputs[base.model_name]["supp_properties"][key][subkey] = []
                                outputs[base.model_name]["supp_properties"][key][subkey] = (
                                    np.concatenate(
                                        (
                                            outputs[base.model_name]["supp_properties"][key][
                                                subkey
                                            ],
                                            supp_properties[key][subkey],
                                        )
                                    )
                                )

        self.pipeline_outputs = outputs
        return outputs

    def create_grid(
        self,
        override_instrument: Union[Instrument, None] = None,
        save: bool = True,
        overload_out_name: str = "",
        overwrite: bool = False,
    ) -> dict:
        """Creates a grid of SEDs for the given Synthesizer outputs.

        This method assumes each input on CombinedBasis, (redshift, mass,
        and varying parameters) should be combined (e.g. sampling
        every combination of redshift, mass, and varying parameters) to
        create the grid. The 'create_full_grid' method instead assumes
        that the input parameters are already combined, and does not
        sample every combination. Generally the 'create_full_grid' method
        is more useful for the case where you have predrawn parameters
        randomly or from a prior or Latin Hypercube.

        Parameters
        ----------

        override_instrument : Instrument, optional
            If provided, overrides the instrument used for the grid.
        save : bool, optional
            If True, saves the grid to a file.
        overload_out_name : str, optional
            If provided, overrides the output name for the grid.
        overwrite : bool, optional
            If True, overwrites the existing grid file if it exists.

        Returns:
        -------
        dict
            A dictionary containing the grid of SEDs, photometry, and properties.
        -----------

        """
        if not self.draw_parameter_combinations:
            return self.create_full_grid(
                override_instrument,
                overwrite=overwrite,
                save=save,
                overload_out_name=overload_out_name,
            )

        if overload_out_name != "":
            out_name = overload_out_name
        else:
            out_name = self.out_name

        if os.path.exists(f"{self.out_dir}/{out_name}") and not overwrite:
            logger.warning(f"File {self.out_dir}/{out_name} already exists. Skipping.")
            self.load_grid_from_file(f"{self.out_dir}/{out_name}")
            return

        pipeline_outputs = self.load_bases()

        base_filters = self.bases[0].instrument.filters.filter_codes
        for i, base in enumerate(self.bases):
            if base.instrument.filters.filter_codes != base_filters:
                raise ValueError(
                    f"""Base {i} has different filters to base 0.
                    Cannot combine bases with different filters."""
                )

        if override_instrument is not None:
            # Check all filters in override_instrument are in the base filters
            for filter_code in override_instrument.filters.filter_codes:
                if filter_code not in base_filters:
                    raise ValueError(
                        f"""Filter {filter_code} not found in base filters.
                        Cannot override instrument."""
                    )

            filter_codes = override_instrument.filters.filter_codes
        else:
            filter_codes = base_filters

        # filter_codes = [i.split("/")[-1] for i in filter_codes]

        all_outputs = []
        all_params = []
        all_supp_params = []

        ignore_keys = ["redshift"]

        total_property_names = {}
        for i, base in enumerate(self.bases):
            if len(self.bases) > 1:
                total_property_names[base.model_name] = [
                    f"{base.model_name}/{i}"
                    for i in base.varying_param_names
                    if i not in ignore_keys
                ]
            else:
                total_property_names[base.model_name] = base.varying_param_names
            params = pipeline_outputs[base.model_name]["properties"]
            rename_keys = [i for i in base.varying_param_names if i not in ignore_keys]
            for key in list(params.keys()):
                if key in rename_keys:
                    # rename the key to be the base name + parameter name
                    params[f"{base.model_name}/{key}"] = params[key]

        supp_param_keys = list(pipeline_outputs[self.bases[0].model_name]["supp_properties"].keys())
        assert all(
            [
                i in pipeline_outputs[self.bases[0].model_name]["supp_properties"]
                for i in supp_param_keys
            ]
        ), f"""Not all bases have the same supplementary parameters.
            {supp_param_keys} not found in
            {pipeline_outputs[self.bases[0].model_name]["supp_properties"].keys()}"""

        # Deal with any supplementary model parameters.
        # Currently we require that all bases have the same supplementary parameters
        supp_params = {}
        supp_param_units = {}
        for i, base in enumerate(self.bases):
            supp_params[base.model_name] = {}
            for key in supp_param_keys:
                if isinstance(
                    pipeline_outputs[base.model_name]["supp_properties"][key],
                    dict,
                ):
                    subkeys = list(pipeline_outputs[base.model_name]["supp_properties"][key].keys())
                    # Check if the emission model key is in the subkeys
                    if self.base_emission_model_keys[i] not in subkeys:
                        raise ValueError(
                            f"""Emission model key {self.base_emission_model_keys[i]}
                            not found in {subkeys}.
                            Don't know how to deal with
                            dictionary supplementary parameters with other keys."""
                        )
                    value = pipeline_outputs[base.model_name]["supp_properties"][key][
                        self.base_emission_model_keys[i]
                    ]
                else:
                    value = pipeline_outputs[base.model_name]["supp_properties"][key]

                supp_params[base.model_name][key] = value

        # Check if any of the bases have the same varying parameters
        all_combined_param_names = []
        for key, value in total_property_names.items():
            all_combined_param_names.extend(value)

        # Add our standard parameters that are always included
        param_columns = ["redshift", "log_mass"]
        param_units = {}
        param_units["redshift"] = "dimensionless"
        param_units["log_mass"] = "log10(Mstar/Msun)"

        if len(self.bases) > 1:
            param_columns.append("weight_fraction")
            param_units["weight_fraction"] = "dimensionless"

        param_columns.extend(all_combined_param_names)

        for redshift in tqdm(self.redshifts, desc="Creating grid"):
            for log_total_mass in self.log_stellar_masses:
                total_mass = 10**log_total_mass

                for combination in self.combination_weights:
                    mass_weights = np.array(combination) * total_mass

                    scaled_photometries = []
                    base_param_values = []
                    supp_params_values = []

                    for i, base in enumerate(self.bases):
                        outputs = pipeline_outputs[base.model_name]
                        z_base = outputs["properties"]["redshift"]
                        mask = z_base == redshift
                        mass = outputs["properties"]["mass"][mask]

                        if isinstance(mass, unyt_array):
                            mass = mass.to(Msun).value

                        # Calculate the scaling factor for each base
                        scaling_factors = mass_weights[i] / mass
                        base_photometry = np.array(
                            [
                                pipeline_outputs[base.model_name]["observed_photometry"][
                                    filter_code
                                ][mask]
                                for filter_code in filter_codes
                            ],
                            dtype=np.float32,
                        )

                        # Scale the photometry by the scaling factor
                        scaled_photometry = base_photometry * scaling_factors

                        scaled_photometries.append(scaled_photometry)

                        # Get the varying parameters for this base
                        base_params = {}
                        for param_name in total_property_names[base.model_name]:
                            # Extract the original parameter name without the base prefix
                            orig_param = param_name.split("/")[-1]
                            if f"{base.model_name}/{orig_param}" in outputs["properties"]:
                                base_params[param_name] = outputs["properties"][
                                    f"{base.model_name}/{orig_param}"
                                ][mask]
                            elif orig_param in outputs["properties"]:
                                base_params[param_name] = outputs["properties"][orig_param][mask]

                            # add units
                            if isinstance(base_params[param_name], unyt_array):
                                short_param_name = param_name.split("/")[-1]
                                if short_param_name in UNIT_DICT:
                                    param_units[param_name] = UNIT_DICT[short_param_name]
                                else:
                                    param_units[param_name] = str(base_params[param_name].units)
                            else:
                                # If it's not a unyt_array, assume it's dimensionless
                                param_units[param_name] = str(dimensionless)

                        base_param_values.append(base_params)
                        # Get the supplementary parameters for this base
                        # For any supp params that are a flux or luminosity,
                        # scale them by the scaling factor
                        scaled_supp_params = {}
                        for key, value in supp_params[base.model_name].items():
                            if isinstance(value, dict):
                                scaled_supp_params[key] = {}
                                for subkey, subvalue in value.items():
                                    if isinstance(subvalue, unyt_array):
                                        scaled_supp_params[key][subkey] = (
                                            subvalue[mask] * scaling_factors
                                        )
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
                    combinations = np.meshgrid(
                        *[np.arange(i.shape[-1]) for i in scaled_photometries],
                        indexing="ij",
                    )
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

                        params_array[param_idx, i] = log_total_mass
                        param_idx += 1

                        if len(self.bases) > 1:
                            params_array[param_idx, i] = combination[
                                0
                            ]  # assuming this is weight fraction
                            param_idx += 1

                        # Add all varying parameters from each base
                        for j, base in enumerate(self.bases):
                            for param_name in total_property_names[base.model_name]:
                                if param_name in base_param_values[j]:
                                    params_array[param_idx, i] = base_param_values[j][param_name][
                                        combo_indices[j]
                                    ]
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
        param_units = [param_units[i] for i in param_columns]

        out = {
            "photometry": combined_outputs,
            "parameters": combined_params,
            "parameter_names": param_columns,
            "filter_codes": filter_codes,
            "supplementary_parameters": combined_supp_params,
            "supplementary_parameter_names": supp_param_keys,
            "supplementary_parameter_units": supp_param_units,
            "parameter_units": param_units,
        }

        self.grid_photometry = combined_outputs
        self.grid_parameters = combined_params
        self.grid_parameter_names = param_columns
        self.grid_filter_codes = filter_codes
        self.grid_supplementary_parameters = combined_supp_params
        self.grid_supplementary_parameter_names = supp_param_keys

        if save:
            self.save_grid(out, overload_out_name=out_name, overwrite=overwrite)

    def _validate_grid(self, grid_dict: dict, check_type="photometry") -> None:
        """Validate the grid dictionary.

        Parameters
        ----------
        grid_dict : dict
            Dictionary containing the grid data.
            Expected keys are 'photometry', 'parameters', 'parameter_names',
            and 'filter_codes'.
        check_type : str, optional
            Type of data to check, either 'photometry' or 'spectra',
            by default 'photometry'.

        Raises:
        ------
        ValueError
            If any of the required keys are missing or if the data is not expected format.
        """
        required_keys = [
            check_type,
            "parameters",
            "parameter_names",
            "filter_codes",
        ]
        for key in required_keys:
            if key not in grid_dict:
                raise ValueError(f"Missing required key: {key}")

        if not isinstance(grid_dict[check_type], np.ndarray):
            raise ValueError(f"{check_type} must be a numpy array.")

        if not isinstance(grid_dict["parameters"], np.ndarray):
            raise ValueError("Parameters must be a numpy array.")

        if not isinstance(grid_dict["parameter_names"], list):
            raise ValueError("Parameter names must be a list.")

        if not isinstance(grid_dict["filter_codes"], (list, np.ndarray)):
            raise ValueError("Filter codes must be a list.")

        # Check for NAN/INF in photometry and parameters

        assert not np.any(np.isnan(grid_dict[check_type])), (
            f"{check_type} contains NaN values. Please check the input data."
        )
        assert not np.any(np.isinf(grid_dict[check_type])), (
            f"{check_type} contains infinite values. Please check the input data."
        )
        assert not np.any(np.isnan(grid_dict["parameters"])), (
            "Parameters contain NaN values. Please check the input data."
        )
        assert not np.any(np.isinf(grid_dict["parameters"])), (
            "Parameters contain infinite values. Please check the input data."
        )

    def save_grid(
        self,
        grid_dict: dict,  # E.g. output from create_grid
        overload_out_name: str = "",
        overwrite: bool = False,
        grid_params_to_save=["model_name"],
    ) -> None:
        """Save the grid to a file.

        Parameters
        ----------
        grid_dict : dict
            Dictionary containing the grid data.
            Expected keys are 'photometry', 'parameters', 'parameter_names',
            and 'filter_codes'.

        out_name : str, optional
            Name of the output file, by default 'grid.hdf5'
        """
        check_type = "photometry" if "photometry" in grid_dict else "spectra"
        self._validate_grid(grid_dict, check_type=check_type)

        # Check if the output directory exists, if not create it
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        if overload_out_name != "":
            out_name = overload_out_name
        else:
            out_name = self.out_name

        if not out_name.endswith(".hdf5"):
            out_name = f"{out_name}.hdf5"
        # Create the full output path
        full_out_path = os.path.join(self.out_dir, out_name)
        # Check if the file already exists
        if os.path.exists(full_out_path) and not overwrite:
            logger.warning(f"File {full_out_path} already exists. Skipping.")
            return
        elif os.path.exists(full_out_path) and overwrite:
            logger.warning(f"File {full_out_path} already exists. Overwriting.")
            os.remove(full_out_path)
        # Create a new HDF5 file
        with h5py.File(full_out_path, "w") as f:
            # Create a group for the grid data
            grid_group = f.create_group("Grid")
            # Create datasets for the photometry and parameters
            if "photometry" in grid_dict:
                grid_group.create_dataset(
                    "Photometry", data=grid_dict["photometry"], compression="gzip"
                )
            if "spectra" in grid_dict:
                grid_group.create_dataset("Spectra", data=grid_dict["spectra"], compression="gzip")

            grid_group.create_dataset(
                "Parameters", data=grid_dict["parameters"], compression="gzip"
            )

            if "supplementary_parameters" in grid_dict:
                grid_group.create_dataset(
                    "SupplementaryParameters",
                    data=grid_dict["supplementary_parameters"],
                    compression="gzip",
                )

            # Create a dataset for the parameter names
            f.attrs["ParameterNames"] = grid_dict["parameter_names"]
            f.attrs["FilterCodes"] = grid_dict["filter_codes"]
            if "supplementary_parameters" in grid_dict:
                f.attrs["SupplementaryParameterNames"] = grid_dict["supplementary_parameter_names"]
                f.attrs["SupplementaryParameterUnits"] = grid_dict["supplementary_parameter_units"]
            f.attrs["PhotometryUnits"] = "nJy"

            if "parameter_units" in grid_dict:
                f.attrs["ParameterUnits"] = grid_dict["parameter_units"]

            for param in grid_params_to_save:
                out = []
                for base in self.bases:
                    out.append(str(getattr(base, param)))
                f.attrs[param] = out

            # Add a timestamp

            f.attrs["Grids"] = [base.grid.grid_name for base in self.bases]

            f.attrs["CreationDT"] = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Add anything else as a dataset
            for key, value in grid_dict.items():
                if key not in [
                    "photometry",
                    "parameters",
                    "parameter_names",
                    "filter_codes",
                    "supplementary_parameters",
                    "supplementary_parameter_names",
                    "supplementary_parameter_units",
                    "parameter_units",
                ]:
                    if isinstance(value, (np.ndarray, list)) and isinstance(value[0], str):
                        f.attrs[key] = value
                    else:
                        grid_group.create_dataset(key, data=value, compression="gzip")
                        if isinstance(value, (unyt_array, unyt_quantity)):
                            grid_group[key].attrs["Units"] = value.units

    def plot_galaxy_from_grid(
        self,
        index: int,
        show: bool = True,
        save: bool = False,
    ):
        """Plot a galaxy at a grid index.

        Parameters
        ----------
        index : int
            Index of the galaxy in the grid.
        show : bool, optional
            If True, shows the plot. Defaults to True.
        save : bool, optional
            If True, saves the plot. Defaults to False.
        """
        if self.grid_photometry is None:
            raise ValueError("Grid photometry not created. Run create_grid() first.")

        if not hasattr(self, "pipeline_outputs"):
            self.load_bases()

        # Get the parameters for this index
        params = self.grid_parameters[:, index]

        # Get the photometry for this index
        photometry = self.grid_photometry[:, index]

        # Get the filter codes
        filter_codes = [f"JWST/{i}" for i in self.grid_filter_codes]
        filterset = FilterCollection(filter_codes, verbose=False)

        # For each basis, look at which parameters for that basis and match to spectra.

        combined_spectra = []
        total_wavelengths = []
        for i, base in enumerate(self.bases):
            # Get the varying parameters for this base
            base_params = {}
            for i, param_name in enumerate(self.grid_parameter_names):
                # Extract the original parameter name without the base prefix
                if "/" in param_name:
                    basis, orig_param = param_name.split("/")
                else:
                    basis = base.model_name
                    orig_param = param_name
                if basis == base.model_name:
                    base_params[orig_param] = params[i]

            basis_params = list(self.pipeline_outputs[base.model_name]["properties"].keys())

            total_mask = np.ones(
                len(self.pipeline_outputs[base.model_name]["properties"][basis_params[0]]),
                dtype=bool,
            )
            for key in base_params.keys():
                if key not in basis_params:
                    continue
                all_values = self.pipeline_outputs[base.model_name]["properties"][key]
                _i = all_values == base_params[key]
                total_mask = np.logical_and(total_mask, _i)

            assert (
                np.sum(total_mask) == 1
            ), f"""Found {np.sum(total_mask)} matches for {base.model_name}
                with parameters {base_params}. Expected 1 match."""
            j = np.where(total_mask)[0][0]

            # Get the spectra for this index
            spectra = self.pipeline_outputs[base.model_name]["observed_spectra"][j]

            flux_unit = spectra.units
            # get the mass of the spectra and the expected mass to renormalise the spectra
            mass = self.pipeline_outputs[base.model_name]["properties"]["mass"][j]
            expected_mass = 10 ** params[1] * Msun
            scaling_factor = expected_mass / mass
            spectra = spectra * scaling_factor
            # Append the spectra to the combined spectra
            combined_spectra.append(spectra)
            wavs = self.pipeline_outputs[base.model_name]["wavelengths"]
            total_wavelengths.append(wavs)

        for i, wavs in enumerate(total_wavelengths):
            if i == 0:
                continue
            assert np.all(
                wavs == total_wavelengths[0]
            ), f"""Wavelengths for base {i} do not match base 0.
                {wavs} != {total_wavelengths[0]}"""

        weight_pos = "weight_fraction" == self.grid_parameter_names
        weights = params[weight_pos]

        # Only works for combining 2 bases at the moment
        weights = np.array((weights[0], 1 - weights[0]))

        # Stack the spectra according to the combination weights.
        # Spectra has shape (wav, n_bases)
        combined_spectra = np.array(combined_spectra)

        combined_spectra = combined_spectra * weights[:, np.newaxis]

        combined_spectra_summed = np.sum(combined_spectra, axis=0)

        # apply redshift

        combined_spectra_summed = combined_spectra_summed

        fig, ax = plt.subplots(figsize=(10, 6))

        photwavs = filterset.pivot_lams

        ax.scatter(
            photwavs,
            photometry,
            label="Photometry",
            color="red",
            s=10,
            path_effects=[PathEffects.withStroke(linewidth=4, foreground="white")],
        )

        ax.plot(
            wavs * (1 + params[0]),
            combined_spectra_summed,
            label="Combined Spectra",
            color="blue",
        )

        for i, base in enumerate(self.bases):
            # Get the spectra for this index
            spectra = combined_spectra[i]

            ax.plot(
                wavs * (1 + params[0]),
                spectra,
                label=f"{base.model_name} Spectra",
                alpha=0.5,
                linestyle="--",
            )

        ax.set_xlabel("Wavelength (AA)")

        ax.set_xlim(0.8 * np.min(photwavs), 1.2 * np.max(photwavs))

        ax.set_yscale("log")
        ax.set_ylim(1e-2, None)
        ax.legend()

        def ab_to_jy(f):
            return 1e9 * 10 ** (f / (-2.5) - 8.9)

        def jy_to_ab(f):
            f = f / 1e9
            return -2.5 * np.log10(f) + 8.9

        secax = ax.secondary_yaxis("right", functions=(jy_to_ab, ab_to_jy))

        secax.yaxis.set_major_formatter(ScalarFormatter())
        secax.yaxis.set_minor_formatter(ScalarFormatter())
        secax.set_ylabel("Flux Density [AB mag]")

        ax.set_xlabel(f"Wavelength ({wavs.units})")
        ax.set_ylabel(f"Flux Density ({flux_unit})")

        # Text box with parameters and values

        textstr = "\n".join(
            [f"{key}: {value:.2f}" for key, value in zip(self.grid_parameter_names, params)]
        )
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(
            0.5,
            0.98,
            f"index: {index}\n" + textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
            horizontalalignment="center",
        )

        if show:
            plt.show()

        if save:
            fig.savefig(
                f"{self.out_dir}/{self.out_name}_{index}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

        return fig

    def load_grid_from_file(
        self,
        file_path: str,
    ) -> dict:
        """Load the grid from a file.

        Parameters
        ----------
        file_path : str
            Path to the HDF5 file containing the grid data.

        Returns:
        -------
        dict
            Dictionary containing the grid data.
        """
        with h5py.File(file_path, "r") as f:
            grid_data = {
                "parameters": f["Grid"]["Parameters"][()],
                "parameter_names": f.attrs["ParameterNames"],
                "filter_codes": f.attrs["FilterCodes"],
            }

            if "Photometry" in f["Grid"]:
                grid_data["photometry"] = f["Grid"]["Photometry"][()]
                self.grid_photometry = grid_data["photometry"]

            if "Spectra" in f["Grid"]:
                grid_data["spectra"] = f["Grid"]["Spectra"][()]
                self.grid_spectra = grid_data["spectra"]

        self.grid_parameters = grid_data["parameters"]
        self.grid_parameter_names = grid_data["parameter_names"]
        self.grid_filter_codes = grid_data["filter_codes"]

        return grid_data

    def _validate_bases(self, pipeline_outputs, skip_inst=False) -> None:
        # ===== VALIDATION =====
        # Check all bases have the same number of galaxies
        ngal = len(pipeline_outputs[self.bases[0].model_name]["properties"]["mass"])
        for i, base in enumerate(self.bases):
            model_name = base.model_name
            if len(pipeline_outputs[model_name]["properties"]["mass"]) != ngal:
                raise ValueError(
                    f"""Base {i} has different number of galaxies to base 0.
                    Cannot combine bases with different number of galaxies."""
                )

        # Validate input arrays
        for array_name, array in [
            ("redshifts", self.redshifts),
            ("log_stellar_masses", self.log_stellar_masses),
            ("combination_weights", self.combination_weights),
        ]:
            if len(array) != ngal:
                raise ValueError(
                    f"""{array_name} length {len(array)} does not match
                    number of galaxies {ngal}."""
                )

        if not skip_inst:
            # Validate filters
            base_filters = self.bases[0].instrument.filters.filter_codes
            for i, base in enumerate(self.bases):
                if base.instrument.filters.filter_codes != base_filters:
                    raise ValueError(
                        f"""Base {i} has different filters to base 0.
                        Cannot combine bases with different filters."""
                    )

    def create_full_grid(
        self,
        override_instrument: Union[Instrument, None] = None,
        save: bool = True,
        overload_out_name: str = "",
        overwrite: bool = False,
        spectral_mode=False,
    ) -> None:
        """Create a complete grid of SEDs by combining galaxy bases.

        This method handles both single-base and multi-base scenarios,
        properly scaling according to masses and weights.
        It constructs a grid without sampling, generating the full set of
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
            raise AssertionError(
                """Cannot create full grid with draw_parameter_combinations
                  set to True. Set to False to create full grid."""
            )

        # Load base model data
        pipeline_outputs = self.load_bases(load_spectra=spectral_mode)

        self._validate_bases(pipeline_outputs, skip_inst=spectral_mode)

        base_filters = self.bases[0].instrument.filters.filter_codes
        ngal = len(pipeline_outputs[self.bases[0].model_name]["properties"]["mass"])

        # Determine which filters to use
        if override_instrument is not None:
            # Ensure all requested filters exist in the base
            for filter_code in override_instrument.filters.filter_codes:
                if filter_code not in base_filters:
                    raise ValueError(
                        f"""Filter {filter_code} not found in base filters.
                        Cannot override instrument."""
                    )
            filter_codes = override_instrument.filters.filter_codes
        else:
            filter_codes = base_filters

        # Strip path info from filter codes
        # filter_codes = [code.split("/")[-1] for code in filter_codes]

        # ===== PARAMETER SETUP =====
        # Set up parameter names
        ignore_keys = ["redshift"]
        param_columns = ["redshift", "log_mass"]

        # Add weight fraction parameter for multiple bases
        is_multi_base = len(self.bases) > 1
        if is_multi_base:
            param_columns.append("weight_fraction")

        # Set up per-base parameter tracking
        total_property_names = {}
        for base in self.bases:
            model_name = base.model_name
            # Track parameters unique to this base
            if len(self.bases) > 1:
                varying_params = [
                    f"{model_name}/{param}"
                    for param in base.varying_param_names
                    if param not in ignore_keys
                ]
            else:
                varying_params = [
                    param for param in base.varying_param_names if param not in ignore_keys
                ]
            total_property_names[model_name] = varying_params
            param_columns.extend(varying_params)

            # Rename parameter keys in the pipeline outputs
            params = pipeline_outputs[model_name]["properties"]
            for param in base.varying_param_names:
                if param not in ignore_keys and param in params:
                    params[f"{model_name}/{param}"] = params[param]

        # ===== SUPPLEMENTARY PARAMETER SETUP =====
        # Get the list of supplementary parameters from the first base
        supp_param_keys = list(pipeline_outputs[self.bases[0].model_name]["supp_properties"].keys())

        # Validate all bases have the same supplementary parameters
        for key in supp_param_keys:
            for base in self.bases:
                if key not in pipeline_outputs[base.model_name]["supp_properties"]:
                    raise ValueError(
                        f"""Supplementary parameter {key}
                        not found in base {base.model_name}."""
                    )

        # Process supplementary parameters for each base
        supp_params = {}
        supp_param_units = {}

        for i, base in enumerate(self.bases):
            model_name = base.model_name
            supp_params[model_name] = {}

            for key in supp_param_keys:
                value = pipeline_outputs[model_name]["supp_properties"][key]

                # Handle nested dictionary case (typically for emission models)
                if isinstance(value, dict):
                    emission_key = self.base_emission_model_keys[i]
                    if emission_key not in value:
                        raise ValueError(
                            f"""Emission model key {emission_key} not
                            found in supplementary parameter {key}."""
                        )
                    supp_params[model_name][key] = value[emission_key]
                else:
                    supp_params[model_name][key] = value

        # ===== PROCESS EACH galaxy =====
        all_outputs = []
        all_params = []
        all_supp_params = []

        for pos in range(ngal):
            redshift = self.redshifts[pos]
            log_total_mass = self.log_stellar_masses[pos]
            # mass in solar masses
            total_mass = 10**log_total_mass
            #
            weights = self.combination_weights[pos]

            # Create per-base data structures to hold galaxy information at this redshift
            base_data = []

            # For each base, extract galaxies at this redshift
            # and calculate scaling factors
            for i, base in enumerate(self.bases):
                model_name = base.model_name
                model_output = pipeline_outputs[model_name]

                masses = np.array([model_output["properties"]["mass"][pos]])

                if is_multi_base:
                    # Multiple bases: scale by weight for this base
                    scaling_factors = weights[i] * total_mass / masses
                else:
                    # Single base: scale by total mass directly
                    scaling_factors = total_mass / masses

                scaling_factors = (
                    scaling_factors.value
                    if isinstance(scaling_factors, unyt_array)
                    else scaling_factors
                )

                if not spectral_mode:
                    # Get photometry for all filters and scale it
                    photometry = np.array(
                        [model_output["observed_photometry"][code][pos] for code in filter_codes],
                        dtype=np.float32,
                    )
                    scaled_phot = photometry * scaling_factors
                else:
                    # For spectral mode, use the observed spectra
                    scaled_phot = np.array(
                        [model_output["observed_spectra"][pos]],
                        dtype=np.float32,
                    )
                    # Scale the spectra by the scaling factor
                    scaled_phot = scaled_phot * scaling_factors[:, np.newaxis]

                # Extract parameters for this base
                params_dict = {}
                for param_name in total_property_names.get(model_name, []):
                    original_param = param_name.split("/")[-1]
                    # Check both possible parameter locations
                    if f"{model_name}/{original_param}" in model_output["properties"]:
                        params_dict[param_name] = model_output["properties"][
                            f"{model_name}/{original_param}"
                        ][pos]
                    elif original_param in model_output["properties"]:
                        params_dict[param_name] = model_output["properties"][original_param][pos]
                # Process supplementary parameters
                supp_dict = {}
                for key, value in supp_params[model_name].items():
                    if isinstance(value, dict):
                        supp_dict[key] = {}
                        for subkey, subvalue in value.items():
                            if check_scaling(subvalue):
                                supp_dict[key][subkey] = subvalue[pos] * scaling_factors
                            elif check_log_scaling(subvalue):
                                supp_dict[key][subkey] = subvalue[pos] + np.log10(scaling_factors)
                            else:
                                supp_dict[key][subkey] = subvalue[pos]
                            supp_param_units[key] = (
                                str(subvalue.units)
                                if isinstance(subvalue, unyt_array)
                                else "dimensionless"
                            )
                    else:
                        if check_scaling(value):
                            supp_dict[key] = value[pos] * scaling_factors
                        elif check_log_scaling(value):
                            supp_dict[key] = value[pos] + np.log10(scaling_factors)
                        else:
                            supp_dict[key] = value[pos]
                        supp_param_units[key] = (
                            str(value.units) if isinstance(value, unyt_array) else "dimensionless"
                        )

                # Store all relevant data for this base

                base_data.append(
                    {
                        "photometry": scaled_phot,
                        "params": params_dict,
                        "supp_params": supp_dict,
                        "num_items": len(masses),
                    }
                )

            # Decide how to process based on number of bases
            if not is_multi_base:
                # SINGLE BASE CASE - just use the data directly
                base = base_data[0]
                num_items = base["num_items"]

                # Use photometry directly
                output = base["photometry"]
                output = np.squeeze(output)
                output_array = output[:, np.newaxis] if output.ndim == 1 else output

                # Create parameter array
                params_array = np.zeros((len(param_columns), num_items))
                param_idx = 0

                param_units = []

                # Fill standard parameters
                params_array[param_idx, :] = redshift
                param_units.append("dimensionless")
                param_idx += 1
                params_array[param_idx, :] = log_total_mass
                param_units.append("log10_Msun")
                param_idx += 1

                # Fill varying parameters
                for param_name in total_property_names.get(self.bases[0].model_name, []):
                    if param_name in base["params"]:
                        params_array[param_idx, :] = base["params"][param_name]
                        if param_name.split("/")[-1].lower() in UNIT_DICT.keys():
                            param_units.append(UNIT_DICT[param_name.split("/")[-1].lower()])
                        else:
                            param_units.append(
                                str(base["params"][param_name].units)
                                if isinstance(base["params"][param_name], unyt_array)
                                else "dimensionless"
                            )

                    param_idx += 1

                # Process supplementary parameters
                supp_array = np.zeros((len(supp_param_keys), num_items))
                for k, param_name in enumerate(supp_param_keys):
                    data = base["supp_params"][param_name]
                    if isinstance(data, unyt_array):
                        data = data.value
                    supp_array[k, :] = data

            else:
                # MULTI-BASE CASE - create combinations

                # Get the number of items per base
                items_per_base = [base["num_items"] for base in base_data]

                # Create meshgrid of indices for combination
                mesh_indices = np.meshgrid(*[np.arange(n) for n in items_per_base], indexing="ij")

                # Reshape to get all combinations:
                # array of shape (total_combinations, num_bases)
                combinations = np.array([indices.flatten() for indices in mesh_indices]).T
                num_combinations = len(combinations)

                # Create output arrays
                n = len(filter_codes) if not spectral_mode else base_data[0]["photometry"].shape[1]
                output_array = np.zeros((n, num_combinations))
                params_array = np.zeros((len(param_columns), num_combinations))
                supp_array = np.zeros((len(supp_param_keys), num_combinations))
                param_units = []

                # Fill arrays
                for i, combo_indices in enumerate(combinations):
                    # Fill photometry - combine the weighted contributions
                    for j, base_idx in enumerate(combo_indices):
                        output_array[:, i] += base_data[j]["photometry"][base_idx]

                    # Fill parameters
                    param_idx = 0

                    # Standard parameters first
                    params_array[param_idx, i] = redshift
                    if i == 0:
                        param_units.append("dimensionless")
                    param_idx += 1
                    params_array[param_idx, i] = log_total_mass
                    if i == 0:
                        param_units.append("log10_Msun")
                    param_idx += 1

                    params_array[param_idx, i] = weights[0]  # weight fraction
                    param_idx += 1
                    if i == 0:
                        param_units.append("dimensionless")

                    # Add all varying parameters from each base
                    for j, base in enumerate(self.bases):
                        model_name = base.model_name
                        for param_name in total_property_names.get(model_name, []):
                            if param_name in base_data[j]["params"]:
                                params_array[param_idx, i] = base_data[j]["params"][param_name]
                                if param_name.split("/")[-1].lower() in UNIT_DICT.keys():
                                    param_units.append(UNIT_DICT[param_name.split("/")[-1].lower()])
                                else:
                                    param_units.append(
                                        str(base_data[j]["params"][param_name].units)
                                        if isinstance(
                                            base_data[j]["params"][param_name], unyt_array
                                        )
                                        else "dimensionless"
                                    )
                                param_idx += 1

                    # Fill supplementary parameters
                    for k, param_name in enumerate(supp_param_keys):
                        total_value = 0
                        for j, base_idx in enumerate(combo_indices):
                            data = base_data[j]["supp_params"][param_name]
                            if isinstance(data, unyt_array):
                                total_value += data[base_idx].value
                            elif hasattr(data, "__len__") and len(data) > base_idx:
                                total_value += data[base_idx]
                            else:
                                total_value += data  # Scalar case
                        supp_array[k, i] = total_value

            # Append results for this redshift to the global arrays
            # Give output_array shape (n, 1) if it is 1D
            # if output_array.ndim == 1:
            #    output_array = output_array[:, np.newaxis]
            all_outputs.append(output_array)
            all_params.append(params_array)
            all_supp_params.append(supp_array)

        # ===== COMBINE RESULTS =====

        combined_outputs = np.hstack(all_outputs)
        combined_params = np.hstack(all_params)
        combined_supp_params = np.hstack(all_supp_params)

        # Convert units dict to list
        supp_param_units_list = [
            supp_param_units.get(name, str(dimensionless)) for name in supp_param_keys
        ]

        # Print summary
        logger.info(f"Combined outputs shape: {combined_outputs.shape}")
        logger.info(f"Combined parameters shape: {combined_params.shape}")
        logger.info(f"Combined supplementary parameters shape: {combined_supp_params.shape}")
        if not spectral_mode:
            logger.info(f"Filter codes: {filter_codes}")
        else:
            logger.info("Spectral mode enabled, using wavelengths as filter codes.")
        logger.info(f"Parameter names: {param_columns}")
        logger.info(f"Parameter units: {param_units}")

        # Check combined_outputs is 2D
        if combined_outputs.ndim == 1:
            raise ValueError(
                "Combined outputs should be a 2D array with \
                 shape (n_filters, n_galaxies)."
            )

        assert len(param_units) == len(param_columns), (
            "Parameter units length does not match parameter columns length."
            f"Expected {len(param_columns)}, got {len(param_units)}."
        )

        if not spectral_mode:
            assert combined_outputs.shape[0] == len(filter_codes), (
                "Output photometry shape does not match number of filters."
                f"Expected {len(filter_codes)}, got {combined_outputs.shape[0]}."
            )

        assert combined_params.shape[0] == len(param_columns), (
            "Output parameters shape does not match number of parameter columns."
            f"Expected {len(param_columns)}, got {combined_params.shape[0]}."
        )

        assert combined_supp_params.shape[0] == len(supp_param_keys), (
            "Output supplementary parameters shape does not match number of keys."
            f"Expected {len(supp_param_keys)}, got {combined_supp_params.shape[0]}."
        )

        # Create output dictionary
        out = {
            "parameters": combined_params,
            "parameter_names": param_columns,
            "supplementary_parameters": combined_supp_params,
            "supplementary_parameter_names": supp_param_keys,
            "supplementary_parameter_units": supp_param_units_list,
            "parameter_units": param_units,
        }

        # Update object attributes
        if spectral_mode:
            out["spectra"] = combined_outputs
            self.grid_spectra = combined_outputs
            # 'Grid filter codes' can just be the wavelength array here
            self.grid_filter_codes = model_output["wavelengths"].to("um").value
            out["filter_codes"] = self.grid_filter_codes
        else:
            out["photometry"] = combined_outputs
            out["filter_codes"] = filter_codes
            self.grid_photometry = combined_outputs
            self.grid_filter_codes = filter_codes

        self.grid_parameters = combined_params
        self.grid_parameter_names = param_columns
        self.grid_supplementary_parameters = combined_supp_params
        self.grid_supplementary_parameter_names = supp_param_keys
        self.grid_supplementary_parameter_units = supp_param_units_list
        self.grid_parameter_units = param_units

        # Save results if requested
        if save:
            self.save_grid(out, overload_out_name=overload_out_name, overwrite=overwrite)

    def create_spectral_grid(
        self,
        save: bool = True,
        overload_out_name: str = "",
        overwrite: bool = False,
    ) -> dict:
        """Creates a parameter grid for spectroscopic observations.

        Wrapper for `create_full_grid`, but specifically for spectroscopic outputs,
        to match e.g. NIRSpec IFU, or DESI etc. The only spectral sampling
        currently supported is the one used in the instrument/grid combination.

        Parameters
        ----------
        save : bool, optional
            If True, saves the grid to a file.
        overload_out_name : str, optional
            If provided, overrides the output name for the grid.
        overwrite : bool, optional
            If True, overwrites the existing grid file if it exists.

        Returns:
        -------
        dict
            A dictionary containing the grid of SEDs, photometry, and properties.
        """
        return self.create_full_grid(
            override_instrument=None,
            save=save,
            overload_out_name=overload_out_name,
            overwrite=overwrite,
            spectral_mode=True,
        )


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
        emitter_params: dict = None,
        cosmo: Cosmology = Planck18,
        param_order: Union[None, list] = None,
        param_units: dict = None,
        param_transforms: dict[callable] = None,
        out_flux_unit: str = "nJy",
        required_keys=["redshift", "log_mass"],
        extra_functions: List[callable] = None,
        normalize_method: str = None,
        output_type: str = "photo_fnu",
        include_phot_errors: bool = False,
        set_self=False,
        depths: Union[np.ndarray, unyt_array] = None,
        depth_sigma: int = 5,
        noise_models: Union[None, Dict[str, EmpiricalUncertaintyModel]] = None,
        fixed_params: dict = None,
        photometry_to_remove=None,
        ignore_params: list = None,
        ignore_scatter: bool = False,
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
        photometry_to_remove : list
            List of photometry to remove from the output.
            This is used to remove specific filters from the output.
            If None, no photometry is removed. Default is None.
            Should match filter codes in the instrument filters.
        ignore_params : list
            List of parameters which are sampled which won't be checked for use against the model.
        ignore_scatter : bool
            If True, ignore scatter in the empirical uncertainty model. Default is False.

        """
        if fixed_params is None:
            fixed_params = {}
        if param_units is None:
            param_units = {}
        if param_transforms is None:
            param_transforms = {}
        if emitter_params is None:
            emitter_params = {"stellar": [], "galaxy": []}
        if extra_functions is None:
            extra_functions = []
        if photometry_to_remove is None:
            photometry_to_remove = []
        if not isinstance(required_keys, list):
            raise TypeError(f"required_keys must be a list. Got {type(required_keys)} instead.")
        if ignore_params is None:
            ignore_params = []

        assert isinstance(grid, Grid), f"Grid must be a subclass of Grid. Got {type(grid)} instead."
        self.grid = grid

        assert isinstance(instrument, Instrument), f"""Instrument must be a subclass of Instrument.
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
        self.depths = depths
        self.ignore_params = ignore_params
        self.ignore_scatter = ignore_scatter

        if len(photometry_to_remove) > 0:
            self.update_photo_filters(
                photometry_to_remove=photometry_to_remove, photometry_to_add=None
            )

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
        self.num_emitter_params = np.sum([len(emitter_params[key]) for key in emitter_params])
        self.emission_model.save_spectra(emission_model_key)

        self.total_possible_keys = (
            self.sfh_params
            + self.zdist_params
            + self.optional_sfh_params
            + self.optional_zdist_params
            + required_keys
        )

    def update_photo_filters(self, photometry_to_remove=None, photometry_to_add=None):
        """Update the photometric filters used in the simulation.

        This method allows you to modify the set of photometric filters
        used in the simulation by removing or adding specific filters.
        It updates the instrument's filter collection accordingly.

        Parameters
        ----------
        photometry_to_remove : list, optional
            List of filter codes to remove from the current set of filters.
            If None, no filters will be removed.
        photometry_to_add : list, optional
            List of filter codes to add to the current set of filters.
            If None, no filters will be added.

        """
        if photometry_to_remove is None:
            photometry_to_remove = []
        if photometry_to_add is None:
            photometry_to_add = []

        filter_codes = self.instrument.filters.filter_codes
        new_filters = []

        # Remove specified filters
        for filter_code in filter_codes:
            if filter_code not in photometry_to_remove:
                new_filters.append(filter_code)

        # Add specified filters if they are not already present
        for filter_code in photometry_to_add:
            if filter_code not in new_filters:
                new_filters.append(filter_code)

        self.instrument.filters = FilterCollection(filter_codes=new_filters)
        print(f"Updated filters: {self.instrument.filters.filter_codes}")

    @classmethod
    def from_grid(
        cls,
        grid_path: str,
        override_synthesizer_grid_dir: Union[None, str, bool] = True,
        override_emission_model: Union[None, EmissionModel] = None,
        **kwargs,
    ):
        """Create a GalaxySimulator from a grid file.

        This method reads a grid file in HDF5 format, extracts the necessary
        components such as the grid, instrument, cosmology, SFH model,
        and emission model, and then instantiates a GalaxySimulator object.

        Parameters
        ----------
        grid_path : str
            Path to the grid file in HDF5 format.
        override_synthesizer_grid_dir : Union[None, str], optional
            If provided, this directory will override the synthesizer grid directory
            specified in the grid file. This is useful for using a model
            on a different computer or environment where the grid directory
            is not the same as the one used to create the grid file.
            If True, and the grid_dir saved in the file does not exist,
            it will check for a SYNTHESIZER_GRID_DIR environment variable
            and use that as the grid directory. If a string is provided,
            it will use that as the grid directory.
        override_emission_model : Union[None, EmissionModel], optional
            If provided, this emission model will override the one in the grid file.
        **kwargs : dict
            Additional keyword arguments to pass to the GalaxySimulator constructor.

        Returns:
        -------
        GalaxySimulator
            An instance of GalaxySimulator initialized with the grid data.
        """
        # Open h5py, look for 'Model' and instatiate by reading the grid.

        if not os.path.exists(grid_path):
            raise FileNotFoundError(
                f"Grid path {grid_path} does not exist. Cannot create GalaxySimulator."
            )

        with h5py.File(grid_path, "r") as f:
            if "Model" not in f:
                raise ValueError(
                    f"""Grid file {grid_path} does not contain 'Model' group.
                    Cannot create GalaxySimulator."""
                )

            model_group = f["Model"]

            # Step 1. Make grid
            lam = unyt_array(
                model_group["Instrument/Filters/Header/Wavelengths"][:], units=Angstrom
            )

            grid_name = model_group.attrs["grid_name"]
            grid_dir = model_group.attrs.get("grid_dir", None)
            if override_synthesizer_grid_dir is not None and not os.path.exists(grid_dir):
                if isinstance(override_synthesizer_grid_dir, str):
                    grid_dir = override_synthesizer_grid_dir
                elif override_synthesizer_grid_dir is True:
                    # Check for SYNTHESIZER_GRID_DIR environment variable
                    grid_dir = os.getenv("SYNTHESIZER_GRID_DIR", None)
                    if grid_dir is None:
                        raise ValueError("SYNTHESIZER_GRID_DIR environment variable not set.")

            grid = Grid(grid_name, grid_dir, new_lam=lam)

            # Step 2. Make instrument
            instrument = Instrument._from_hdf5(model_group["Instrument"])

            # Step 3 - recreate cosmology
            cosmo_mapping = model_group.attrs.get("cosmology", None)

            try:
                cosmo = Cosmology.from_format(cosmo_mapping, format="yaml")
            except Exception:
                from astropy.cosmology import Planck18 as cosmo

                print("Failed to load cosmology from HDF5. Using Planck18 instead.")

            # Step 4 - Collect sfh_model

            sfh_model_name = model_group.attrs.get("sfh_class", None)

            sfh_model = getattr(SFH, sfh_model_name, None)
            if sfh_model is None:
                raise ValueError(
                    f"""SFH model {sfh_model_name} not found in SFH module.
                    Cannot create GalaxySimulator."""
                )

            zdist_model_name = model_group.attrs.get("metallicity_distribution_class", None)

            zdist_model = getattr(ZDist, zdist_model_name, None)
            if zdist_model is None:
                raise ValueError(
                    f"""ZDist model {zdist_model_name} not found in ZDist module.
                    Cannot create GalaxySimulator."""
                )

            # recreate emission model
            em_group = model_group["EmissionModel"]
            emission_model_key = model_group.attrs.get("emission_model_key", "total")

            if "emission_model_key" in kwargs:
                emission_model_key = kwargs.pop("emission_model_key")

            if override_emission_model is not None:
                emission_model = override_emission_model

            else:
                emission_model_name = em_group.attrs["name"]
                import synthesizer.emission_models as em
                import synthesizer.emission_models.attenuation as dm
                import synthesizer.emission_models.dust.emission as dem

                emission_model = getattr(em, emission_model_name, None)

                if emission_model is None:
                    raise ValueError(
                        f"Emission model {emission_model_name} not found in synthesizer.emission_models. Cannot create GalaxySimulator."  # noqa: E501
                    )

                if "dust_law" in em_group.attrs:
                    dust_model_name = em_group.attrs["dust_law"]
                    dust_model = getattr(dm, dust_model_name, None)

                    if dust_model is None:
                        raise ValueError(
                            f"Dust model {dust_model_name} not found in synthesizer.emission_models.dust. Cannot create GalaxySimulator."  # noqa: E501
                        )

                    dust_model_params = {}
                    dust_param_keys = em_group.attrs["dust_attenuation_keys"]
                    dust_param_values = em_group.attrs["dust_attenuation_values"]
                    dust_param_units = em_group.attrs["dust_attenuation_units"]

                    for key, value, unit in zip(
                        dust_param_keys, dust_param_values, dust_param_units
                    ):
                        if unit != "":
                            dust_model_params[key] = unyt_array(value, unit)
                        else:
                            dust_model_params[key] = value

                    dust_model = dust_model(**dust_model_params)
                else:
                    dust_model = None

                if "dust_emission" in em_group.attrs:
                    dust_emission_model_name = em_group.attrs["dust_emission"]
                    dust_emission_model = getattr(dem, dust_emission_model_name, None)

                    if dust_emission_model is None:
                        raise ValueError(
                            f"Dust emission model {dust_emission_model_name} not found in synthesizer.emission_models. Cannot create from_grid."  # noqa: E501
                        )

                    dust_emission_model_params = {}
                    dust_emission_param_keys = em_group.attrs["dust_emission_keys"]
                    dust_emission_param_values = em_group.attrs["dust_emission_values"]
                    dust_emission_param_units = em_group.attrs["dust_emission_units"]
                    for key, value, unit in zip(
                        dust_emission_param_keys,
                        dust_emission_param_values,
                        dust_emission_param_units,
                    ):
                        if unit != "":
                            dust_emission_model_params[key] = unyt_array(value, unit)
                        else:
                            dust_emission_model_params[key] = value
                    cmb = dust_emission_model_params.pop("cmb_factor", None)
                    dust_emission_model_params.pop("temperature_z", None)
                    if cmb is not None:
                        dust_emission_model_params["cmb_heating"] = cmb != 1
                    dust_emission_model = dust_emission_model(**dust_emission_model_params)
                else:
                    dust_emission_model = None

                em_keys = em_group.attrs["parameter_keys"]
                em_values = em_group.attrs["parameter_values"]
                em_units = em_group.attrs["parameter_units"]

                emission_model_params = {}
                for key, value, unit in zip(em_keys, em_values, em_units):
                    if unit != "":
                        emission_model_params[key] = unyt_array(value, unit)
                    elif value.isnumeric() or value.replace(".", "", 1).isdigit():
                        emission_model_params[key] = float(value)
                    else:
                        emission_model_params[key] = value

                if dust_model is not None:
                    emission_model_params["dust_curve"] = dust_model

                if dust_emission_model is not None:
                    emission_model_params["dust_emission_model"] = dust_emission_model

                # get arguments from inspect of emission_model
                sig = inspect.signature(emission_model).parameters
                if "dust_emission" in sig and "dust_emission_model" not in sig:
                    emission_model_params["dust_emission"] = dust_emission_model
                    emission_model_params.pop("dust_emission_model", None)

                emission_model = emission_model(
                    grid=grid,
                    **emission_model_params,
                )

            # Step 5 - Collect emitter params

            stellar_params = model_group.attrs.get("stellar_params", {})

            emitter_params = {
                "stellar": stellar_params,
                "galaxy": {},
            }

            # Step 7 - work out order and units
            param_order = f.attrs["ParameterNames"]
            units = f.attrs.get("ParameterUnits")
            ignore = ["dimensionless", "log", "mag"]

            param_units = {}
            for p, u in zip(param_order, units):
                skip = False
                for i in ignore:
                    if i in u:
                        skip = True
                        break
                if not skip:
                    param_units[p] = Unit(u)

            # Step 8 - fixed_params

            fixed_param_names = model_group.attrs.get("fixed_param_names", [])
            fixed_param_values = model_group.attrs.get("fixed_param_values", [])
            fixed_param_units = model_group.attrs.get("fixed_param_units", [])
            fixed_params = {}
            for name, value, unit in zip(fixed_param_names, fixed_param_values, fixed_param_units):
                if unit != "":
                    fixed_params[name] = unyt_array(value, unit)
                else:
                    fixed_params[name] = value

            # Step 9 - Collect param transforms.
            # TEMP
            param_transforms = {}

            if "Transforms" in model_group:
                transform_group = model_group["Transforms"]
                for key in transform_group.keys():
                    # need to evaluate the function
                    code = transform_group[key][()].decode("utf-8")
                    code = f"\n{code}\n"
                    # Remove excess indentation
                    code = inspect.cleandoc(code)
                    func = exec(code, globals(), locals())
                    func_name = code.split("def ")[-1].split("(")[0]
                    func = locals()[func_name]
                    param_transforms[key] = func
                    if "new_parameter_name" in transform_group[key].attrs:
                        new_key = transform_group[key].attrs["new_parameter_name"]
                        param_transforms[key] = (new_key, func)

            dict_create = dict(
                sfh_model=sfh_model,
                zdist_model=zdist_model,
                grid=grid,
                emission_model=emission_model,
                emission_model_key=emission_model_key,
                instrument=instrument,
                cosmo=cosmo,
                emitter_params=emitter_params,
                param_order=param_order,
                param_units=param_units,
                param_transforms=param_transforms,
                fixed_params=fixed_params,
            )
            dict_create.update(kwargs)
            return cls(**dict_create)

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

        params.update(self.fixed_params)

        for key in self.required_keys:
            if key not in params:
                raise ValueError(f"Missing required parameter {key}. Cannot create photometry.")

        mass = 10 ** params["log_mass"] * Msun

        # Check if we have sfh_params and zdist_params

        for key in params:
            if key in self.param_units:
                params[key] = params[key] * self.param_units[key]

        for key in self.param_transforms:
            value = self.param_transforms[key]
            if isinstance(value, tuple):
                name = self.param_transforms[key][0]
                func = self.param_transforms[key][1]

                if key in params:
                    params[name] = func(params[key])
                else:
                    params[name] = func(params)
            elif callable(value):
                params[key] = value(params[key])

        # Check if we have all SFH and ZDist parameters
        for key in self.sfh_params + self.zdist_params:
            if key not in params:
                raise ValueError(
                    f"""Missing required parameter {key} for SFH or ZDist.
                    Cannot create photometry."""
                )

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
                if key in self.ignore_params:
                    found = True
                    break
            if not found:
                logger.info(f"Emitter params are {self.emitter_params}")
                raise ValueError(
                    f"Parameter {key} not found in emitter params.Cannot create photometry."
                )

            else:
                found_params.append(key)

        # Check we understand all the parameters

        # assert len(found_params) - len(self.ignore_params) >= self.num_emitter_params, (
        #    f"Found {len(found_params)} parameters but expected {self.num_emitter_params}."
        #    "Cannot create photometry."
        # )

        stellar_keys = {}
        if "stellar" in self.emitter_params:
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
            outputs["sfh_time_abs"] = self.cosmo.age(galaxy.redshift).to("Myr").value * Myr
            outputs["sfh_time_abs"] = outputs["sfh_time_abs"] - time

        if "lnu" in self.output_type:
            fluxes = spec.lnu
            outputs["lnu"] = copy.deepcopy(fluxes)

        if "photo_lnu" in self.output_type:
            fluxes = galaxy.stars.spectra[self.emission_model_key].get_photo_lnu(
                self.instrument.filters
            )
            fluxes = fluxes.photo_lnu
            outputs["photo_lnu"] = copy.deepcopy(fluxes)

        if "fnu" in self.output_type or "photo_fnu" in self.output_type:
            # Apply IGM and distance
            galaxy.get_observed_spectra(self.cosmo)

            if "photo_fnu" in self.output_type:
                fluxes = galaxy.stars.spectra[self.emission_model_key].get_photo_fnu(
                    self.instrument.filters
                )
                outputs["photo_fnu"] = fluxes.photo_fnu
                outputs["photo_wav"] = fluxes.filters.pivot_lams

                if "fnu" in self.output_type:
                    fluxes = galaxy.stars.spectra[self.emission_model_key]
                    outputs["fnu"] = copy.deepcopy(fluxes.fnu)
                    # print(np.sum(np.isnan(fluxes)), np.sum(fluxes == 0))
                    outputs["fnu_wav"] = copy.deepcopy(
                        galaxy.stars.spectra[self.emission_model_key].lam * (1 + galaxy.redshift)
                    )

        if self.out_flux_unit == "AB":

            def convert(f):
                return -2.5 * np.log10(f.to(Jy).value) + 8.9

            if "photo_fnu" in self.output_type:
                fluxes = convert(outputs["photo_fnu"])
                outputs["photo_fnu"] = copy.deepcopy(fluxes)
            if "fnu" in self.output_type:
                fluxes = convert(outputs["fnu"])
                # turn inf to nan
                fluxes[np.isinf(fluxes)] = 99
                outputs["fnu"] = fluxes
        elif self.out_flux_unit == "asinh":
            raise NotImplementedError(
                """asinh fluxes not implemented yet.
                Please use AB or Jy units."""
            )
        else:
            if "photo_fnu" in self.output_type:
                outputs["photo_fnu"] = outputs["photo_fnu"].to(self.out_flux_unit).value
            if "fnu" in self.output_type:
                outputs["fnu"] = outputs["fnu"].to(self.out_flux_unit).value

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
        if self.ignore_scatter:
            return fluxes, None

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


class GridFromSimOutput:
    """GridFromSimOutput class to create a Grid from simulation output."""

    def __init__(
        self,
        sim_output_path: str,
        feature_columns: List[str] = None,
        parameter_columns: List[str] = None,
    ):
        """Create a Grid from a simulation output file.

        This class reads a simulation output file and creates a Grid object
        from the data contained within it. The simulation output file is expected
        to be in a tabular format.

        Parameters
        ----------
        sim_output : Union[str, h5py.File]
            Path to the simulation output file or an open h5py.File object.
        """


def test_out_of_distribution(
    observed_photometry: np.ndarray,
    simulated_photometry: np.ndarray,
    sigma_threshold: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Identifies and removes samples from a dataset that are out-of-distribution.

    The function calculates the Mahalanobis distance for each simulated sample
    to the mean of the observed samples, accounting for the covariance
    between filters. Samples with a distance greater than the specified
    sigma_threshold are considered outliers and are removed.

    Args:
        observed_photometry (np.ndarray): The reference dataset, expected to
                                          have a shape of (N_filters, N_obs_samples).
        simulated_photometry (np.ndarray): The dataset to be filtered, expected
                                           to have a shape of (N_filters, N_sim_samples).
        sigma_threshold (float, optional): The number of standard deviations
                                           (in Mahalanobis distance) to use as the
                                           outlier threshold. Defaults to 5.0.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - A NumPy array containing the filtered simulated photometry.
              The shape will be (N_filters, N_inliers).
            - A 1D NumPy array containing the indices of the rows (samples) that were
              identified as outliers and removed from the original simulated_photometry.
    """
    if (
        observed_photometry.shape[1] == simulated_photometry.shape[1]
        and observed_photometry.shape[0] != simulated_photometry.shape[0]
    ):
        observed_photometry = (
            observed_photometry.T
        )  # Transpose if filters are rows and samples are columns
        simulated_photometry = simulated_photometry.T  # Ensure both are in the same shape

    if observed_photometry.shape[0] != simulated_photometry.shape[0]:
        raise ValueError(
            """observed_photometry and simulated_photometry must
            have the same number of filters (rows)."""
        )

    # Transpose data so that samples are rows and filters (features) are columns
    # Shape becomes (N_samples, N_filters)
    obs_data = observed_photometry.T
    sim_data = simulated_photometry.T

    # Calculate the mean vector and inverse covariance matrix of the observed data
    try:
        mean_obs = np.mean(obs_data, axis=0)
        cov_obs = np.cov(obs_data, rowvar=False)
        inv_cov_obs = inv(cov_obs)
    except np.linalg.LinAlgError:
        raise ValueError(
            """Could not compute the inverse covariance matrix of
            the observed_photometry. This can happen if the data
            is not full rank (e.g., filters are perfectly correlated)."""
        )

    # Calculate the Mahalanobis distance for each simulated sample
    # from the observed distribution
    mahalanobis_distances = np.zeros(sim_data.shape[0])
    for i in range(sim_data.shape[0]):
        delta = sim_data[i] - mean_obs
        # Manual calculation of Mahalanobis distance: sqrt(delta' * inv_cov * delta)
        mahalanobis_distances[i] = np.sqrt(delta @ inv_cov_obs @ delta.T)

    # Identify the indices of outliers
    outlier_indices = np.where(mahalanobis_distances > sigma_threshold)[0]
    inlier_indices = np.where(mahalanobis_distances <= sigma_threshold)[0]

    # Filter the original simulated_photometry array using the inlier indices
    # We filter the original array with shape (N_filters, N_samples)
    filtered_sim_photometry = simulated_photometry[:, inlier_indices]

    logger.info(f"Original number of samples: {simulated_photometry.shape[1]}")
    logger.info(f"Number of outliers removed ({sigma_threshold}-sigma): {len(outlier_indices)}")
    logger.info(f"Number of samples remaining: {filtered_sim_photometry.shape[1]}")

    return filtered_sim_photometry, outlier_indices


if __name__ == "__main__":
    logger.error("This is a module, not a script.")
