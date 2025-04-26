# ignore warnings for readability
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import synthesizer
from synthesizer.parametric import Galaxy
from synthesizer.emission_models.attenuation import Inoue14
from synthesizer.emission_models import PacmanEmission, TotalEmission, EmissionModel, IntrinsicEmission, StellarEmissionModel, STELLAR_MODELS, IncidentEmission
from synthesizer.emission_models.attenuation import PowerLaw
from synthesizer.emissions import plot_spectra
from synthesizer.emission_models.dust.emission import Greybody
from synthesizer.grid import Grid
from synthesizer.parametric import SFH, Stars, ZDist
from synthesizer import check_openmp
from synthesizer.instruments import Instrument, FilterCollection

from typing import Dict, Any, List, Tuple, Union, Optional, Type
from abc import ABC, abstractmethod
import copy
from scipy.stats import uniform, loguniform
from astropy.io import ascii
from unyt import unyt_array, unyt_quantity, erg, cm, s, Angstrom, um, Hz, m, nJy, K, Msun, Myr, yr, Unit, kg
from unyt.equivalencies import SpectralEquivalence
from astropy.cosmology import Planck18, Cosmology, z_at_value
import astropy.units as u
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from tqdm import tqdm
from ltu_ili_testing import (generate_emission_models, generate_sfh_basis, 
                            generate_metallicity_distribution, sed_grid_generator, 
                            generate_constant_R, GalaxyBasis, CombinedBasis,
                            calculate_muv)

warnings.filterwarnings('ignore')

# all medium and wide band filters for JWST NIRCam
filter_codes = [
    "JWST/NIRCam.F070W",
    "JWST/NIRCam.F090W",
    "JWST/NIRCam.F115W",
    "JWST/NIRCam.F140M",
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
    "JWST/NIRCam.F360M",
    "JWST/NIRCam.F410M",
    "JWST/NIRCam.F430M",
    "JWST/NIRCam.F444W",
    "JWST/NIRCam.F460M",
    "JWST/NIRCam.F480M",
]

# Consistent wavelength grid for both SPS grids and filters
new_wav = generate_constant_R(R=300)

filterset = FilterCollection(filter_codes, new_lam=new_wav)
instrument = Instrument('JWST', filters=filterset)

# --------------------------------------------------------------

# Configuration

# --------------------------------------------------------------

# Parameters which can be changed without rerunning the Pipeline: 
#   - redshift, masses, weights and instrument (if subset of filters is used)

max_redshift = 20 # gives maximum age of SFH at a given redshift
redshift = np.arange(6, 12, 1)   #={'prior': uniform, 'min': 5, 'max': 15, 'size': 10}
cosmo = Planck18 # cosmology to use for age calculations

# SFH parameters

masses = np.arange(5, 10, 1) # log10 stellar mass
weights = np.array([np.array([i, 1-i]) for i in np.arange(0, 1.1, 0.25)]) # weights for the two populations

# First Population Model

# ---------------------------------------------------------------

grid = Grid('bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps.hdf5',
            grid_dir='/home/tharvey/work/synthesizer_grids/',
            new_lam=new_wav)

# Metallicity 
Z_dist = ZDist.DeltaConstant
min_logZ = -3 # minimum log10 metallicity
max_logZ = 0.3 # maximum log10 metallicity
N_Z = 5 # number of metallicity distributions to generate
Z_dists = [Z_dist(log10metallicity=i) for i in np.linspace(min_logZ, max_logZ, N_Z)]


# SFH parameters
sfh_type = SFH.LogNormal
sfh_param_names = ['tau', 'peak_age']
tau_peak = np.array([    
        (0.1, -300),  # Recent burst, rising SFH
        (0.05, -300), # Very recent burst           
        (0.05,  lambda max_age: 0.1*max_age), # Very recent burst       
        (0.1,  lambda max_age: 0.15*max_age), # older burst, declining SFH
        (0.6,  lambda max_age:0.4*max_age), # Fairly flat, slight decline 
        (0.2,  lambda max_age: 0.2*max_age), # Fairly flat, slight decline 
        (0.2,  lambda max_age:0.6*max_age), # Quiescent 
        (0.3, -50), # slowly rising
        (0.14,  -50), # medium rising
        (0.6,  lambda max_age: 0.66 * max_age),
        (0.6,  lambda max_age: 0.8 * max_age),# early burst, now quiescent
        (2.5,  lambda max_age: -0.5*max_age), # early and constant
        (0.5,  100), # early and constant

])
sfh_param_units = [None, Myr]


sfh_models, redshifts = generate_sfh_basis(
    sfh_type=sfh_type,
    sfh_param_names=sfh_param_names,
    sfh_param_arrays=tau_peak,
    redshifts=redshift,
    max_redshift=max_redshift,
    cosmo=cosmo,
    sfh_param_units=sfh_param_units,
)

# Emission parameters

emission_model = TotalEmission(
    grid=grid,
    fesc=0.1,
    fesc_ly_alpha=0.1,
    dust_curve=PowerLaw(slope=-0.7),
    dust_emission_model=None,
)

# List of other varying or fixed parameters. Either a distribution to pull from or a list.
# Can be any parameter which can be property of emitter or galaxy and processed by the emission model.
galaxy_params={
    'tau_v': {
        'prior': loguniform, 
        'loc': 0.0,
        'scale': 1.0,
        'a': 1e-3, # minimum value
        'b': 1.0, # maximum value
        'size': 10, # number of samples
        'units': None,
    }
}

popII_basis = GalaxyBasis(
    model_name='Pop II',
    redshifts=redshifts,
    grid=grid,
    emission_model=emission_model,
    sfhs=sfh_models,
    cosmo=cosmo,
    instrument=instrument,
    metal_dists=Z_dists,
    galaxy_params=galaxy_params,
    redshift_dependent_sfh=True,
    params_to_ignore=['max_age'], # This is dependent on the redshift and should not be included in the basis
)


# Second Population Model
# ---------------------------------------------------------------

popIII_grid = Grid('yggdrasil-1.3.3-PopIII_salpeter-10,1,500',
                    grid_dir='/home/tharvey/work/synthesizer_grids/',
                    read_lines=False, new_lam=new_wav)


# Pop III parameters
sfh_array = np.array([(0, 10), (0, 20)]) #(0, 30), (10, 20), (10, 30), (20, 30), (30, 50), (50, 70), (70, 100)]) 
sfh_param_units = [Myr, Myr]
popIII_sfhs, redshifts = generate_sfh_basis(
    sfh_type=SFH.Constant,
    sfh_param_names=['min_age', 'max_age'],
    sfh_param_arrays=sfh_array,
    redshifts=redshift,
    max_redshift=max_redshift,
    cosmo=cosmo,
    sfh_param_units=sfh_param_units,
    calculate_min_age=True,
)

popIII_metal_dists = ZDist.DeltaConstant(metallicity=0)

if 'incident' not in grid.available_spectra:
    # Need a custom emission model for the Pop III grid if loading the nebular spectra. 
    popIII_emission_model = EmissionModel(
        label = 'Pop III',
        extract=popIII_grid.available_spectra[0],
        grid=popIII_grid,
        emitter='stellar'
    )
else:
    popIII_emission_model = IncidentEmission(
        grid=popIII_grid,
    )

popIII_basis = GalaxyBasis(
    model_name='Pop III',
    redshifts=redshifts,
    grid=popIII_grid,
    emission_model=popIII_emission_model,
    sfhs=popIII_sfhs,
    cosmo=cosmo,
    instrument=instrument,
    metal_dists=popIII_metal_dists,
    redshift_dependent_sfh=True,
)

# --------------------------------------------------------------
# Combine the two population models

combined_basis = CombinedBasis(
    bases=[popII_basis, popIII_basis],
    total_stellar_masses=unyt_array(10**masses, units=Msun),
    base_emission_model_keys=['total', popIII_emission_model.label],
    combination_weights=weights,
    redshifts=redshifts,
    out_name='combined_basis',
    out_dir='/home/tharvey/work/output/',
)

# Passing in extra analysis function to pipeline to calculate mUV. Any funciton could be passed in. 
combined_basis.process_bases(overwrite=False, mUV=(calculate_muv, cosmo))

combined_basis.create_grid(overwrite=True)


#for i in range(0, 50000, 500):
#    combined_basis.plot_galaxy_from_grid(i, show=False, save=True)


