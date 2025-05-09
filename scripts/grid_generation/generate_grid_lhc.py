# ignore warnings for readability
import warnings
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import synthesizer
from synthesizer.parametric import Galaxy
from synthesizer.emission_models.attenuation import Inoue14, Madau96
from synthesizer.emission_models import PacmanEmission, TotalEmission, EmissionModel, IntrinsicEmission, StellarEmissionModel, STELLAR_MODELS, IncidentEmission
from synthesizer.emission_models.attenuation import PowerLaw, Calzetti2000
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
                            generate_metallicity_distribution, 
                            generate_constant_R, GalaxyBasis, CombinedBasis,
                            calculate_muv, draw_from_hypercube)
'''try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

except ImportError:
    rank, size = 0, 1'''

# Issues
# Minimum Age longer than maximum age!

# Filters
# ---------------------------------------------------------------
# all medium and wide band filters for JWST NIRCam
filter_codes = [
    "JWST/NIRCam.F070W", "JWST/NIRCam.F090W", "JWST/NIRCam.F115W", "JWST/NIRCam.F140M",
    "JWST/NIRCam.F150W", "JWST/NIRCam.F162M", "JWST/NIRCam.F182M", "JWST/NIRCam.F200W",
    "JWST/NIRCam.F210M", "JWST/NIRCam.F250M", "JWST/NIRCam.F277W", "JWST/NIRCam.F300M",
    "JWST/NIRCam.F335M", "JWST/NIRCam.F356W", "JWST/NIRCam.F360M", "JWST/NIRCam.F410M",
    "JWST/NIRCam.F430M", "JWST/NIRCam.F444W", "JWST/NIRCam.F460M", "JWST/NIRCam.F480M",
]
instrument = 'JWST'

# Consistent wavelength grid for both SPS grids and filters
new_wav = generate_constant_R(R=300)

path = f'{os.path.dirname(__file__)}/filters/{instrument}.hdf5'

if 'cosma' in path:
    computer = 'cosma'
else:
    computer = 'linux-desktop'


if os.path.exists(path):
    filterset = FilterCollection(path=path, new_lam=new_wav)
else:
    filterset = FilterCollection(filter_codes=filter_codes, new_lam=new_wav)

instrument = Instrument(instrument, filters=filterset)


if computer == 'cosma':
    grid_dir = '/cosma7/data/dp276/dc-harv3/work/grids/'
    out_dir =  '/cosma7/data/dp276/dc-harv3/work/sbi/output/'

elif computer == 'linux-desktop':
    grid_dir = '/home/tharvey/work/synthesizer_grids/'
    out_dir = '/home/tharvey/work/output/'


try:
    n_proc = int(sys.argv[1])
except:
    n_proc = 6

# params

Nmodels = 1e3
redshift = (5, 12)
masses = (5, 11)
weights = (0, 1)
max_redshift = 20 # gives maximum age of SFH at a given redshift
cosmo = Planck18 # cosmology to use for age calculations

# ---------------------------------------------------------------
# Pop II

tau_v = (0.0, 2.0)
log_zmet = (-3, 0.3)

# SFH
sfh_type = SFH.LogNormal
tau = (0.05, 2.5)
peak_age = (-1, 1) # normalized to maximum age of the universe at that redshift
sfh_param_units = [None, None]

# ---------------------------------------------------------------
# Pop III

min_age_popIII = (0.00, 30) # Myr
sfh_timescale_popIII = (0.01, 30) # Length of Time - e.g. min_age_popIII + popIII_sfh_timescale

# ---------------------------------------------------------------

# Generate the grid. Could also seperate hyper-parameters for each model. 

full_params = {
    'redshift': redshift,
    'masses': masses,
    'weights': weights,
    'tau_v': tau_v,
    'log_zmet': log_zmet,
    'tau': tau,
    'peak_age': peak_age,
    'min_age_popIII': min_age_popIII,
    'sfh_timescale_popIII': sfh_timescale_popIII
}

# Generate the grid

param_grid = draw_from_hypercube(Nmodels, full_params, rng=42)

# Unpack the parameters

all_param_dict = {}
for i, key in enumerate(full_params.keys()):
    all_param_dict[key] = param_grid[:, i]


grid = Grid('bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps.hdf5',
            grid_dir=grid_dir,
            new_lam=new_wav)

# Metallicity 
Z_dists = [ZDist.DeltaConstant(log10metallicity=log_z) for log_z in all_param_dict['log_zmet']]

redshifts = np.array(all_param_dict['redshift'])

# Pop II SFH
sfh_param_arrays = np.vstack((all_param_dict['tau'], all_param_dict['peak_age'])).T
sfh_models, _ = generate_sfh_basis(
    sfh_type=sfh_type,
    sfh_param_names=['tau', 'peak_age_norm'],
    sfh_param_arrays=sfh_param_arrays,
    redshifts=redshifts,
    max_redshift=max_redshift,
    cosmo=cosmo,
    sfh_param_units=sfh_param_units,
    iterate_redshifts=False,
    calculate_min_age=False,
)

# Emission parameters
emission_model = TotalEmission(
    grid=grid,
    fesc=0.1,
    fesc_ly_alpha=0.1,
    dust_curve=Calzetti2000(), 
    dust_emission_model=None,
)

# List of other varying or fixed parameters. Either a distribution to pull from or a list.
# Can be any parameter which can be property of emitter or galaxy and processed by the emission model.
galaxy_params={
    'tau_v': all_param_dict['tau_v'],
}

sfh_name = str(sfh_type).split('.')[-1].split("'")[0]

popII_basis = GalaxyBasis(
    model_name=f'Pop_II_{sfh_name}_SFH_{redshift[0]}_z_{redshift[1]}_logN_{np.log10(Nmodels):.1f}',
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
    build_grid=False,
)


# Second Population Model
# ---------------------------------------------------------------

popIII_grid = Grid('yggdrasil-1.3.3-PopIII_salpeter-10,1,500',
                    grid_dir=grid_dir,
                    read_lines=False, new_lam=new_wav)


# Pop III parameters

# SFH
sfh_array = np.vstack((all_param_dict['min_age_popIII'], all_param_dict['sfh_timescale_popIII'])).T

sfh_param_units = [Myr, Myr]
popIII_sfhs, _ = generate_sfh_basis(
    sfh_type=SFH.Constant,
    sfh_param_names=['min_age', 'sfh_timescale'],
    sfh_param_arrays=sfh_array,
    redshifts=redshifts,
    max_redshift=max_redshift,
    cosmo=cosmo,
    sfh_param_units=sfh_param_units,
    calculate_min_age=False,
    iterate_redshifts=False,
)

popIII_metal_dists = ZDist.DeltaConstant(metallicity=0)

if 'incident' not in grid.available_spectra:
    # Need a custom emission model for the Pop III grid if loading the nebular spectra. 
    popIII_emission_model = EmissionModel(
        label = 'Pop_III',
        extract=popIII_grid.available_spectra[0],
        grid=popIII_grid,
        emitter='stellar'
    )
else:
    popIII_emission_model = IncidentEmission(
        grid=popIII_grid,
    )

popIII_basis = GalaxyBasis(
    model_name='Pop_III',
    redshifts=redshifts,
    grid=popIII_grid,
    emission_model=popIII_emission_model,
    sfhs=popIII_sfhs,
    cosmo=cosmo,
    instrument=instrument,
    metal_dists=popIII_metal_dists,
    redshift_dependent_sfh=True,
    build_grid=False,
)

# --------------------------------------------------------------
# Combine the two population models

weights = np.array([np.array([i, 1-i]) for i in all_param_dict['weights']])


combined_basis = CombinedBasis(
    bases=[popII_basis, popIII_basis],
    total_stellar_masses=unyt_array(10**all_param_dict['masses'], units=Msun),
    base_emission_model_keys=['total', popIII_emission_model.label],
    combination_weights=weights,
    redshifts=redshifts,
    out_name='LHC_sampled',
    out_dir=out_dir,
    draw_parameter_combinations=False, # Since we have already drawn the parameters, we don't need to combine them again.
)

# Passing in extra analysis function to pipeline to calculate mUV. Any funciton could be passed in. 
combined_basis.process_bases(overwrite=False, mUV=(calculate_muv, cosmo), n_proc=n_proc)

combined_basis.create_grid(overwrite=True, out_name='combined_basis_LHC')
