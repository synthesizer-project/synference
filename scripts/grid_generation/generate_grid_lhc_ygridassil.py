# ignore warnings for readability
import numpy as np
import os
import sys
from synthesizer.emission_models import TotalEmission, EmissionModel, IncidentEmission
from synthesizer.emission_models.attenuation import Calzetti2000, ParametricLi08 #noqa
from synthesizer.grid import Grid
from synthesizer.parametric import SFH, ZDist
from synthesizer.instruments import Instrument, FilterCollection

from unyt import unyt_array, Msun, Myr
from astropy.cosmology import Planck18
from ltu_ili_testing import (generate_sfh_basis, 
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

path = f'{os.path.dirname(__file__)}/filters/{instrument}.hdf5'

if 'cosma' in path:
    computer = 'cosma'
else:
    computer = 'linux-desktop'


if os.path.exists(path):
    filterset = FilterCollection(path=path)
else:
    filterset = FilterCollection(filter_codes=filter_codes)



# Consistent wavelength grid for both SPS grids and filters
new_wav = generate_constant_R(R=300, auto_start_stop=True, 
                            filterset=filterset, max_redshift=15)

filterset.resample_filters(new_lam=new_wav)

instrument = Instrument(instrument, filters=filterset)


if computer == 'cosma':
    grid_dir = '/cosma7/data/dp276/dc-harv3/work/grids/'
    out_dir =  '/cosma7/data/dp276/dc-harv3/work/sbi/output/'

elif computer == 'linux-desktop':
    grid_dir = '/home/tharvey/work/synthesizer_grids/'
    out_dir = '/home/tharvey/work/output/'


try:
    n_proc = int(sys.argv[1])
except Exception:
    n_proc = 6

# params

Nmodels = 10_000
redshift = (5, 12)
masses = (4, 9)
max_redshift = 20 # gives maximum age of SFH at a given redshift
cosmo = Planck18 # cosmology to use for age calculations

# ---------------------------------------------------------------
# Pop II

log_zmet = 0 # max of grid (e.g. 0.04)

# SFH
sfh_type = SFH.Constant

min_age= (0.00, 30) # Myr
sfh_timescale = (0.01, 30) # Length of Time - e.g. min_age_popIII + popIII_sfh_timescale
sfh_param_units = [Myr, Myr]


# ---------------------------------------------------------------

# Generate the grid. Could also seperate hyper-parameters for each model. 

full_params = {
    'redshift': redshift,
    'masses': masses,
    'sfh_timescale': sfh_timescale,
    'min_age': min_age,
}

# Generate the grid

param_grid = draw_from_hypercube(Nmodels, full_params, rng=42)

# Unpack the parameters

all_param_dict = {}
for i, key in enumerate(full_params.keys()):
    all_param_dict[key] = param_grid[:, i]
# yggdrasil-1.3.3-PopIII_salpeter-10,1,500
# yggdrasil-1.3.3-POPIII-fcov_0.5_salpeter-10,1,500.hdf5
# yggdrasil-1.3.3-POPIII-fcov_1_salpeter-10,1,500.hdf5

grid = Grid('yggdrasil-1.3.3-POPIII-fcov_1_salpeter-10,1,500',
                    grid_dir=grid_dir,
                    read_lines=False, new_lam=new_wav)


# Pop III parameters

# SFH
sfh_array = np.vstack((all_param_dict['min_age'], all_param_dict['sfh_timescale'])).T
redshifts = np.array(all_param_dict['redshift'])

sfh_models, _ = generate_sfh_basis(
    sfh_type=sfh_type,
    sfh_param_names=['min_age', 'sfh_timescale'],
    sfh_param_arrays=sfh_array,
    redshifts=redshifts,
    max_redshift=max_redshift,
    cosmo=cosmo,
    sfh_param_units=sfh_param_units,
    calculate_min_age=False,
    iterate_redshifts=False,
)

# Metallicity 
Z_dists = ZDist.DeltaConstant(metallicity=0)

# Emission parameters
if 'incident' not in grid.available_spectra:
    # Need a custom emission model for the Pop III grid if loading the nebular spectra. 
    emission_model = EmissionModel(
        label = 'Pop_III',
        extract=grid.available_spectra[0],
        grid=grid,
        emitter='stellar'
    )
else:
    emission_model = IncidentEmission(
        grid=grid,
    )

# List of other varying or fixed parameters. Either a distribution to pull from or a list.
# Can be any parameter which can be property of emitter or galaxy and processed by the emission model.
galaxy_params = {}

name = f'{grid.grid_name}_{redshift[0]}_z_{redshift[1]}_logN_{np.log10(Nmodels):.1f}_v1'



popII_basis = GalaxyBasis(
    model_name=f'sps_{name}',
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

combined_basis = CombinedBasis(
    bases=[popII_basis],
    total_stellar_masses=unyt_array(10**all_param_dict['masses'], units=Msun),
    base_emission_model_keys=['Pop_III'],
    combination_weights=None,
    redshifts=redshifts,
    out_name=f'grid_{name}',
    out_dir=out_dir,
    draw_parameter_combinations=False, # Since we have already drawn the parameters, we don't need to combine them again.
)

# Passing in extra analysis function to pipeline to calculate mUV. Any funciton could be passed in. 
combined_basis.process_bases(overwrite=True, mUV=(calculate_muv, cosmo), n_proc=n_proc, verbose=False)

# Create grid - kinda overkill for a single case, but it does work.
combined_basis.create_grid(overwrite=True)
