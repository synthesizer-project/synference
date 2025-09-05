# ignore warnings for readability
import os
import sys

import numpy as np
from astropy.cosmology import Planck18
from synthesizer.emission_models import TotalEmission
from synthesizer.emission_models.attenuation import (
    Calzetti2000,
)  # noqa
from synthesizer.grid import Grid
from synthesizer.instruments import FilterCollection, Instrument
from synthesizer.parametric import SFH, ZDist
from unyt import Myr

from sbifitter import (
    GalaxyBasis,
    calculate_muv,
    calculate_mass_weighted_age,
    draw_from_hypercube,
    generate_constant_R,
    generate_sfh_basis,
)

"""try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

except ImportError:
    rank, size = 0, 1"""

# Issues
# Minimum Age longer than maximum age!

# Filters
# ---------------------------------------------------------------
# all medium and wide band filters for JWST NIRCam
filter_codes = [
    "HST/ACS_WFC.F435W",
    "HST/ACS_WFC.F475W",
    "HST/ACS_WFC.F606W",
    "JWST/NIRCam.F070W",
    "HST/ACS_WFC.F775W",
    "HST/ACS_WFC.F814W",
    "HST/ACS_WFC.F850LP",
    "JWST/NIRCam.F090W",
    "HST/WFC3_IR.F105W",
    "HST/WFC3_IR.F110W",
    "JWST/NIRCam.F115W",
    "HST/WFC3_IR.F125W",
    "JWST/NIRCam.F140M",
    "HST/WFC3_IR.F140W",
    "JWST/NIRCam.F150W",
    "HST/WFC3_IR.F160W",
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
    "JWST/MIRI.F560W",
    "JWST/MIRI.F770W",
]
instrument = "HST+JWST"

path = f"{os.path.dirname(__file__)}/filters/{instrument}.hdf5"

if os.path.exists(path):
    print(f"Loading filters from {path}")
    filterset = FilterCollection(path=path)
else:
    filterset = FilterCollection(filter_codes=filter_codes)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    filterset.write_filters(path)


if "cosma" in path:
    computer = "cosma"
else:
    computer = "linux-desktop"


# Consistent wavelength grid for both SPS grids and filters
new_wav = generate_constant_R(
    R=300, auto_start_stop=True, filterset=filterset, max_redshift=15
)

filterset.resample_filters(new_lam=new_wav)

instrument = Instrument(instrument, filters=filterset)

# Check for SYNTHESIZER_GRID_DIR environment variable
grid_dir = os.environ["SYNTHESIZER_GRID_DIR"]


# path for this file

dir_path = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(os.path.dirname(os.path.dirname(dir_path)), "grids/")

try:
    n_proc = int(sys.argv[1])
except Exception:
    n_proc = 6

# params

overwrite = True
Nmodels = 100  # 00
redshift = (0.001, 12)
masses = (4, 12)  # log10 of stellar mass in solar masses
max_redshift = 20  # gives maximum age of SFH at a given redshift
cosmo = Planck18  # cosmology to use for age calculations
fesc = 0.0  # escape fraction of ionizing photons
fesc_ly_alpha = 0.0  # escape fraction of Ly-alpha photons
dust_emission = None  # No dust emission model for this grid, but can be added later.
dust_curve = Calzetti2000()  # Dust attenuation curve to use for the grid.
# ---------------------------------------------------------------
# Pop II

tau_v = (0.0, 3.0)
log_zmet = (-4, -1.39)  # max of grid (e.g. 0.04)

# SFH
sfh_type = SFH.LogNormal
tau = (0.05, 2.5)
peak_age = (
    0,
    0.99,
)  # normalized to maximum age of the universe at that redshift.

# ---------------------------------------------------------------

# Generate the grid. Could also seperate hyper-parameters for each model.

param_prior_ranges = {
    "redshift": redshift,
    "log_masses": masses,
    "tau_v": tau_v,
    "log_zmet": log_zmet,
    "tau": tau,
    "peak_age": peak_age,
}

# Draw samples from Latin Hypercube
all_param_dict = draw_from_hypercube(
    param_prior_ranges,
    Nmodels,
    rng=42,
)

# Get samples from the LHC draw dict
redshifts = all_param_dict["redshift"]
masses = all_param_dict["log_masses"]

# Load Synthesizer SPS grid
grid = Grid(
    "bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps.hdf5",
    grid_dir=grid_dir,
    new_lam=new_wav,
)

# Metallicity
Z_dists = [
    ZDist.DeltaConstant(log10metallicity=log_z) for log_z in all_param_dict["log_zmet"]
]

# Create LogNormal SFH from parameters.
# These
sfh_models, _ = generate_sfh_basis(
    sfh_type=sfh_type,
    sfh_param_names=["tau", "peak_age_norm"],
    sfh_param_arrays=(all_param_dict["tau"], all_param_dict["peak_age"]),
    redshifts=redshifts,
    max_redshift=max_redshift,
    cosmo=cosmo,
)


# Emission parameters
emission_model = TotalEmission(
    grid=grid,
    fesc=fesc,
    fesc_ly_alpha=fesc_ly_alpha,
    dust_curve=dust_curve,  # ParametricLi08(model='SMC'),
    dust_emission_model=dust_emission,
)

# List of other varying or fixed parameters. Either a distribution to pull from or a list.
# Can be any parameter which can be property of emitter or galaxy
# and processed by the emission model.
galaxy_params = {
    "tau_v": all_param_dict["tau_v"],
}

sfh_name = str(sfh_type).split(".")[-1].split("'")[0]

name = f"BPASS_Chab_{sfh_name}_SFH_{redshift[0]}_z_{redshift[1]}_logN_{np.log10(Nmodels):.1f}_Calzetti_v2"  # noqa: E501

basis = GalaxyBasis(
    model_name=f"sps_{name}",
    redshifts=redshifts,
    grid=grid,
    emission_model=emission_model,
    sfhs=sfh_models,
    log_stellar_masses=masses,
    cosmo=cosmo,
    instrument=instrument,
    metal_dists=Z_dists,
    galaxy_params=galaxy_params,
    params_to_ignore=[
        "max_age"
    ],  # This is dependent on the redshift and should not be included in the basis
)


def z_to_max_age(params, max_redshift=20):
    """Convert redshift to maximum age of the SFH at that redshift."""
    z = params["redshift"]
    from astropy.cosmology import Planck18 as cosmo

    age = cosmo.age(z) - cosmo.age(max_redshift)
    age = age.to_value("Myr") * Myr
    return age


# This is the simple way-
# it runs the following three steps for you.

basis.create_mock_cat(
    out_name=f"grid_{name}",
    out_dir=out_dir,
    overwrite=overwrite,
    n_proc=n_proc,
    verbose=False,
    batch_size=50_000,
    mUV=(calculate_muv, cosmo),  # Calculate mUV for the mock catalogue.
    mwa=calculate_mass_weighted_age,  # Calculate MWA for the mock catalogue.
    parameter_transforms_to_save={
        "max_age": z_to_max_age
    },  # Save function to calculate the maximum age of the SFH at that redshift.
)

"""
# This is the complex way
combined_basis = CombinedBasis(
    bases=[basis],
    total_stellar_masses=unyt_array(10 ** all_param_dict["masses"], units=Msun),
    base_emission_model_keys=["total"],
    combination_weights=None,
    redshifts=redshifts,
    out_name=f"grid_{name}",
    out_dir=out_dir,
    draw_parameter_combinations=False,
)

# Passing in extra analysis function to pipeline to calculate mUV.
combined_basis.process_bases(
    overwrite=False, mUV=(calculate_muv, cosmo), n_proc=n_proc, verbose=False
)

# Create grid - kinda overkill for a single case, but it does work.
combined_basis.create_grid(overwrite=True)
"""
