# ignore warnings for readability
import os
import sys

import numpy as np
from astropy.cosmology import Planck18
from synthesizer.emission_models import PacmanEmission
from synthesizer.emission_models.attenuation import (
    Calzetti2000,
)  # noqa
from synthesizer.emission_models.dust.emission import Blackbody
from synthesizer.emission_models.stellar.models import IncidentEmission, IntrinsicEmission
from synthesizer.grid import Grid
from synthesizer.instruments import FilterCollection, Instrument
from synthesizer.parametric import SFH, ZDist
from unyt import K, Myr

from synference import (
    GalaxyBasis,
    calculate_mass_weighted_age,
    calculate_muv,
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
filter_codes = [
    "Paranal/VISTA.Z",
    "Paranal/VISTA.Y",
    "Paranal/VISTA.J",
    "Paranal/VISTA.H",
    "Paranal/VISTA.Ks",
    "Subaru/HSC.g",
    "Subaru/HSC.r",
    "Subaru/HSC.i",
    "Subaru/HSC.z",
    "Subaru/HSC.Y",
    "CFHT/MegaCam.u",
    "CFHT/MegaCam.g",
    "CFHT/MegaCam.r",
    "CFHT/MegaCam.i",
    "CFHT/MegaCam.z",
    "Euclid/VIS.vis",
    "Euclid/NISP.Y",
    "Euclid/NISP.J",
    "Euclid/NISP.H",
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
    "JWST/MIRI.F1000W",
    "JWST/MIRI.F1130W",
    "JWST/MIRI.F1280W",
    "JWST/MIRI.F1500W",
    "JWST/MIRI.F1800W",
    "JWST/MIRI.F2100W",
    "JWST/MIRI.F2550W",
    "Spitzer/IRAC.I1",
    "Spitzer/IRAC.I2",
    "Spitzer/IRAC.I3",
    "Spitzer/IRAC.I4",
]
instrument = "GENERAL_SURVEY"

path = f"{os.path.dirname(__file__)}/filters/{instrument}.hdf5"

if os.path.exists(path):
    print(f"Loading filters from {path}")
    filterset = FilterCollection(path=path)
else:
    filterset = FilterCollection(filter_codes=filter_codes)


# Consistent wavelength grid for both SPS grids and filters
new_wav = generate_constant_R(R=300, auto_start_stop=True, filterset=filterset, max_redshift=20)

filterset.resample_filters(new_lam=new_wav)

instrument = Instrument(instrument, filters=filterset)

# Check for SYNTHESIZER_GRID_DIR environment variable

grid_dir = os.environ["SYNTHESIZER_GRID_DIR"]

# path for this file

dir_path = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(dir_path))), "grids/")

try:
    n_proc = int(sys.argv[1])
except Exception:
    n_proc = 6

print(f"Number of processes/task: {n_proc}")

from mpi4py import MPI

# params

overwrite = True
Nmodels = 25_000  # 00
redshift = (0.001, 14)
masses = (4, 12)  # log10 of stellar mass in solar masses
max_redshift = 20  # gives maximum age of SFH at a given redshift
cosmo = Planck18  # cosmology to use for age calculations


# ---------------------------------------------------------------
# Pop II

zmet = 0

# ---------------------------------------------------------------

# Generate the grid. Could also seperate hyper-parameters for each model.

param_prior_ranges = {
    "redshift": redshift,
    "log_masses": masses,
    "log_stellar_age": (4.0, 9.0),  # in log10 of years
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
    "yggdrasil-1.3.3-PopIII_salpeter-10,1,500",
    grid_dir=grid_dir,
    new_lam=new_wav,
)

# Metallicity
Z_dists = [ZDist.DeltaConstant(metallicity=zmet) for i in range(Nmodels)]

# Create ages for SFH models
sfh_models = [10 ** log_age * Myr for log_age in all_param_dict["log_stellar_age"]]


# Emission parameters
emission_model = IncidentEmission(
    grid=grid,
)

name = f"Yggdrasil_Chab_Burst_SFH_{redshift[0]}_z_{redshift[1]}_logN_{np.log10(Nmodels):.1f}_v1"  # noqa: E501
#name = "test_sbi"

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
    galaxy_params=None,
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
    emission_model_key=emission_model.label,
    verbose=False,
    batch_size=50_000,
    mUV=(calculate_muv, cosmo),  # Calculate mUV for the mock catalogue.
    mwa=calculate_mass_weighted_age,  # Calculate MWA for the mock catalogue.
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
