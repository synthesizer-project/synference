# ignore warnings for readability
"""Prospector-Beta-like library generation for synference.

Redshift, log stellar mass, log stellar metallicity, and the Continuity-SFH
``logsfr_ratios``/age bins are drawn jointly from
``synference.sample_prospector_beta_prior``, which replicates the
Prospector-Beta conditional priors (Wang, Leja, et al. 2023, ApJL 944, L58):
a redshift-evolving stellar mass function, a dynamic p(z) ~ N(z) dV/dz, the
Gallazzi et al. (2005) mass-metallicity relation, and a mass/redshift-shifted
Student-t Continuity-SFH prior tied to the Behroozi et al. (2019) cosmic
SFRD. See ``synference/src/synference/priors_beta.py`` for the implementation
and ``.claude/plans/`` for design notes.

Dust (Av/slope/dust_bump_amplitude/fesc_lya) is drawn independently via
``draw_from_hypercube``, matching Prospector-Beta's own (non-conditional)
dust priors as closely as the single-component Pacman/Calzetti dust curve
in this local ``synthesizer`` checkout allows (no birth-cloud/diffuse split).
"""

import copy  # noqa
import os
import sys

import numpy as np
from astropy.cosmology import Planck18
from synthesizer.emission_models.attenuation import (
    Calzetti2000,
)  # noqa
from synthesizer.emission_models.generators.dust import Greybody
from synthesizer.emission_models.stellar.pacman_model import (
    PacmanEmission,
)  # noqa
from synthesizer.grid import Grid
from synthesizer.instruments import FilterCollection, Instrument
from synthesizer.parametric import SFH, ZDist
from tqdm import tqdm
from unyt import K

try:
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.Get_rank()  # Get the rank of the current process
    size = MPI.COMM_WORLD.Get_size()  # Get the total number of processes
except ImportError:
    rank, size = 0, 1

print(f"Rank {rank} with {size} processes available.")

from synference import (
    GalaxyBasis,
    calculate_beta,
    calculate_burstiness,
    calculate_colour,
    calculate_d4000,
    calculate_line_ew,
    calculate_line_flux,
    calculate_mass_weighted_age,
    calculate_muv,
    calculate_Ndot_ion,
    calculate_sfh_quantile,
    calculate_surviving_mass,
    calculate_xi_ion0,
    draw_from_hypercube,
    generate_constant_R,
    sample_prospector_beta_prior,
)

# Filters
# ---------------------------------------------------------------
# all medium and wide band filters for JWST NIRCam
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

dir_path = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(dir_path))), "libraries/")

try:
    n_proc = int(sys.argv[1])
except Exception:
    n_proc = 6

print(f"Number of processes/task: {n_proc}")

av_to_tau_v = 1.086  # conversion factor from Av to tau_v for the dust attenuation curve
overwrite = False  # whether to overwrite existing grids
Nmodels = 100_000  # number of models to generate
grid_name = "BPASS"  # name for the grid
cat_type = "spectra"  # spectra or photometry
cosmo = Planck18  # cosmology to use for age calculations
emission_key = "total"  # 'attenuated' if no dust emission or 'emergent' if fesc > 0
seed = 42  # Seed for reproducibility

# --- Prospector-Beta prior hyperparameters (Wang, Leja, et al. 2023) ---
nbins_sfh = 7  # number of Continuity-SFH bins
zred_range = (1e-3, 15.0)
mass_range = (7.0, 12.5)  # log10(M*/Msun)
met_range = (-1.98, 0.19)  # log10(Z/Zsun), Gallazzi+05 grid limits
logsfr_ratio_range = (-5.0, 5.0)
logsfr_ratio_tscale = 0.3
const_phi = True  # clamp the SMF to its [0.2, 3.0] validity range (matches TemplateLibrary["beta"])

# --- Independent (non-conditional) dust priors ---
# Prospector-Beta's dust2/dust_index map onto this repo's existing Av/slope
# Calzetti-attenuation parametrization; dust1/dust_ratio (birth-cloud vs.
# diffuse split) are not replicated since PacmanEmission only exposes a
# single tau_v component.
logAv = (-3, 0.7)  # log-uniform between 0.001 and ~5.0 magnitudes
slope_range = (-2.0, 0.5)  # Prospector-Beta's dust_index prior (TopHat)
fesc_lya_range = (0.0, 1.0)
dust_bump_amplitude_range = (0.0, 5.0)

name = (
    f"{grid_name}_Chab_beta_SFH_{zred_range[0]}_z_{zred_range[1]}_"
    f"logN_{np.log10(Nmodels):.1f}_Calzetti_v1"
)
print(f"{out_dir}/library_{name}.hdf5")
if os.path.exists(f"{out_dir}/library_{name}.hdf5") and not overwrite:
    print(f"Library {name} already exists, skipping.")
    sys.exit(0)

mask = np.zeros(Nmodels, dtype=bool)
galaxies_per_node = Nmodels // size
start_idx = rank * galaxies_per_node
end_idx = start_idx + galaxies_per_node
if rank == size - 1:  # Last node gets the remainder
    end_idx = Nmodels
mask[start_idx:end_idx] = True

if int(sys.argv[2]) == 1:
    mask = np.ones(Nmodels, dtype=bool)

batch_size = np.sum(mask) + 1

grid_dict = {
    "FSPS": "fsps-3.2-mist-miles_chabrier03-0.1,300_cloudy-c23.01-sps",
    "BPASS": "bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps",
}

# --- Draw the Prospector-Beta-conditioned parameters ---
# Every rank draws the same full-size, seeded hypercube for reproducibility,
# but `mask` restricts the expensive per-galaxy mass/met/SFH quantile
# transform to this rank's slice (same idiom as the SFH masking in
# final_library_generation_multinode.py).
print("Drawing samples from the Prospector-Beta prior.")
beta_params = sample_prospector_beta_prior(
    Nmodels,
    nbins_sfh=nbins_sfh,
    zred_range=zred_range,
    mass_range=mass_range,
    met_range=met_range,
    logsfr_ratio_range=logsfr_ratio_range,
    logsfr_ratio_tscale=logsfr_ratio_tscale,
    const_phi=const_phi,
    cosmo=cosmo,
    rng=seed,
    mask=mask,
    verbose=rank == 0,
)
redshifts = beta_params["redshift"]

# --- Independent dust priors (plain joint Latin Hypercube) ---
print("Drawing dust parameters from Latin Hypercube.")
dust_params = draw_from_hypercube(
    {
        "log_Av": logAv,
        "slope": slope_range,
        "fesc_lya": fesc_lya_range,
        "dust_bump_amplitude": dust_bump_amplitude_range,
    },
    Nmodels,
    rng=seed,
    unlog_keys=["log_Av"],
)

# Create the grid
grid = Grid(
    grid_dict[grid_name],
    grid_dir=grid_dir,
    new_lam=new_wav,
)
print(grid.available_lines)

# Metallicity: one DeltaConstant ZDist per galaxy, from the MZR-conditioned draw.
Z_dists = [
    ZDist.DeltaConstant(log10metallicity=log_z)
    for log_z in tqdm(beta_params["log_zmet"], desc="Creating ZDist", disable=rank != 0)
]

# SFH: Continuity SFH built directly from the mass/redshift-conditioned
# logsfr_ratios and redshift-dependent age bins (no init_from_prior needed,
# since the ratios have already been drawn by sample_prospector_beta_prior).
sfh_models = []
for i in tqdm(range(Nmodels), desc="Building Continuity SFHs", disable=rank != 0):
    if mask[i]:
        sfh_models.append(
            SFH.Continuity(
                logsfr_ratios=beta_params["logsfr_ratios"][i],
                agebins=beta_params["agebins"][i],
            )
        )
    else:
        sfh_models.append(None)

dust_emission = Greybody(temperature=40 * K, emissivity=1.5)

# Essentially CF00 with explicit fesc and fesc_ly_alpha parameters.
print("Creating emission model.")
emission_model = PacmanEmission(
    grid=grid,
    tau_v="tau_v",
    dust_curve=Calzetti2000(slope="slope", ampl="dust_bump_amplitude"),
    dust_emission=dust_emission,
    fesc=0.0,  # escape fraction of ionizing photons
    fesc_ly_alpha="fesc_lya",  # escape fraction of Lyman-alpha photons
)

galaxy_params = {
    "tau_v": dust_params["Av"] / av_to_tau_v,
    "slope": dust_params["slope"],
    "fesc_lya": dust_params["fesc_lya"],
    "dust_bump_amplitude": dust_params["dust_bump_amplitude"],
}

alt_parametrizations = {
    "tau_v": ("Av", lambda x: x["tau_v"] * av_to_tau_v),
}

print(f"Creating basis for {name} with Continuity SFH (Prospector-Beta priors).")
basis = GalaxyBasis(
    model_name=f"sps_{name}",
    redshifts=redshifts,
    grid=grid,
    emission_model=emission_model,
    sfhs=sfh_models,
    cosmo=cosmo,
    instrument=instrument,
    metal_dists=Z_dists,
    galaxy_params=galaxy_params,
    alt_parametrizations=alt_parametrizations,
    redshift_dependent_sfh=True,
    params_to_ignore=["agebins"],
    build_library=False,
    log_stellar_masses=beta_params["log_mass"],
)

multinode = True if sys.argv[2] == "0" else False  # Check if running in multinode mode
compile_grid = True if sys.argv[2] == "1" else False  # Check if running in multinode mode

param_transforms_to_save = {
    "tau_v": lambda x: x["Av"] / av_to_tau_v,  # Save Av instead of tau_v
}

basis.create_mock_library(
    emission_model_key=emission_key,
    out_name=f"library_{name}",
    out_dir=out_dir,
    overwrite=overwrite,
    mUV=(calculate_muv, cosmo),  # Calculate mUV using the provided cosmology
    mass_weighted_age=calculate_mass_weighted_age,  # Calculate mass-weighted age
    sfh_quant_25=(calculate_sfh_quantile, 0.25, True),  # Calculate SFH quantile at 25%
    sfh_quant_50=(calculate_sfh_quantile, 0.50, True),  # Calculate SFH quantile at 50%
    sfh_quant_75=(calculate_sfh_quantile, 0.75, True),  # Calculate SFH quantile at 75%
    UV=(calculate_colour, "U", "V", emission_key, True),  # Calculate UV colour (rest-frame)
    VJ=(calculate_colour, "V", "J", emission_key, True),  # Calculate VJ colour (rest-frame)
    log_surviving_mass=(calculate_surviving_mass, grid),  # Calculate surviving mass
    d4000=(calculate_d4000, emission_key),  # Calculate D4000 using the emission model
    beta=(calculate_beta, emission_key),  # Calculate beta using the qinstrument
    Ha_EW=(
        calculate_line_ew,
        emission_model,
        "Ha",
        emission_key,
    ),  # Calculate EW of H-alpha line
    Ha_flux=(
        calculate_line_flux,
        emission_model,
        "Ha",
        emission_key,
        cosmo,
    ),  # Calculate flux of H-alpha line
    OIII_EW=(
        calculate_line_ew,
        emission_model,
        "O3",
        emission_key,
    ),  # Calculate EW of OIII doublet
    OIII_flux=(
        calculate_line_flux,
        emission_model,
        "O3",
        emission_key,
        cosmo,
    ),  # Calculate flux of OIII doublet
    burstiness=calculate_burstiness,
    xi_ion0=(calculate_xi_ion0, emission_model, emission_key),
    Ndot_ion=(calculate_Ndot_ion, emission_key),
    n_proc=n_proc,
    verbose=False,
    batch_size=batch_size,
    parameter_transforms_to_save=param_transforms_to_save,
    compile_grid=compile_grid,
    multi_node=multinode,
    cat_type=cat_type,
    em_lines_to_save=["H 1 6562.80A", "O 3 5006.84A"],
    spectra_to_save=["dust_emission"],
)
