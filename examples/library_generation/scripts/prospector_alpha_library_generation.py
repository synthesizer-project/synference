# ignore warnings for readability
"""Prospector-alpha-like library generation for synference.

Prospector-alpha (Leja et al. 2017) predates Prospector-Beta
(``prospector_beta_library_generation.py`` in this directory) and, unlike
it, has no hierarchical/parameter-dependent priors: total stellar mass and
stellar metallicity are drawn independently (no mass-metallicity relation,
no stellar-mass-function weighting), and the non-parametric Dirichlet-SFH
uses a *fixed* set of 6 age bins (not conditioned on redshift, unlike
Beta's Continuity-SFH). The one non-trivial piece is the Dirichlet-SFH
prior itself, drawn via ``synference.sample_dirichlet_sfr_fractions``
(``src/synference/priors_alpha.py``), which reproduces the Betancourt
(2010)/Leja et al. (2017) stick-breaking construction and is passed
directly to ``synthesizer.parametric.SFH.Dirichlet``.

Dust (Av/slope/dust_bump_amplitude/fesc_lya) is drawn independently via
``draw_from_hypercube``, matching Prospector-alpha's own (non-conditional)
dust priors: the diffuse-ISM screen (Av/slope/dust_bump_amplitude) maps onto
FSPS's ``dust_type=4`` (Kriek & Conroy 2013) curve, i.e. this repo's
``Calzetti2000(slope, ampl)``. The birth-cloud/diffuse split
(``dust1``/``dust_ratio``) *is* replicated via
``synthesizer.emission_models.stellar.pacman_model.BimodalPacmanEmission``
(``tau_v_birth = tau_v_ism * dust_ratio``, fixed ``PowerLaw(slope=-1.0)`` for
the birth-cloud curve, matching FSPS's fixed ``dust1_index=-1.0`` default and
``dust_tesc=7.0`` [10 Myr] age pivot); ``dust_ratio`` is drawn from
Prospector's own ``ClippedNormal(mean=1.0, sigma=0.3, 0-2)`` prior via
``synference.sample_dust_ratio``. Prospector-alpha's AGN torus
(``fagn``/``agn_tau``) and Draine & Li (2007) dust-emission
(``duste_umin``/``duste_qpah``/``duste_gamma``) parameters are also not
drawn: this pipeline's dust emission uses a fixed ``Greybody`` model (as in
every other script in this directory), not the Draine & Li templates or an
AGN torus component, so those parameters would have nothing to attach to.
"""

import os
import sys

import numpy as np
from astropy.cosmology import Planck18
from synthesizer.emission_models.attenuation import (
    Calzetti2000,
    PowerLaw,
)  # noqa
from synthesizer.emission_models.generators.dust import Greybody
from synthesizer.emission_models.stellar.pacman_model import (
    BimodalPacmanEmission,
)  # noqa
from synthesizer.grid import Grid
from synthesizer.instruments import FilterCollection, Instrument
from synthesizer.parametric import SFH, ZDist
from tqdm import tqdm
from unyt import K, Myr, dimensionless, unyt_array

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
    sample_dirichlet_sfr_fractions,
    sample_dust_ratio,
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

try:
    run_mode = int(sys.argv[2])
except Exception:
    run_mode = 2  # default: single-node, no grid compilation

print(f"Number of processes/task: {n_proc}")

av_to_tau_v = 1.086  # conversion factor from Av to tau_v for the dust attenuation curve


# Module-level (picklable) parameter transforms used by GalaxyBasis/create_mock_library,
# which run with n_proc worker processes and an optional multi_node path.
def _tau_v_ism_to_av(x):
    return x["tau_v_ism"] * av_to_tau_v


def _tau_v_birth_to_dust_ratio(x):
    return x["tau_v_birth"] / x["tau_v_ism"]


def _av_to_tau_v_ism(x):
    return x["Av"] / av_to_tau_v


def _av_to_tau_v_birth(x):
    return (x["Av"] / av_to_tau_v) * x["dust_ratio"]


overwrite = False  # whether to overwrite existing grids
Nmodels = 100_000  # number of models to generate
grid_name = "BPASS"  # name for the grid
cat_type = "spectra"  # spectra or photometry
cosmo = Planck18  # cosmology to use for age calculations
emission_key = "total"  # 'attenuated' if no dust emission or 'emergent' if fesc > 0
seed = 42  # Seed for reproducibility

# --- Prospector-alpha prior hyperparameters (Leja et al. 2017) ---
redshift_range = (0.01, 14.0)  # not part of the alpha template itself (usually fixed to spec-z)
log_mass_range = (8.0, 12.0)  # log10(M*/Msun), matches total_mass ~ LogUniform(1e8, 1e12)
logzsol_range = (-2.0, 0.19)  # independent (no mass-metallicity relation, unlike Beta)

# Fixed Dirichlet-SFH age bins (Gyr lookback time), identical for every galaxy.
alpha_agelims_gyr = np.array([1e-9, 0.1, 0.3, 1.0, 3.0, 6.0, 13.6])
nbins_sfh = len(alpha_agelims_gyr) - 1  # 6 bins -> 5 z_fraction latent variables

# --- Independent (non-conditional) dust priors ---
# Prospector-alpha's dust2/dust_index map onto this repo's existing Av/slope
# Calzetti-attenuation parametrization (diffuse ISM screen, FSPS dust_type=4
# / Kriek & Conroy 2013). The birth-cloud/diffuse split is replicated via
# dust_ratio (dust1 = dust2 * dust_ratio, Prospector's own ClippedNormal
# prior) and a fixed PowerLaw birth-cloud curve (FSPS dust1_index=-1.0
# default) below, applied only to stars younger than the age_pivot passed to
# BimodalPacmanEmission (FSPS dust_tesc=7.0 [10 Myr] default).
logAv = (-3, 0.7)  # log-uniform between 0.001 and ~5.0 magnitudes
slope_range = (-2.0, 0.5)  # Prospector-alpha's dust_index prior (TopHat)
fesc_lya_range = (0.0, 1.0)
dust_bump_amplitude_range = (0.0, 5.0)
dust_ratio_mean, dust_ratio_sigma = 1.0, 0.3  # Prospector-alpha's dust_ratio prior (ClippedNormal)
dust_ratio_range = (0.0, 2.0)

name = (
    f"{grid_name}_Chab_alpha_SFH_{redshift_range[0]}_z_{redshift_range[1]}_"
    f"logN_{np.log10(Nmodels):.1f}_CF00_v1"
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

if run_mode == 1:
    mask = np.ones(Nmodels, dtype=bool)

batch_size = np.sum(mask) + 1

grid_dict = {
    "FSPS": "fsps-3.2-mist-miles_chabrier03-0.1,300_cloudy-c23.01-sps",
    "BPASS": "bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps",
}

# --- Draw the independent Prospector-alpha parameters (plain joint LHC) ---
print("Drawing samples from Latin Hypercube.")
alpha_params = draw_from_hypercube(
    {
        "redshift": redshift_range,
        "log_mass": log_mass_range,
        "log_zmet": logzsol_range,
        "log_Av": logAv,
        "slope": slope_range,
        "fesc_lya": fesc_lya_range,
        "dust_bump_amplitude": dust_bump_amplitude_range,
    },
    Nmodels,
    rng=seed,
    unlog_keys=["log_Av"],
)
redshifts = alpha_params["redshift"]

# dust_ratio's ClippedNormal prior isn't expressible via draw_from_hypercube
# (uniform/log-uniform ranges only), so it gets its own truncated-normal draw.
alpha_params["dust_ratio"] = sample_dust_ratio(
    Nmodels,
    mean=dust_ratio_mean,
    sigma=dust_ratio_sigma,
    mini=dust_ratio_range[0],
    maxi=dust_ratio_range[1],
    rng=seed,
)

# --- Draw the Dirichlet-SFH stick-breaking z_fraction latent variables ---
# Not expressible via draw_from_hypercube: each of the (nbins_sfh - 1)
# components needs its own, position-dependent Beta marginal for the
# stick-breaking construction to yield a symmetric Dirichlet(1,...,1) prior
# over the nbins_sfh SFR fractions (see priors_alpha.py).
print("Drawing Dirichlet-SFH z_fraction parameters.")
z_fraction = sample_dirichlet_sfr_fractions(Nmodels, nbins_sfh, rng=seed)

# Fixed age bins (identical for every galaxy, unlike Beta's redshift-dependent bins).
agebins = unyt_array(
    np.column_stack([alpha_agelims_gyr[:-1], alpha_agelims_gyr[1:]]), "Gyr"
).to(Myr)

# Create the grid
grid = Grid(
    grid_dict[grid_name],
    grid_dir=grid_dir,
    new_lam=new_wav,
    use_precision=np.float32,
)
print(grid.available_lines)

# Metallicity: one DeltaConstant ZDist per galaxy, independently drawn (no MZR).
Z_dists = [
    ZDist.DeltaConstant(log10metallicity=log_z)
    for log_z in tqdm(alpha_params["log_zmet"], desc="Creating ZDist", disable=rank != 0)
]

# SFH: Dirichlet SFH built directly from the drawn z_fraction and the fixed
# alpha age bins (shared by every galaxy).
sfh_models = []
for i in tqdm(range(Nmodels), desc="Building Dirichlet SFHs", disable=rank != 0):
    if mask[i]:
        sfh_models.append(SFH.Dirichlet(z_fraction=z_fraction[i], agebins=agebins))
    else:
        sfh_models.append(None)

# Charlot & Fall (2000) two-component dust, with explicit fesc and
# fesc_ly_alpha parameters: a diffuse-ISM screen (tau_v_ism, matching FSPS's
# dust2/dust_type=4) attenuates all stars, plus an extra birth-cloud screen
# (tau_v_birth = tau_v_ism * dust_ratio, matching FSPS's dust1) attenuating
# only stars younger than age_pivot (FSPS dust_tesc default, 10 Myr). The
# birth-cloud curve is a fixed PowerLaw(slope=-1.0), matching FSPS's fixed
# dust1_index default (Prospector-alpha/-beta never make dust1_index free).
print("Creating emission model.")
emission_model = BimodalPacmanEmission(
    grid=grid,
    tau_v_ism="tau_v_ism",
    tau_v_birth="tau_v_birth",
    dust_curve_ism=Calzetti2000(slope="slope", ampl="dust_bump_amplitude"),
    dust_curve_birth=PowerLaw(slope=-1.0),
    age_pivot=7 * dimensionless,
    dust_emission_ism=Greybody(temperature=40 * K, emissivity=1.5),
    dust_emission_birth=Greybody(temperature=40 * K, emissivity=1.5),
    fesc=0.0,  # escape fraction of ionizing photons
    fesc_ly_alpha="fesc_lya",  # escape fraction of Lyman-alpha photons
)

galaxy_params = {
    "tau_v_ism": alpha_params["Av"] / av_to_tau_v,
    "tau_v_birth": (alpha_params["Av"] / av_to_tau_v) * alpha_params["dust_ratio"],
    "slope": alpha_params["slope"],
    "fesc_lya": alpha_params["fesc_lya"],
    "dust_bump_amplitude": alpha_params["dust_bump_amplitude"],
}

alt_parametrizations = {
    "tau_v_ism": ("Av", _tau_v_ism_to_av),
    "tau_v_birth": ("dust_ratio", _tau_v_birth_to_dust_ratio),
}

print(f"Creating basis for {name} with Dirichlet SFH (Prospector-alpha priors).")
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
    redshift_dependent_sfh=False,  # alpha's age bins are fixed, not redshift-dependent
    build_library=False,
    log_stellar_masses=alpha_params["log_mass"],
)

multinode = run_mode == 0  # Check if running in multinode mode
compile_grid = run_mode == 1  # Check if compiling the grid

param_transforms_to_save = {
    "tau_v_ism": _av_to_tau_v_ism,  # Save Av instead of tau_v_ism
    "tau_v_birth": _av_to_tau_v_birth,  # Save dust_ratio instead of tau_v_birth
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
    # BimodalPacmanEmission's dust-emission generator leaf nodes
    # (young_dust_emission_birth/_ism, old_dust_emission) each need their own
    # spectra plus their energy-balance inputs (young_reprocessed,
    # young_attenuated_nebular/_ism, old_reprocessed, old_attenuated) cached
    # on the emitter, or Pipeline._get_lines crashes with MissingSpectraType/
    # KeyError when em_lines_to_save is requested: EmissionModel.save_spectra()
    # marks everything not listed here as save=False, but get_lines() walks
    # every node of the full "total" tree (not just the requested lines'
    # ancestors), and Greybody._generate_lines() requires its own node's Sed,
    # plus DustEmission.get_scaling()'s energy-balance inputs, to already be
    # cached from the prior get_spectra() pass. This never showed up with the
    # old single-component PacmanEmission because its one dust_emission node
    # happened to already be the only thing in spectra_to_save.
    spectra_to_save=[
        "dust_emission",
        "young_dust_emission_birth",
        "young_dust_emission_ism",
        "old_dust_emission",
        "young_reprocessed",
        "young_attenuated_nebular",
        "young_attenuated_ism",
        "old_reprocessed",
        "old_attenuated",
    ],
)
