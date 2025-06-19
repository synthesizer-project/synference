import os

import numpy as np
from astropy.cosmology import Planck18
from synthesizer.emission_models.attenuation import Calzetti2000  # noqa
from synthesizer.emission_models.dust.emission import Greybody, IR_templates  # noqa
from synthesizer.emission_models.stellar.pacman_model import PacmanEmission  # noqa
from synthesizer.grid import Grid
from synthesizer.instruments import FilterCollection, Instrument
from synthesizer.parametric import SFH, ZDist
from unyt import Gyr, K, Msun, unyt_array

from sbifitter import (
    CombinedBasis,
    GalaxyBasis,
    SBI_Fitter,
    calculate_muv,
    draw_from_hypercube,
    generate_constant_R,
    generate_sfh_basis,
)

filter_codes = [
    "JWST/NIRCam.F090W",
    "JWST/NIRCam.F115W",
    "JWST/NIRCam.F150W",
    "JWST/NIRCam.F200W",
    "JWST/NIRCam.F277W",
    "JWST/NIRCam.F356W",
    "JWST/NIRCam.F410M",
    "JWST/NIRCam.F444W",
]
filterset = FilterCollection(filter_codes=filter_codes)

# Consistent wavelength grid for both SPS grids and filters
new_wav = generate_constant_R(
    R=300, auto_start_stop=True, filterset=filterset, max_redshift=15
)

filterset.resample_filters(new_lam=new_wav)

instrument = Instrument("HST+JWST", filters=filterset)


grid_dir = os.environ["SYNTHESIZER_GRID_DIR"]


# path for this file

dir_path = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(os.path.dirname(os.path.dirname(dir_path)), "grids/")

# ---------------------------------------------------------------
# Configure model

Nmodels = 100_000
batch_size = 40_000  # number of models to generate in each batch

redshift = (0.01, 12)
masses = (6, 12)
max_redshift = 20  # gives maximum age of SFH at a given redshift
cosmo = Planck18  # cosmology to use for age calculations

# ---------------------------------------------------------------
# SFH
sfh_type = SFH.DelayedExponential
log_tau = (-2, 1)  # log-uniform between 0.01 and 10 Gyr
max_age = (
    0.00,
    0.99,
)  # normalized to maximum age of the universe at that redshift.
sfh_param_units = [Gyr, Gyr]
# Dust
tau_v = (0.0, 4.0)  # V-band optical depth of the ISM
# Metallicity
log_zmet = (
    -3,
    -1.4,
)  # Metallicity in absolute log scale, i.e. log10(Z/Zsun) where Zsun = 0.02
# ---------------------------------------------------------------

# Generate the grid. Could also seperate hyper-parameters for each model.

full_params = {
    "redshift": redshift,
    "masses": masses,
    "tau_v": tau_v,
    "log_zmet": log_zmet,
    "log_tau": log_tau,
    "max_age": max_age,
}

param_grid = draw_from_hypercube(Nmodels, full_params, rng=42)

# Unpack the parameters

all_param_dict = {}
for i, key in enumerate(full_params.keys()):
    all_param_dict[key] = param_grid[:, i]

# ---------------------------------------------------------------
# Synthesizer setup

# Create the grid
grid = Grid(
    "bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps.hdf5",
    grid_dir=grid_dir,
    new_lam=new_wav,
)

# Metallicity Distributions
Z_dists = [
    ZDist.DeltaConstant(log10metallicity=log_z) for log_z in all_param_dict["log_zmet"]
]

# Redshifts
redshifts = np.array(all_param_dict["redshift"])

# SFH Distributions
sfh_param_arrays = np.vstack((all_param_dict["tau"], all_param_dict["max_age"])).T
sfh_models, _ = generate_sfh_basis(
    sfh_type=sfh_type,
    sfh_param_names=["tau", "max_age_norm"],
    sfh_param_arrays=sfh_param_arrays,
    redshifts=redshifts,
    max_redshift=max_redshift,
    cosmo=cosmo,
    sfh_param_units=sfh_param_units,
    iterate_redshifts=False,
    calculate_min_age=False,
)
# Dust Emsission
dust_emission = Greybody(temperature=40 * K, emissivity=1.5)

# Dust attenuation
emission_model = PacmanEmission(
    grid=grid,
    tau_v="tau_v",
    dust_curve=Calzetti2000(),
    dust_emission=dust_emission,
    fesc=0.0,  # escape fraction of ionizing photons
    fesc_ly_alpha=0.1,  # escape fraction of Lyman-alpha photons
)

sfh_name = str(sfh_type).split(".")[-1].split("'")[0]

galaxy_params = {
    "tau_v": all_param_dict["tau_v"]
}  # pass in any other emitter parameter here

name = f"""BPASS_{sfh_name}_SFH_{redshift[0]}_z_{redshift[1]}
        _logN_{np.log10(Nmodels):.1f}_Chab_min_example"""


# ---------------------------------------------------------------
# Grid Generation

# Generate the basis
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
    redshift_dependent_sfh=True,
    params_to_ignore=[],
    build_grid=False,
)
# Build the model here - could combine basis's with different grids etc
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

# Passing in extra analysis function to pipeline to calculate mUV and store it.
combined_basis.process_bases(
    overwrite=True,
    mUV=(calculate_muv, cosmo),
    n_proc=6,
    verbose=False,
    batch_size=batch_size,
)

# Create grid from pipeline outputs
combined_basis.create_grid(overwrite=True)


# ---------------------------------------------------------------
# SBI Fitting

# Initialize SBI
empirical_model_fitter = SBI_Fitter.init_from_hdf5(
    hdf5_path=f"grid_{name}.hdf5", model_name=f"sbi_{name}"
)

# Create explicit feature array for training.
# E.g. can change filters, add colors expclitly,
# add other features (mUV, spectral indices, etc) or
# do normalisation. Can input error models or flags for missing data, etc.
# Can also not do this and use an on the fly simulator.
empirical_model_fitter.create_feature_array_from_raw_photometry(
    extra_features=[],
    normalize_method=None,
)

# Run SBI - validation is done automatically
empirical_model_fitter.run_single_sbi(
    learning_rate=1e-4,
    hidden_features=90,
    num_transforms=4,
    model_type="maf",
    plot=True,
)

# Do inference from model
observed_data_vector = [
    30.2,
    28.7,
    28.5,
    28.0,
    27.5,
    27.3,
    26.5,
    26.2,
]  # Example observed data vector in magnitudes
posterior = empirical_model_fitter.sample_posterior(observed_data_vector)
empirical_model_fitter.recover_SED(observed_data_vector)

# Optimize hyperparameters using Optuna - parameters are saved in SQLite database
empirical_model_fitter.optimize_sbi(
    study_name="sbi_min_example",
    suggested_hyperparameters={
        "learning_rate": [1e-2, 1e-5],
        "num_transforms": [4, 8],
        "hidden_features": [50, 100],
    },
    persistent_storage=True,
)
