# ignore warnings for readability
import os
import sys

import numpy as np
from astropy.cosmology import Planck18
from synthesizer.emission_models.attenuation import (
    Calzetti2000,
)  # noqa
from synthesizer.emission_models.dust.emission import Greybody, IR_templates  # noqa
from synthesizer.emission_models.stellar.pacman_model import (
    BimodalPacmanEmission,
)  # noqa
from synthesizer.grid import Grid
from synthesizer.instruments import FilterCollection, Instrument
from synthesizer.parametric import SFH, ZDist
from tqdm import tqdm
from unyt import K, dimensionless

from sbifitter import (
    CombinedBasis,
    GalaxyBasis,
    calculate_muv,
    calculate_mwa,
    draw_from_hypercube,
    generate_constant_R,
    generate_random_DB_sfh,
    generate_sfh_basis,
)

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
    print(f'Loading filters from {path}')
    filterset = FilterCollection(path=path)
else:
    filterset = FilterCollection(filter_codes=filter_codes)


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

av_to_tau_v = 1.086  # conversion factor from Av to tau_v for the dust attenuation curve

Nmodels = 1000  # 00  # _000
batch_size = 50_000  # number of models to generate in each batch
redshift = (0.01, 12)
masses = (4, 12)
max_redshift = 20  # gives maximum age of SFH at a given redshift
cosmo = Planck18  # cosmology to use for age calculations

# ---------------------------------------------------------------


logAv = (-3, 0.7)  # Log-uniform between 0.001 and 5.0 magnitudes
dust_birth_fraction = (
    0.5,
    2.0,
)  # multiplier for the attenuation av for the birth cloud
log_zmet = (-3, -1.39)  # max of grid (e.g. 0.04)


# SFH
sfh_type = SFH.DenseBasis
tx_alpha = 1  # alpha parameter for the Dense Basis SFH
Nparam_SFH = 3  # number of SFH parameters to draw from the prior
ssfr = (-12, -7)  # log10(sSFR) in yr^-1

# sfh_type = SFH.DelayedExponential
# log_tau = (-2, 1) * Gyr  # log-uniform between 0.01 and 100 Gyr
# max_age = (0.00, 0.99)  # normalized to maximum age of the universe at that redshift.

# ---------------------------------------------------------------


# Generate the grid. Could also seperate hyper-parameters for each model.

full_params = {
    "redshift": redshift,
    "log_masses": masses,
    "log_Av": logAv,  # Av in magnitudes
    "dust_birth_fraction": dust_birth_fraction,
    "log_zmet": log_zmet,
    "ssfr": ssfr,  # log10(sSFR) in yr^-1
}

if sfh_type == SFH.DenseBasis:
    # Add dummy parameters for the SFH
    for i in tqdm(range(Nparam_SFH)):
        j = 100 * (i + 1) / (Nparam_SFH + 1)
        full_params[f"sfh_quantile_{j:.0f}"] = (
            0,
            1,
        )  # dummy parameters for the SFH
elif sfh_type == SFH.DelayedExponential:
    # Add parameters for the delayed exponential SFH
    # full_params["log_tau"] = log_tau
    # full_params["max_age"] = max_age  # normalized to maximum age of the universe
    # at that redshift.
    pass

# Draw samples from Latin Hypercube.
# unlog_keys are keys which should be unlogged after drawing from the hypercube.
# they will be renamed to not include 'log_' after drawing.
all_param_dict = draw_from_hypercube(full_params, Nmodels, rng=42, unlog_keys=["log_Av"])


# Create the grid
grid = Grid(
    "bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps.hdf5",
    grid_dir=grid_dir,
    new_lam=new_wav,
)

# Metallicity
Z_dists = [
    ZDist.DeltaConstant(log10metallicity=log_z) for log_z in all_param_dict["log_zmet"]
]

# Redshifts
redshifts = np.array(all_param_dict["redshift"])


if sfh_type == SFH.DenseBasis:
    # Draw SFH params from prior
    sfh_models = []
    logsfrs = []
    for i in tqdm(range(Nmodels)):
        z = all_param_dict["redshift"][i]
        logmass = all_param_dict["log_masses"][i]
        logssfr = all_param_dict["ssfr"][i]
        logsfr = logmass + logssfr
        logsfrs.append(logsfr)
        sfh, tx = generate_random_DB_sfh(
            Nparam=Nparam_SFH,
            tx_alpha=tx_alpha,
            redshift=z,
            logsfr=logsfr,
            logmass=logmass,
        )
        sfh_models.append(sfh)
        # Reassign parameters
        for j in range(Nparam_SFH):
            all_param_dict[f"sfh_quantile_{100 * (j + 1) / (Nparam_SFH + 1):.0f}"][i] = (
                tx[j]
            )
    full_params.pop("ssfr", None)  # remove ssfr from full_params
    # Add logSFR to all_param_dict
    all_param_dict["log_sfr"] = np.array(logsfrs)
else:
    sfh_models, _ = generate_sfh_basis(
        sfh_type=sfh_type,
        sfh_param_names=["tau", "max_age_norm"],
        sfh_param_arrays=(all_param_dict["tau"], all_param_dict["max_age"]),
        redshifts=redshifts,
        max_redshift=max_redshift,
        cosmo=cosmo,
        iterate_redshifts=False,
        calculate_min_age=False,
    )

dust_emission = Greybody(temperature=40 * K, emissivity=1.5)

# Essentially CF00 with explicit fesc and fesc_ly_alpha parameters.

emission_model = BimodalPacmanEmission(
    grid=grid,
    tau_v_ism="tau_v_ism",
    tau_v_birth="tau_v_birth",
    dust_curve_ism=Calzetti2000(),
    dust_curve_birth=Calzetti2000(),
    age_pivot=7 * dimensionless,
    dust_emission_ism=dust_emission,
    dust_emission_birth=dust_emission,
    fesc=0.0,  # escape fraction of ionizing photons
    fesc_ly_alpha=0.1,  # escape fraction of Lyman-alpha photons
)

# emission_model.plot_emission_tree(show=True)


# List of other varying or fixed parameters. Either a distribution to pull from or a list.
# Can be any parameter which can be property of emitter or
# galaxy and processed by the emission model.
galaxy_params = {
    "tau_v_ism": all_param_dict["Av"] / av_to_tau_v,
    "tau_v_birth": all_param_dict["Av"]
    * all_param_dict["dust_birth_fraction"]
    / av_to_tau_v,  # Av to tau_v for the birth cloud
}

# Dictionary of alternative parametrizations for the galaxy parameters -
# for parametrizing differently to Synthesizer if
# wanted. Should be a dictionary with keys as the parameter names and values as tuples of
# the new parameter name and a function which takes the
# parameter dictionary and returns the new parameter value (so it can be calculated from
# the other parameters if needed).


def db_sf_convert(param, param_dict):
    db_tuple = param_dict["db_tuple"]
    # dp_tuple has the folliwng
    # mass, sfr, tx_alpha, *sfh_quantiles
    if param.startswith("sfh_quantile_"):
        # Convert the SFH quantile parameters to the Dense Basis SFH format
        j = int(np.round(int(param.split("_")[-1]) / 100 * (Nparam_SFH + 1)))
        return db_tuple[j + 2]  # +3 because first three are mass, sfr, tx_alpha
    elif param == "log_sfr":
        # Convert log_sfr to the Dense Basis SFH format
        return db_tuple[1]
    elif param == "log_masses":
        # Convert log_masses to the Dense Basis SFH format
        return db_tuple[0]
    elif param == "tx_alpha":
        # Convert tx_alpha to the Dense Basis SFH format
        return db_tuple[2]
    elif param == "log_ssfr":
        # Convert log_ssfr to the Dense Basis SFH format
        return db_tuple[1] - db_tuple[0]
    else:
        raise ValueError(f"Unknown parameter {param.str} in db_tuple conversion.")


# These convert between the actual galaxy/emitter parameters and the
# parameters we want to sample.
alt_parametrizations = {
    "tau_v_birth": (
        "dust_birth_fraction",
        lambda x: x["tau_v_birth"] / x["tau_v_ism"],
    ),
    "tau_v_ism": ("Av", lambda x: x["tau_v_ism"] * av_to_tau_v),
    "db_tuple": (
        ["log_sfr"]
        + [
            f"sfh_quantile_{100 * (j + 1) / (Nparam_SFH + 1):.0f}"
            for j in range(Nparam_SFH)
        ],
        db_sf_convert,
    ),  # noqa: E501
}


sfh_name = str(sfh_type).split(".")[-1].split("'")[0]

name = f"BPASS_Chab_{sfh_name}_SFH_{redshift[0]}_z_{redshift[1]}_logN_{np.log10(Nmodels):.1f}_CF00_v1"  # noqa: E501

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
    params_to_ignore=[],
    build_grid=False,
)


combined_basis = CombinedBasis(
    bases=[basis],
    log_stellar_masses=all_param_dict["log_masses"],
    base_emission_model_keys=["total"],
    combination_weights=None,
    redshifts=redshifts,
    out_name=f"grid_{name}",
    out_dir=out_dir,
    draw_parameter_combinations=False,  # Since we have already drawn the parameters,
    # we don't need to combine them again.
)

for i in range(10):
    basis.plot_galaxy(
        idx=i,
        out_dir=f"{out_dir}/plots/{name}/",
        log_stellar_mass=all_param_dict["log_masses"][i],
    )

# Passing in extra analysis function to pipeline to calculate mUV.
# Any function could be passed in.
combined_basis.process_bases(
    overwrite=True,
    mUV=(calculate_muv, cosmo),
    mwa=calculate_mwa,
    n_proc=n_proc,
    verbose=False,
    batch_size=batch_size,
)

# Create grid - kinda overkill for a single case, but it does work.
combined_basis.create_grid(overwrite=True)
