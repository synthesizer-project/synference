# ignore warnings for readability
import copy
import os
import sys

import numpy as np
from astropy.cosmology import Planck18
from synthesizer.emission_models.attenuation import (
    MWN18,
)  # noqa

mwn18 = MWN18()
from synthesizer.emission_models.dust.emission import Greybody, IR_templates  # noqa
from synthesizer.emission_models.stellar.pacman_model import (
    BimodalPacmanEmission,
)  # noqa
from synthesizer.grid import Grid
from synthesizer.instruments import FilterCollection, Instrument
from synthesizer.parametric import SFH, ZDist
from tqdm import tqdm
from unyt import K, Myr, dimensionless, unyt_array

from sbifitter import (
    CombinedBasis,
    GalaxyBasis,
    calculate_beta,
    calculate_d4000,
    calculate_mass_weighted_age,
    calculate_muv,
    calculate_sfh_quantile,
    calculate_sfr,
    draw_from_hypercube,
    generate_constant_R,
    generate_random_DB_sfh,
    generate_sfh_basis,
    calculate_surviving_mass,
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
    print(f"Loading filters from {path}")
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


def continuity_agebins(
    redshift,
    cosmo=Planck18,
    Nbins=6,
    first_bin=3 * Myr,
    second_bin=10 * Myr,
    last_bin="15%",
    max_redshift=20,
):
    """
    Generate age bins for the Continuity SFH.
    The first two bins are fixed, the last bin is a percentage of the maximum age at that redshift,
    and the middle bins are logarithmically spaced.

    Parameters
    ----------
    redshift : float
        The redshift at which to calculate the age bins.
    cosmo : astropy.cosmology.Cosmology, optional
        The cosmology to use for age calculations, by default Planck18.
    Nbins : int, optional
        The total number of bins, by default 6.
    first_bin : unyt_quantity, optional
        The ending lookback time for the first bin, by default 3 Myr.
    second_bin : unyt_quantity, optional
        The ending lookback time for the second bin, by default 10 Myr.
    last_bin : str or unyt_quantity, optional
        The ending lookback time for the last bin, which can be a percentage of the max_age
        at that redshift or a fixed value, by default '15%'.
    max_redshift : float, optional
        The maximum redshift to consider for the age bins, by default 20.
    Returns
    -------
    list of unyt_quantity
        The age bins in lookback time.

    """

    # Calculate the maximum age at the given redshift
    max_age = cosmo.age(redshift).to_value("Myr")
    age_at_max = cosmo.age(max_redshift).to_value("Myr")
    available_age = max_age - age_at_max

    bins = [0.0, first_bin.to_value("Myr")]
    if isinstance(last_bin, str) and last_bin.endswith("%"):
        # If last_bin is a percentage, calculate it based on the remaining age
        last_bin_value = available_age * float(last_bin[:-1]) / 100.0
    else:
        # Otherwise, treat last_bin as a fixed value
        last_bin_value = last_bin.to_value("Myr")

    age_at_last_bin = available_age - last_bin_value
    # Generate the middle bins logarithmically spaced
    middle_bins = np.logspace(
        np.log10(second_bin),
        np.log10(age_at_last_bin),
        Nbins - 2,
    )
    # Combine all bins and convert to unyt_quantity
    all_bins = np.concatenate((bins, middle_bins, [available_age]))

    # reshape to (N, 2) and readd relevant bins to mathc shape.
    # e.g. (0, 1e7), (1e7, 1e8), (1e8, 1.5e8), (1.5e8, 2.0e8), (2.0e8, 3.0e8), (3.0e8, 4.0e8)
    all_bins = np.array(
        [(all_bins[i], all_bins[i + 1]) for i in range(len(all_bins) - 1)]
    )
    all_bins = unyt_array(all_bins, "Myr")

    return all_bins


try:
    n_proc = int(sys.argv[1])
except Exception:
    n_proc = 6

av_to_tau_v = 1.086  # conversion factor from Av to tau_v for the dust attenuation curve
overwrite = True  # whether to overwrite existing grids
Nmodels = 200  # 00  # _000
batch_size = 40_000  # number of models to generate in each batch
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
log_zmet = (-4, -1.39)  # max of grid (e.g. 0.04)


sfhs = {
    "delayed_exponential": {
        "sfh_type": SFH.DelayedExponential,
        "sfh_param_names": ["tau", "max_age_norm"],
        "sfh_units": [Gyr, None],
        "tau": (-2, 2),  # log-uniform between 0.01 and 100 Gyr
        "max_age_norm": (
            0.01,
            0.99,
        ),  # normalized to maximum age of the universe at that redshift.
        "unlog_keys": ["tau"],  # tau is in Gyr, so we need to unlog it
    }
}
"""
"delayed_exponential": {
    "sfh_type": SFH.DelayedExponential,
    "sfh_param_names": ["tau", "max_age_norm"],
    "sfh_units": [Gyr, None],
    "tau": (-2, 2),  # log-uniform between 0.01 and 100 Gyr
    "max_age_norm": (0.01, 0.99),  # normalized to maximum age of the universe at that redshift.
    "unlog_keys": ["tau"],  # tau is in Gyr, so we need to unlog it
},
"continuity": { # SWITCH SYNTHESIZER BRANCH AND UNCOMMENT CONTINUITY REFERENCES BELOW
    "sfh_type": SFH.Continuity,
    "agebins": continuity_agebins,
    "df": 2,
    "scale": 1.0,  # scale for students-t prior
    "params_to_ignore": ["max_age", "agebins"],
    "nbins": 6,  # number of bins to use for the Continuity SFH
    "sfh_param_names": [],
},
"dense_basis": {
    "Nparam_SFH": 3,
    "tx_alpha": 1,
    "sfh_type": SFH.DenseBasis,
    "sfh_param_names": [
        "ssfr",
    ],
    "ssfr": (-12, -7),  # log10(sSFR) in yr^-1'
    "params_to_ignore": ["max_age"],
},
"double_powerlaw": {
    "sfh_type": SFH.DoublePowerLaw,
    "sfh_param_names": ["peak_age_norm", "alpha", "beta"],
    "peak_age_norm": (
        0.00,
        0.99,
    ),  # normalized to maximum age of the universe at that redshift.
    "alpha": (-1, 3), # e.g 0.1 to 10^3
    # power-law index for the first part of the SFH - probably should be log-uniform
    "beta": (-1, 3),  # power-law index for the second part of the SFH
    "params_to_ignore": ["max_age"],  # max_age is not used in the DoublePowerLaw SFH
    "sfh_units": [None, None, None],  # units for the parameters
    "unlog_keys": ["alpha", "beta"],
},
"declining_exponential": {
    "sfh_type": SFH.DecliningExponential,
    "sfh_param_names": ["tau", "max_age_norm"],
    "sfh_units": [Gyr, None],
    "tau": (0.01, 10),  # Uniform between 0.01 and 10 Gyr
    "max_age_norm": (0.01, 0.99),  # normalized to maximum age of the universe at that redshift.
},
"""

sfhs = {
    "log_normal": {
        "sfh_type": SFH.LogNormal,
        "sfh_param_names": ["tau", "peak_age_norm"],
        "tau": (0.05, 2.5),  # in Gyr
        "tau_units": [None, None],
        "peak_age_norm": (
            0.001,
            0.99,
        ),  # normalized to maximum age of the universe at that redshift.
        "params_to_ignore": ["max_age"],  # correlates with redshift, so not needed
    }
}


# Generate the grid. Could also seperate hyper-parameters for each model.

full_params_base = {
    "redshift": redshift,
    "log_masses": masses,
    "log_Av": logAv,  # Av in magnitudes
    "dust_birth_fraction": dust_birth_fraction,
    "log_zmet": log_zmet,
}

for sfh_name, sfh_params in sfhs.items():
    full_params = copy.deepcopy(full_params_base)  # start with the base parameters
    sfh_type = sfh_params["sfh_type"]
    sfh_param_names = sfh_params["sfh_param_names"]

    sfh_name = str(sfh_type).split(".")[-1].split("'")[0]

    name = f"BPASS_Chab_{sfh_name}_SFH_{redshift[0]}_z_{redshift[1]}_logN_{np.log10(Nmodels):.1f}_CF00_MWN18_v2"  # noqa: E501
    print(f"{out_dir}/grid_{name}.hdf5")
    if os.path.exists(f"{out_dir}/v2/grid_{name}.hdf5") and not overwrite:
        print(f"Grid {name} already exists, skipping.")
        continue

    for param_name in sfh_param_names:
        full_params[param_name] = sfh_params[param_name]
    if sfh_type == SFH.DenseBasis:
        # Add dummy parameters for the SFH
        for i in tqdm(range(sfh_params["Nparam_SFH"])):
            j = 100 * (i + 1) / (sfh_params["Nparam_SFH"] + 1)
            full_params[f"sfh_quantile_{j:.0f}"] = (
                0,
                1,
            )  # dummy parameters for the SFH
    """elif sfh_type == SFH.Continuity:
        # Add dummy parameters for the Continuity SFH
        for i in tqdm(range(sfh_params["nbins"])):
            j = 100 * (i + 1) / (sfh_params["nbins"] + 1)
            full_params[f"sfh_quantile_{j:.0f}"] = (
                0,
                1,
            )
        # add to SFH_param_names"""

    # Draw samples from Latin Hypercube.
    # unlog_keys are keys which should be unlogged after drawing from the hypercube.
    # they will be renamed to not include 'log_' after drawing.
    all_param_dict = draw_from_hypercube(
        full_params,
        Nmodels,
        rng=42,
        unlog_keys=["log_Av"] + sfh_params.get("unlog_keys", []),
    )  # noqa: E501

    # Create the grid
    grid = Grid(
        "bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps.hdf5",
        grid_dir=grid_dir,
        new_lam=new_wav,
    )
    # Metallicity
    Z_dists = [
        ZDist.DeltaConstant(log10metallicity=log_z)
        for log_z in all_param_dict["log_zmet"]
    ]

    # Redshifts
    redshifts = np.array(all_param_dict["redshift"])

    if sfh_type == SFH.DenseBasis:
        # Draw SFH params from prior
        Nparam_SFH = sfh_params["Nparam_SFH"]
        tx_alpha = sfh_params["tx_alpha"]  # tx_alpha for the Dense Basis SFH
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
                all_param_dict[f"sfh_quantile_{100 * (j + 1) / (Nparam_SFH + 1):.0f}"][
                    i
                ] = tx[j]
        full_params.pop("ssfr", None)  # remove ssfr from full_params
        # Add logSFR to all_param_dict
        all_param_dict["log_sfr"] = np.array(logsfrs)
        # UNCOMMENT IN GENERAL! Just because main doesn't have this brnch.
        """elif sfh_type == SFH.Continuity:
            # Draw from prior.
            sfh_models = []
            for i in tqdm(range(Nmodels)):
                z_i = all_param_dict["redshift"][i]
                agebins = sfh_params["agebins"](z_i, cosmo=cosmo, Nbins=6)
                sfh_models.append(
                    sfh_params["sfh_type"].init_from_prior(
                        agebins,
                        df=sfh_params["df"],
                        scale=sfh_params["scale"],
                        limits=(-30, 30),
                    )
        )"""

    else:
        if "beta" in sfh_param_names:
            all_param_dict["beta"] = (
                -1 * all_param_dict["beta"]
            )  # log-normal SFH has negative beta
        sfh_models, _ = generate_sfh_basis(
            sfh_type=sfh_type,
            sfh_param_names=sfh_param_names,
            sfh_param_arrays=[all_param_dict[param] for param in sfh_param_names],
            sfh_param_units=sfh_params.get("sfh_units", None),
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
        dust_curve_ism=MWN18(),  # Calzetti2000(),
        dust_curve_birth=MWN18(),  # Calzetti2000(),
        age_pivot=7 * dimensionless,
        dust_emission_ism=dust_emission,
        dust_emission_birth=dust_emission,
        fesc=0.0,  # escape fraction of ionizing photons
        fesc_ly_alpha=0.1,  # escape fraction of Lyman-alpha photons
    )

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

    def db_sf_convert(param, param_dict, Nparam_SFH=3):
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
    }

    if sfh_type == SFH.DenseBasis:
        alt_parametrizations["db_tuple"] = (
            ["log_sfr"]
            + [
                f"sfh_quantile_{100 * (j + 1) / (Nparam_SFH + 1):.0f}"
                for j in range(Nparam_SFH)
            ],
            db_sf_convert,
        )  # noqa: E501
    """elif sfh_type == SFH.Continuity:
        alt_parametrizations["logsfr_ratios"] = (
            [f"logsfr_ratio_{j}" for j in range(sfh_params["nbins"] - 1)],
            lambda p, p_dict: p_dict["logsfr_ratios"][int(p.split("_")[-1])],  # noqa: E501
        )"""

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
        params_to_ignore=sfh_params.get("params_to_ignore", []),
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
        try:
            basis.plot_galaxy(
                idx=i,
                out_dir=f"{out_dir}/plots/{name}/",
                log_stellar_mass=all_param_dict["log_masses"][i],
            )
        except ValueError as e:
            print(f"Error plotting galaxy {i}: {e}")
            continue

    # Passing in extra analysis function to pipeline to calculate mUV.
    # Any function could be passed in.
    combined_basis.process_bases(
        overwrite=overwrite,
        mUV=(calculate_muv, cosmo),  # Calculate mUV using the provided cosmology
        mass_weighted_age=calculate_mass_weighted_age,  # Calculate mass-weighted age
        sfr_3=(calculate_sfr, 3 * Myr),  # Calculate SFR averaged over the last 3 Myr
        sfr_10=(calculate_sfr, 10 * Myr),  # Calculate SFR averaged over the last 10 Myr
        sfr_30=(calculate_sfr, 30 * Myr),  # Calculate SFR averaged over the last 30 Myr
        sfr_100=(
            calculate_sfr,
            100 * Myr,
        ),  # Calculate SFR averaged over the last 100 Myr
        sfh_quant_25=(
            calculate_sfh_quantile,
            0.25,
            True,
        ),  # Calculate SFH quantile at 25%
        sfh_quant_50=(
            calculate_sfh_quantile,
            0.50,
            True,
        ),  # Calculate SFH quantile at 50%
        sfh_quant_75=(
            calculate_sfh_quantile,
            0.75,
            True,
        ),  # Calculate SFH quantile at 75%
        # UV=(calculate_colour, 'U','V', "total"), # These are broken as they need to be rest-frame
        log_surviving_mass=(calculate_surviving_mass, grid),  # Calculate surviving mass
        # VJ=(calculate_colour, 'V','J', "total"),
        d4000=calculate_d4000,  # Calculate D4000 index
        beta=calculate_beta,  # Calculate beta using the instrument
        n_proc=n_proc,
        verbose=False,
        batch_size=batch_size,
    )

    # Create grid - kinda overkill for a single case, but it does work.
    combined_basis.create_grid(overwrite=overwrite)
