import os

import numpy as np
from astropy.cosmology import Planck18 as cosmo
from synthesizer import Grid
from synthesizer.emission_models import (
    AttenuatedEmission,
    DustEmission,
    EmissionModel,
    PacmanEmission,
    UnifiedAGN,
)
from synthesizer.emission_models.attenuation import PowerLaw
from synthesizer.emission_models.dust.emission import Blackbody, Greybody
from synthesizer.instruments import FilterCollection, Instrument
from synthesizer.parametric import SFH, ZDist
from tqdm import tqdm
from unyt import K, Msun, cm, deg, erg, g, s

from synference import (
    GalaxyBasis,
    calculate_agn_fraction,
    calculate_beta,
    calculate_d4000,
    calculate_mass_weighted_age,
    calculate_muv,
    calculate_surviving_mass,
    draw_from_hypercube,
    generate_sfh_basis,
)

# Get the NLR and BLR grids
nlr_grid = Grid("qsosed-test_cloudy-c23.01-nlr-test")
blr_grid = Grid("qsosed-test_cloudy-c23.01-blr-test")
stellar_grid = Grid("bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps")

dir_path = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(dir_path))), "grids/")


Nmodels = 1000

full_params = {
    "redshift": (0.01, 10.0),
    "log_masses": (4, 12),
    "tau_v_stellar": (0.0, 3.0),
    "fesc_stellar": (0.0, 1.0),
    "fesc_ly_alpha_stellar": (0.0, 1.0),
    "tau_v_agn": (0.0, 3.0),
    "covering_fraction_blr": (0.1, 0.3),
    "covering_fraction_nlr": (0.1, 0.3),
    "log_eddington_ratio": (-3, 0.5),
    "theta_torus": (30, 70) * deg,
    "inclination": (0, 90) * deg,
    "log_zmet": (-4, -1.39),
    "log_torus_column_density": (19, 24),
    "tau": (0.2, 2.0),
    "peak_age_norm": (0.001, 0.95),  # Fraction of the age of the Universe at the given redshift
}

# Stellar Emission Parameters
dust_curve_stellar = PowerLaw(slope=-0.7)
dust_emission_model_stellar = Blackbody(temperature=100 * K)
sfh_type = SFH.LogNormal
sfh_param_names = ["tau", "peak_age_norm"]
sfh_params = {
    "sfh_units": [None, None],  # tau in Gyr, peak_age is a fraction of max age
}
max_redshift = 20  # No SFR at z>20

# Intrinsic AGN emission
torus_emission_model = Greybody(1000 * K, 1.5)
ionisation_parameter = 0.1
hydrogen_density = 1e5  # in cm^-3
epsilon = 0.1  # Radiative efficiency

# Attenuated AGN emission
dust_curve_agn = PowerLaw(slope=-1.0)
dust_emission_model_agn = Greybody(45 * K, 1.2)

# Dust Emission Parameters
dust_emission_model_galaxy = Greybody(4 * K, 1.2)

emission_key = "total"
overwrite = False
cat_type = "spectra"  # or spectra

name = f"AGN_test_library_{cat_type}_v1"


# BH Parameters
all_param_dict = draw_from_hypercube(
    full_params, Nmodels, rng=42, unlog_keys=["log_torus_column_density", "log_eddington_ratio"]
)  # noqa: E501


def bh_mass_from_mstar(mstar, alpha=7.45, beta=1.05, scatter=0.25):
    """Calculate black hole mass from stellar mass using the relation from Reines & Volonteri (2015)."""  # noqa: E501
    log_mstar_11 = np.log10(mstar / (1e11 * Msun))
    log_mbh = alpha + beta * log_mstar_11
    log_mbh += np.random.normal(0, scatter, size=log_mbh.shape)
    return 10**log_mbh * Msun


def accretion_from_eddington(mbh, edd_frac=0.1):
    """Calculate accretion rate from Eddington ratio."""
    l_edd = 1.26e38 * (mbh / Msun) * erg / s
    acc_rate = (edd_frac * l_edd) / (0.1 * 9e20 * erg / g)
    return acc_rate.to("Msun/yr")


def torus_vdisp_from_mbh(mbh):
    """Calculate torus velocity dispersion from black hole mass."""
    # Estimate R_BLR in light days
    r_blr_ld = 10 * (mbh / (1e8 * Msun)) ** 0.5
    r_blr = r_blr_ld * 259e15 * cm
    v_disp = np.sqrt(6.67e-8 * mbh.to("g") / r_blr) * cm / s
    return v_disp


# Instrument
# ------------------------------------------------------------

instrument = "HST+JWST"

path = f"/home/tharvey/work/synference/examples/library_generation/filters/{instrument}.hdf5"
filterset = FilterCollection(path=path)
instrument = Instrument(instrument, filters=filterset)

# -----------------------------------------------------------------
# Setup Emission Models
# -----------------------------------------------------------------

# Stellar Emission
pc_model = PacmanEmission(
    grid=stellar_grid,
    tau_v="tau_v_stellar",
    dust_curve=dust_curve_stellar,
    dust_emission_ism=dust_emission_model_stellar,
    fesc="fesc_stellar",
    fesc_ly_alpha="fesc_ly_alpha_stellar",
    label="stellar_total",
)

# AGN Model
uniagn = UnifiedAGN(
    nlr_grid,
    blr_grid,
    covering_fraction_nlr="covering_fraction_nlr",
    covering_fraction_blr="covering_fraction_blr",
    torus_emission_model=torus_emission_model,
    ionisation_parameter=ionisation_parameter,
    hydrogen_density=hydrogen_density,
    label="agn_intrinsic",
)


# Define an emission model to attenuate the intrinsic AGN emission
att_agn_model = AttenuatedEmission(
    dust_curve=dust_curve_agn,
    apply_to=uniagn,
    tau_v="tau_v_agn",
    emitter="blackhole",
    label="agn_attenuated",
)

# Now combine the AGN and stellar emission
gal_intrinsic = EmissionModel(
    label="total_intrinsic",
    combine=(uniagn, pc_model["intrinsic"]),
    emitter="galaxy",
)

# Now combine the attenuated AGN and stellar emission
gal_attenuated = EmissionModel(
    label="total_attenuated",
    combine=(att_agn_model, pc_model["attenuated"]),
    related_models=(gal_intrinsic,),
    emitter="galaxy",
)

# And now include the dust emission
gal_dust = DustEmission(
    dust_emission_model=dust_emission_model_galaxy,
    dust_lum_intrinsic=gal_intrinsic,
    dust_lum_attenuated=gal_attenuated,
    emitter="galaxy",
    label="dust_emission",
)
gal_total = EmissionModel(
    label="total",
    combine=(gal_attenuated, gal_dust),
    related_models=(gal_intrinsic,),
    emitter="galaxy",
)

# gal_total.plot_emission_tree()
# -----------------------------------------------------------------
# Get parameters from sampling

bh_masses = bh_mass_from_mstar(10 ** all_param_dict["log_masses"] * Msun)
eddington_ratios = all_param_dict["eddington_ratio"]
accretion_rates = accretion_from_eddington(bh_masses, edd_frac=eddington_ratios)
# For simplicity, set the accretion disc metallicity to be the same as the stellar metallicity.
accretion_disc_zmet = 10 ** all_param_dict["log_zmet"]
redshifts = all_param_dict["redshift"]

# These should all have unique names (even if they are nested inside dicts) or they won't be saved.
galaxy_params = {
    "tau_v_stellar": all_param_dict["tau_v_stellar"],
    "fesc_stellar": all_param_dict["fesc_stellar"],
    "fesc_ly_alpha_stellar": all_param_dict["fesc_ly_alpha_stellar"],
    "bh": {
        "mass": bh_masses,
        "accretion_rate": accretion_rates,
        "metallicity": accretion_disc_zmet,
        "epsilon": epsilon,
        "column_densities": all_param_dict["torus_column_density"] * cm**-2,
        "tau_v_agn": all_param_dict["tau_v_agn"],
        "covering_fraction_blr": all_param_dict["covering_fraction_blr"],
        "covering_fraction_nlr": all_param_dict["covering_fraction_nlr"],
        "inclination": all_param_dict["inclination"],
        "theta_torus": all_param_dict["theta_torus"],
    },
}


# -----------------------------------------------------------------

# 1.82698655128479
# find which parameter array has this value

for key, value in galaxy_params["bh"].items():
    if np.any(np.isclose(value, 1.82698655128479)):
        print(f"Found in {key}")

Z_dists = [
    ZDist.DeltaConstant(log10metallicity=log_z)
    for log_z in tqdm(all_param_dict["log_zmet"], desc="Creating ZDist")
]

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


basis = GalaxyBasis(
    model_name=f"sps_{name}",
    redshifts=redshifts,
    grid=stellar_grid,
    emission_model=gal_total,
    sfhs=sfh_models,
    cosmo=cosmo,
    instrument=instrument,
    metal_dists=Z_dists,
    galaxy_params=galaxy_params,
    alt_parametrizations={},
    redshift_dependent_sfh=True,
    build_grid=False,
    log_stellar_masses=all_param_dict["log_masses"],
    params_to_ignore=["max_age"],
)

basis.create_mock_library(
    emission_model_key=emission_key,
    out_name=f"library_{name}",
    out_dir=out_dir,
    overwrite=overwrite,
    mUV=(calculate_muv, cosmo),  # Calculate mUV using the provided cosmology
    mass_weighted_age=calculate_mass_weighted_age,  # Calculate mass-weighted age
    # UV=(calculate_colour, "U", "V", emission_key, True),  # Calculate UV colour (rest-frame)
    # VJ=(calculate_colour, "V", "J", emission_key, True),  # Calculate VJ colour (rest-frame)
    log_surviving_mass=(calculate_surviving_mass, stellar_grid),  # Calculate surviving mass
    d4000=(calculate_d4000, emission_key),  # Calculate D4000 using the emission model
    beta=(calculate_beta, emission_key),  # Calculate beta using the qinstrument
    agn_fraction=(calculate_agn_fraction, emission_key),
    n_proc=6,
    verbose=False,
    batch_size=40_000,
    parameter_transforms_to_save={},
    compile_grid=True,
    multi_node=False,
    spectra_to_save=["agn_attenuated"],
)
