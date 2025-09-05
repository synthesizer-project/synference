# -*- coding: utf-8 -*-
import os
import warnings

from .grid import (
    generate_sfh_grid, generate_metallicity_distribution, generate_emission_models,
    generate_sfh_basis, GalaxyBasis, CombinedBasis,  calculate_sfr, grid_folder,
    calculate_muv, draw_from_hypercube, GalaxySimulator, generate_random_DB_sfh,
    test_out_of_distribution, calculate_mass_weighted_age, calculate_lum_weighted_age,
    calculate_flux_weighted_age, calculate_colour, calculate_d4000, calculate_beta, calculate_balmer_decrement,
    calculate_line_flux, calculate_line_ew, calculate_sfh_quantile, calculate_surviving_mass, SUPP_FUNCTIONS
)

from .utils import (
    load_grid_from_hdf5, calculate_min_max_wav_grid, generate_constant_R,
    list_parameters, rename_overlapping_parameters, FilterArithmeticParser,
    timeout_handler, TimeoutException, create_sqlite_db, f_jy_err_to_asinh,
    f_jy_to_asinh, check_scaling, detect_outliers, compare_methods_feature_importance,
    analyze_feature_contributions, optimize_sfh_xlimit, make_serializable
)

from .noise_models import (
    EmpiricalUncertaintyModel, create_uncertainty_models_from_EPOCHS_cat,
    DepthUncertaintyModel, UncertaintyModel, AsinhEmpiricalUncertaintyModel,
    GeneralEmpiricalUncertaintyModel, save_unc_model_to_hdf5, load_unc_model_from_hdf5
)

from .custom_runner import SBICustomRunner

try:
    from .sbi_runner import SBI_Fitter, MissingPhotometryHandler, Simformer_Fitter
except ImportError as e:
    print(e)
    print('Dependencies for SBI not installed. Only the grid generation functions will be available.')


#from .simformer import UncertainityModelTask
warnings.filterwarnings('ignore')

__all__ = [
    "generate_sfh_grid",
    "generate_metallicity_distribution",
    "generate_emission_models",
    "generate_sfh_basis",
    "GalaxyBasis",
    "generate_constant_R",
    "draw_from_hypercube",
    "grid_folder",
    "CombinedBasis",
    "calculate_muv",
    "calculate_sfr",
    "calculate_mass_weighted_age",
    "calculate_lum_weighted_age",
    "calculate_flux_weighted_age",
    "calculate_beta",
    "calculate_balmer_decrement",
    "calculate_line_flux",
    "calculate_line_ew",
    "calculate_d4000",
    "calculate_colour",
    "calculate_sfh_quantile",
    "calculate_surviving_mass",
    "SUPP_FUNCTIONS",
    "SBI_Fitter",
    "MissingPhotometryHandler",
    "Simformer_Fitter",
    "GalaxySimulator",
    "generate_random_DB_sfh",
    "EmpiricalUncertaintyModel",
    "AsinhEmpiricalUncertaintyModel",
    "test_out_of_distribution",
    "create_uncertainty_models_from_EPOCHS_cat",
    "load_grid_from_hdf5",
    "calculate_min_max_wav_grid",
    "list_parameters",
    "rename_overlapping_parameters",
    "FilterArithmeticParser",
    "timeout_handler",
    "TimeoutException",
    "create_sqlite_db",
    "f_jy_err_to_asinh",
    "f_jy_to_asinh",
    "check_scaling",
    "detect_outliers",
    "compare_methods_feature_importance",
    "analyze_feature_contributions",
    #"UncertainityModelTask",
    "DepthUncertaintyModel",
    "UncertaintyModel",
    "GeneralEmpiricalUncertaintyModel",
    "save_unc_model_to_hdf5",
    "load_unc_model_from_hdf5",
    "optimize_sfh_xlimit"
]