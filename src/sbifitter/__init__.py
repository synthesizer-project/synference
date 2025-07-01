# -*- coding: utf-8 -*-
import os


from .grid import (
    generate_sfh_grid, generate_metallicity_distribution, generate_emission_models, 
    generate_sfh_basis, GalaxyBasis, CombinedBasis,
    calculate_muv, calculate_mwa, draw_from_hypercube, GalaxySimulator, generate_random_DB_sfh,
    EmpiricalUncertaintyModel, test_out_of_distribution, create_uncertainity_models_from_EPOCHS_cat
)

from .utils import (
    load_grid_from_hdf5, calculate_min_max_wav_grid, generate_constant_R,
    list_parameters, rename_overlapping_parameters, FilterArithmeticParser,
    timeout_handler, TimeoutException, create_sqlite_db, f_jy_err_to_asinh,
    f_jy_to_asinh
)


try:
    from .sbi import SBI_Fitter, MissingPhotometryHandler, Simformer_Fitter
except ImportError as e:
    print(e)
    print('Dependencies for SBI not installed. Only the grid generation functions will be available.')


__all__ = [
    "generate_sfh_grid",
    "generate_metallicity_distribution",
    "generate_emission_models",
    "generate_sfh_basis",
    "GalaxyBasis",
    "generate_constant_R",
    "draw_from_hypercube",
    "CombinedBasis",
    "calculate_muv",
    "calculate_mwa",
    "SBI_Fitter",
    "MissingPhotometryHandler",
    "Simformer_Fitter",
    "GalaxySimulator",
    "generate_random_DB_sfh",
    "EmpiricalUncertaintyModel",
    "test_out_of_distribution",
    "create_uncertainity_models_from_EPOCHS_cat",
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
]