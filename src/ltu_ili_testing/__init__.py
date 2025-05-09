# -*- coding: utf-8 -*-

from .grid import (
    generate_sfh_grid, generate_metallicity_distribution, generate_emission_models, 
    generate_sfh_basis, GalaxyBasis, generate_constant_R, CombinedBasis,
    calculate_muv, draw_from_hypercube
)

try:
    from .sbi import SBI_Fitter
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
    "SBI_Fitter",
]