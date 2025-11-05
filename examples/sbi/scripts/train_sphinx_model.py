from synference import SBI_Fitter
import numpy as np

fitter = SBI_Fitter.init_from_hdf5(
    "SPHINX_inference",
    "/cosma/apps/dp276/dc-harv3/synference/libraries/grid_SPHINX20.hdf5",
)

def log10floor(x, floor=-2):
    x = np.array(x)
    x[x < floor] = floor
    mask = ~np.isfinite(x)
    x[mask] = floor
    return np.log10(x)

fitter.create_feature_array_from_raw_photometry(parameter_transformations={'sfr_10':log10floor,'sfr_100':log10floor, 'mean_stellar_age_mass':log10floor}, parameters_to_remove=['sfr_3'])


fitter.run_single_sbi(
        name_append="nsf",
        model_type='nsf',
        custom_config_yaml="/cosma/apps/dp276/dc-harv3/synference/examples/sbi/best_params_2101.yaml",
)