"""
Basic model for SBIFitter
"""

import os

from sbifitter import SBI_Fitter

file_path = os.path.dirname(os.path.realpath(__file__))
grid_folder = os.path.join(os.path.dirname(os.path.dirname(file_path)), "grids")
output_folder = os.path.join(os.path.dirname(os.path.dirname(file_path)), "models")

grid_path = f"""{grid_folder}/grid_BPASS_Chab_LogNormal_SFH_z=0.1_logN_4.0_Calzetti_v1.hdf5"""  # noqa: E501

fitter = SBI_Fitter.init_from_hdf5(
    "basic_model",
    grid_path,
    return_output=False,
    device="cpu",
)


fitter.create_feature_array_from_raw_photometry(
    extra_features=[], normalize_method=None, parameters_to_remove=["redshift"]
)


fitter.run_single_sbi(
    n_nets=3,
    backend="lampe",
    engine="NPE",
    name_append="_ensemble_lampe_nsf",
    stop_after_epochs=15,
    hidden_features=[180, 150, 120],
    learning_rate=0.0004,
    num_transforms=[16, 10, 6],
    model_type="nsf",
)
