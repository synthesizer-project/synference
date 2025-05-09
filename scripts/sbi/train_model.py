from ltu_ili_testing import SBI_Fitter
from unyt import Jy
import numpy as np


grid_path = '/home/tharvey/work/output/BPASS_Chab_LogNorm_5_z_12_phot_grid.hdf5'

fitter = SBI_Fitter.init_from_hdf5('BPASS_Chab_LogNorm_5_z_12_phot_grid_test3', grid_path)

depths = 10**((np.array([30] * 20)-8.90)/-2.5) * Jy # 30 AB mag in all 22 filters


# Put 5 realizations of each flux into the fitter scattered with a Gaussian at fixed magnitude. 
fitter.create_feature_array_from_raw_photometry()#scatter_fluxes=5, depths=depths)

# Plot a diagnostic showing distribution of parameters to recover. 
fitter.plot_histogram_parameter_array()

fitter.run_single_sbi(n_nets=2, engine='NPE', stop_after_epochs=20, model_type=['mdn', 'maf'], plot=True)

