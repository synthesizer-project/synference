
from synference import SBI_Fitter
from unyt import um
from ili import FCN
import numpy as np
from astropy.io import fits


fitter = SBI_Fitter.init_from_hdf5(
    "Spectra_BPASS_Chab_Continuity_SFH_0.01_z_14_logN_4.4_Calzetti_v4_multinode",
    "/cosma7/data/dp276/dc-harv3/work/sbi_grids/grid_spectra_BPASS_Chab_Continuity_SFH_0.01_z_14_logN_4.4_Calzetti_v4_multinode.hdf5" # noqa: E501
)
tab = fits.getdata('/cosma/apps/dp276/dc-harv3/synference/priv/jwst_nirspec_prism_disp.fits')
wavs = tab['WAVELENGTH'] * um
R = tab['R']

fitter.create_feature_array(flux_units="log10 nJy", crop_wavelength_range=(0.6, 5.0), 
                            resample_wavelengths=wavs, inst_resolution_wavelengths=wavs, 
                            inst_resolution_r=R, theory_r=np.inf, min_flux_value=-10)

n_hidden = [256, 128, 64]
embedding_net= FCN(n_hidden=n_hidden, n_input=len(fitter.feature_array[0]))


fitter.run_single_sbi(
    model_type='nsf',
    num_transforms=35,
    embedding_net=embedding_net,
    name_append='attempt2'
)