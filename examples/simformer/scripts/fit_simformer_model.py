import astropy.units as u
import numpy as np
from astropy.table import Table

from synference import Simformer_Fitter

fitter = Simformer_Fitter.load_saved_model(
    model_name="simformer_v2_initial",
    grid_path="/cosma/apps/dp276/dc-harv3/synference/grids/grid_BPASS_Chab_DenseBasis_SFH_0.01_z_12_logN_5.0_CF00_v2.hdf5",
    model_file="/cosma/apps/dp276/dc-harv3/synference/models/simformer_v2",
)

cat_path = (
    "/cosma7/data/dp276/dc-harv3/work/catalogs/JADES-DR3-GS_MASTER_Sel-F277W+F356W+F444W_v13.fits"
)


# Deal with data:

cat = Table.read(cat_path, hdu="OBJECTS")
cat_sel = Table.read(cat_path, hdu="SELECTION")
mask = (
    cat_sel["Austin+25_EAZY_fsps_larson_zfree_0.32as"] == True  # noqa: E712
)  # & (cat_sel['5.60<z<6.50_EAZY_fsps_larson_zfree_0.32as'] == True)

cat = cat[mask]

cat.sort("MAG_APER_F444W_aper_corr")


def mag_cols_syntax(band):
    """Returns the syntax for the magnitude columns in the table."""
    return f"MAG_APER_{band}_aper_corr"


def flux_cols_syntax(band):
    """Returns the syntax for the flux columns in the table."""
    return f"FLUX_APER_{band}_aper_corr_Jy"


def magerr_cols_syntax(band):
    """Returns the syntax for the magnitude error columns in the table."""
    return f"loc_depth_{band}"


new_band_names = fitter.feature_names

bands = [
    band.split(".")[-1]
    for band in new_band_names
    if not (band.startswith("unc_") or band == "redshift")
]
new_table = Table()


for band in bands:
    new_table[mag_cols_syntax(band)] = cat[mag_cols_syntax(band)]  # [:, 0]
    err = (cat[magerr_cols_syntax(band)] * u.ABmag).to(u.nJy).value / 5
    phot_njy = (cat[flux_cols_syntax(band)] * u.Jy).to(u.nJy).value
    new_table[f"unc_{band}"] = np.abs(2.5 * err / (np.log(10) * phot_njy))
    sig = phot_njy / err
    mask = sig < 1
    new_table[f"unc_{band}"][mask] = 2.5 / np.log(10)
    new_table[mag_cols_syntax(band)][mask] = 40.0


# new_table["redshift"] = cat['input_redshift'] #redshift_50"]

conversion_dict = {mag_cols_syntax(band): new_band for band, new_band in zip(bands, new_band_names)}
conversion_dict.update(
    {f"unc_{band}": f"unc_{new_band}" for band, new_band in zip(bands, new_band_names)}
)

new_table = new_table[:5]

post_tab = fitter.fit_catalogue(
    new_table,
    columns_to_feature_names=conversion_dict,
    flux_units="AB",
    return_feature_array=False,
    check_out_of_distribution=False,
)

post_tab.write(
    "/cosma7/data/dp276/dc-harv3/work/catalogs/JADES-DR3-GS_MASTER_Sel-F277W+F356W+F444W_v13_fitted_simformer.fits"
)
