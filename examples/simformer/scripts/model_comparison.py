from synference import Simformer_Fitter, load_unc_model_from_hdf5

grid_pop3_path = "/cosma7/data/dp276/dc-harv3/work/sbi_grids/grid_Yggdrasil_Chab_Burst_SFH_0.001_z_14_logN_4.4_v1.hdf5"  # noqa: E501

fitter_pop3 = Simformer_Fitter.init_from_hdf5(
    "Yggdrasil_Chab_Burst_SFH_0.001_z_14_logN_4.4_v1", grid_pop3_path
)

bpass_path = "/cosma7/data/dp276/dc-harv3/work/sbi_grids/grid_BPASS_Chab_DenseBasis_SFH_0.01_z_14_logN_5.0_Calzetti_v3_multinode.hdf5"  # noqa: E501

fitter_bpass = Simformer_Fitter.init_from_hdf5(
    "BPASS_Chab_DenseBasis_SFH_0.01_z_14_logN_5.0_Calzetti_v3_multinode", bpass_path
)


noise_model_path = "/cosma/apps/dp276/dc-harv3/synference/priv/JOF_psfmatched_asinh_noise_model.h5"  # noqa: E501
noise_models = load_unc_model_from_hdf5(noise_model_path)
hst_filters = ["F435W", "F606W", "F775W", "F814W", "F850LP"]

nm = {f"JWST/NIRCam.{i}": j for i, j in noise_models.items() if i not in hst_filters}
nm.update({f"HST/ACS_WFC.{i}": j for i, j in noise_models.items() if i in hst_filters})


fa_input = dict(
    extra_features=["redshift"],
    empirical_noise_models=nm,
    include_errors_in_feature_array=True,
    scatter_fluxes=True,
    photometry_to_remove=[
        "CTIO/DECam.u",
        "CTIO/DECam.g",
        "CTIO/DECam.r",
        "CTIO/DECam.i",
        "CTIO/DECam.z",
        "CTIO/DECam.Y",
        "LSST/LSST.u",
        "LSST/LSST.g",
        "LSST/LSST.r",
        "LSST/LSST.i",
        "LSST/LSST.z",
        "LSST/LSST.Y",
        "PAN-STARRS/PS1.g",
        "PAN-STARRS/PS1.r",
        "PAN-STARRS/PS1.i",
        "PAN-STARRS/PS1.w",
        "PAN-STARRS/PS1.z",
        "PAN-STARRS/PS1.y",
        "Paranal/VISTA.Z",
        "Paranal/VISTA.Y",
        "Paranal/VISTA.J",
        "Paranal/VISTA.H",
        "Paranal/VISTA.Ks",
        "Subaru/HSC.g",
        "Subaru/HSC.r",
        "Subaru/HSC.i",
        "Subaru/HSC.z",
        "Subaru/HSC.Y",
        "CFHT/MegaCam.u",
        "CFHT/MegaCam.g",
        "CFHT/MegaCam.r",
        "CFHT/MegaCam.i",
        "CFHT/MegaCam.z",
        "Euclid/VIS.vis",
        "Euclid/NISP.Y",
        "Euclid/NISP.J",
        "Euclid/NISP.H",
        "HST/ACS_WFC.F475W",
        "HST/WFC3_IR.F105W",
        "JWST/NIRCam.F070W",
        "HST/WFC3_IR.F110W",
        "HST/WFC3_IR.F125W",
        "JWST/NIRCam.F140M",
        "HST/WFC3_IR.F140W",
        "HST/WFC3_IR.F160W",
        "JWST/NIRCam.F360M",
        "JWST/NIRCam.F430M",
        "JWST/NIRCam.F460M",
        "JWST/NIRCam.F480M",
        "JWST/MIRI.F560W",
        "JWST/MIRI.F770W",
        "JWST/MIRI.F1000W",
        "JWST/MIRI.F1130W",
        "JWST/MIRI.F1280W",
        "JWST/MIRI.F1500W",
        "JWST/MIRI.F1800W",
        "JWST/MIRI.F2100W",
        "JWST/MIRI.F2550W",
        "Spitzer/IRAC.I1",
        "Spitzer/IRAC.I2",
        "Spitzer/IRAC.I3",
        "Spitzer/IRAC.I4",
    ],
)


fitter_pop3.create_feature_array_from_raw_photometry(**fa_input)
fitter_bpass.create_feature_array_from_raw_photometry(**fa_input)


fitter_bpass.plot_histogram_feature_array()
fitter_pop3.plot_histogram_feature_array()
fitter_pop3.run_single_sbi(name_append="v1")

fitter_bpass.run_single_sbi(name_append="v1")
