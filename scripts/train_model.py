from ltu_ili_testing import SBI_Fitter


grid_path = '/home/tharvey/work/output/output.hdf5'

fitter = SBI_Fitter.init_from_hdf5('test', grid_path)

fitter.create_feature_array_from_raw_photometry()

fitter.run_single_sbi()