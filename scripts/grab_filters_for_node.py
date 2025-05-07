from synthesizer.instruments import Instrument, FilterCollection
from ltu_ili_testing import generate_constant_R

filter_codes = [
    "JWST/NIRCam.F070W",
    "JWST/NIRCam.F090W",
    "JWST/NIRCam.F115W",
    "JWST/NIRCam.F140M",
    "JWST/NIRCam.F150W",
    "JWST/NIRCam.F162M",
    "JWST/NIRCam.F182M",
    "JWST/NIRCam.F200W",
    "JWST/NIRCam.F210M",
    "JWST/NIRCam.F250M",
    "JWST/NIRCam.F277W",
    "JWST/NIRCam.F300M",
    "JWST/NIRCam.F335M",
    "JWST/NIRCam.F356W",
    "JWST/NIRCam.F360M",
    "JWST/NIRCam.F410M",
    "JWST/NIRCam.F430M",
    "JWST/NIRCam.F444W",
    "JWST/NIRCam.F460M",
    "JWST/NIRCam.F480M",
]

# Consistent wavelength grid for both SPS grids and filters
new_wav = generate_constant_R(R=300)

filterset = FilterCollection(filter_codes, new_lam=new_wav)

filterset.write_filters('/cosma7/data/dp276/dc-harv3/work/ltu-ili_testing/scripts/filters/JWST.hdf5')