from ltu_ili_testing import GalaxySimulator, SBI_Fitter, calculate_muv

import numpy as np
import torch
from synthesizer.emission_models import PacmanEmission, TotalEmission, EmissionModel, IntrinsicEmission
from synthesizer.emission_models.attenuation import PowerLaw, Calzetti2000
from synthesizer.grid import Grid
from synthesizer.parametric import SFH, ZDist
from synthesizer.instruments import Instrument, FilterCollection, Filter
from unyt import Myr, erg, Hz, s

device = "cuda" 

grid_dir = '/home/tharvey/work/synthesizer_grids/'
grid_name = 'bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps.hdf5'

grid = Grid(
    grid_name,
    grid_dir=grid_dir,
)

filter_codes = ["JWST/NIRCam.F090W","JWST/NIRCam.F115W","JWST/NIRCam.F150W","JWST/NIRCam.F162M",
                "JWST/NIRCam.F182M","JWST/NIRCam.F200W","JWST/NIRCam.F210M","JWST/NIRCam.F250M",
                "JWST/NIRCam.F277W","JWST/NIRCam.F300M","JWST/NIRCam.F335M","JWST/NIRCam.F356W",
                "JWST/NIRCam.F410M","JWST/NIRCam.F444W"]
filterset = FilterCollection(filter_codes)
instrument = Instrument('JWST', filters=filterset)

sfh = SFH.LogNormal
zdist = ZDist.DeltaConstant

priors = {
    'redshift': (5.0, 10.0),
    'log_mass': (7.0, 10.0),
    'log10metallicity': (-3.0, 0.3),
    'tau_v': (0.0, 1.5),
    'peak_age': (0, 500),
    'max_age': (500, 1000),
    'tau': (0.3, 1.5)
}

emission_model = TotalEmission(
    grid=grid,
    fesc=0.1,
    fesc_ly_alpha=0.1,
    dust_curve=Calzetti2000(), 
    dust_emission_model=None,
)

# This tells the emission model we will have a parameter called 'tau_v' on the stellar emitter.
emitter_params = {'stellar':['tau_v']}


simulator = GalaxySimulator(
    sfh_model=sfh,
    zdist_model=zdist,
    grid=grid,
    instrument=instrument,
    emission_model=emission_model,
    emission_model_key='total',
    emitter_params=emitter_params,
    param_units = {'peak_age':Myr, 'max_age':Myr},
    normalize_method=calculate_muv,
    output_type='photo_fnu',
    out_flux_unit='ABmag',
)

inputs = ['redshift', 'log_mass', 'log10metallicity', 'tau_v', 'peak_age', 'max_age', 'tau']

def run_simulator(params, return_type='tensor'):
    if isinstance(params, torch.Tensor):
        params = params.cpu().numpy()
    if isinstance(params, dict):
        params = {i: params[i] for i in inputs}
    elif isinstance(params, (list, tuple, np.ndarray)):
        params = {inputs[i]: params[i] for i in range(len(inputs))}
    input_dict = {i: params[i] for i in inputs}
    phot = simulator(params)
    if return_type == 'tensor':
        return torch.tensor(phot[np.newaxis, :], dtype=torch.float32).to(device)
    else:
        return phot
    
fitter = SBI_Fitter(
    name='online_test',
    simulator=run_simulator,
    parameter_names=inputs,
    raw_photometry_names=simulator.instrument.filters.filter_codes+['norm'],    
)

fitter.run_single_sbi(
    engine='SNPE',
    learning_type='online',
    override_prior_ranges=priors,
    num_simulations=5000,
    plot=False,
)