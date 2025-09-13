"""
Online learning example with a simulator
========================================

Here we setup a `GalaxySimulator` to generate samples from our model,
and train an online model where we condition the posterior to a specific observation.

"""

import os

import numpy as np
import torch
from synthesizer.emission_models import TotalEmission
from synthesizer.emission_models.attenuation import Calzetti2000
from synthesizer.grid import Grid
from synthesizer.instruments import FilterCollection, Instrument
from synthesizer.parametric import SFH, ZDist
from unyt import Myr

from synference import GalaxySimulator, SBI_Fitter, calculate_muv

device = "cpu"


grid_dir = os.environ["SYNTHESIZER_GRID_DIR"]
dir_path = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(os.path.dirname(os.path.dirname(dir_path)), "grids/")

grid_name = "bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps.hdf5"

grid = Grid(
    grid_name,
    grid_dir=grid_dir,
)

filter_codes = [
    "JWST/NIRCam.F090W",
    "JWST/NIRCam.F115W",
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
    "JWST/NIRCam.F410M",
    "JWST/NIRCam.F444W",
]
filterset = FilterCollection(filter_codes)
instrument = Instrument("JWST", filters=filterset)

sfh = SFH.LogNormal
zdist = ZDist.DeltaConstant

priors = {
    "redshift": (5.0, 10.0),
    "log_mass": (7.0, 10.0),
    "log10metallicity": (-3.0, 0.3),
    "tau_v": (0.0, 1.5),
    "peak_age": (0, 500),
    "max_age": (500, 1000),
    "tau": (0.3, 1.5),
}

emission_model = TotalEmission(
    grid=grid,
    fesc=0.1,
    fesc_ly_alpha=0.1,
    dust_curve=Calzetti2000(),
    dust_emission_model=None,
)

# This tells the emission model we will have a parameter called 'tau_v'
# on the stellar emitter.
emitter_params = {"stellar": ["tau_v"]}


simulator = GalaxySimulator(
    sfh_model=sfh,
    zdist_model=zdist,
    grid=grid,
    instrument=instrument,
    emission_model=emission_model,
    emission_model_key="total",
    emitter_params=emitter_params,
    param_units={"peak_age": Myr, "max_age": Myr},
    normalize_method=calculate_muv,
    output_type="photo_fnu",
    out_flux_unit="ABmag",
)

inputs = [
    "redshift",
    "log_mass",
    "log10metallicity",
    "tau_v",
    "peak_age",
    "max_age",
    "tau",
]


# Create a simulator function
def run_simulator(params, return_type="tensor"):
    if isinstance(params, torch.Tensor):
        params = params.cpu().numpy()
    if isinstance(params, dict):
        params = {i: params[i] for i in inputs}
    elif isinstance(params, (list, tuple, np.ndarray)):
        params = {inputs[i]: params[i] for i in range(len(inputs))}

    phot = simulator(params)
    if return_type == "tensor":
        return torch.tensor(phot[np.newaxis, :], dtype=torch.float32).to(device)
    else:
        return phot


# Create our fitter and pass in the simulator.
fitter = SBI_Fitter(
    name="online_test",
    simulator=run_simulator,
    parameter_names=inputs,
    raw_observation_names=simulator.instrument.filters.filter_codes + ["norm"],
)

# Run our SBI.
fitter.run_single_sbi(
    engine="SNPE",
    learning_type="online",
    override_prior_ranges=priors,
    num_simulations=5000,
    plot=False,
)
