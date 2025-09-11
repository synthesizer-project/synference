"""Test suite for the GalaxySimulator class."""

import numpy as np
import pytest
from astropy.cosmology import Planck18 as cosmo
from synthesizer.parametric import SFH, ZDist
from unyt import Myr, nJy, uJy

from synference import DepthUncertaintyModel, GalaxySimulator
from synference.fixtures import (  # noqa E402
    grid_dir,
    mock_emission_model,
    mock_instrument,
    simple_zdist,
    test_grid,
)


@pytest.fixture
def test_simulator(test_grid, mock_instrument, mock_emission_model):
    """Fixture to create a test GalaxySimulator object."""
    simulator = GalaxySimulator(
        sfh_model=SFH.LogNormal,
        zdist_model=ZDist.DeltaConstant,
        grid=test_grid,
        instrument=mock_instrument,
        emission_model=mock_emission_model,
        emitter_params={"stellar": ["tau_v"]},
        emission_model_key=mock_emission_model.label,
        param_transforms={
            "max_age": ("max_age", lambda x: cosmo.age(x["redshift"]).to_value("Myr") * Myr)
        },
        param_units={"peak_age": Myr},
    )

    return simulator


class TestGalaxySimulator:
    """Test suite for the GalaxySimulator class."""

    def test_initialization(self, test_simulator):
        """Test the initialization of the GalaxySimulator."""
        simulator = test_simulator

        assert simulator.sfh_model == SFH.LogNormal
        assert simulator.zdist_model == ZDist.DeltaConstant

        assert simulator.num_emitter_params == 1

        print(simulator)

    def test_init_from_grid(self):
        """Test initializing GalaxySimulator from a grid file."""
        simulator = GalaxySimulator.from_grid(
            grid_path=f"{grid_dir}/sbi_test_grid.hdf5", override_synthesizer_grid_dir="test_grids/"
        )
        assert isinstance(simulator, GalaxySimulator)

    def test_init_with_noise(self, test_grid, mock_instrument, mock_emission_model):
        """Test initializing GalaxySimulator with noise models."""
        noise_models = {
            filter_name: DepthUncertaintyModel(
                depth_ab=29.5,
            )
            for filter_name in mock_instrument.filters.filter_codes
        }
        simulator = GalaxySimulator(
            sfh_model=SFH.LogNormal,
            zdist_model=ZDist.DeltaConstant,
            grid=test_grid,
            instrument=mock_instrument,
            emission_model=mock_emission_model,
            noise_models=noise_models,
            emission_model_key=mock_emission_model.label,
            emitter_params={"stellar": ["tau_v"]},
            param_transforms={
                "max_age": ("max_age", lambda x: cosmo.age(x["redshift"]).to_value("Myr") * Myr)
            },
            param_units={"peak_age": Myr},
        )
        assert simulator.noise_models == noise_models

        # test a single simulate with noise
        params = {
            "redshift": 0.5,
            "tau": 1.0,
            "peak_age": 200.0,
            "log10metallicity": -2.0,
            "tau_v": 1.0,
            "log_mass": 9.0,
        }
        phot = simulator.simulate(params)
        assert isinstance(phot, np.ndarray)
        assert np.isfinite(phot).all()

    def test_init_with_depths(self, test_grid, mock_instrument, mock_emission_model):
        """Test initializing GalaxySimulator with fixed depths."""
        depths = [29.0] * len(mock_instrument.filters)
        depths = np.array(depths)
        depths = 10 ** (-(depths - 23.9) / 2.5) * uJy
        simulator = GalaxySimulator(
            sfh_model=SFH.LogNormal,
            zdist_model=ZDist.DeltaConstant,
            grid=test_grid,
            instrument=mock_instrument,
            emission_model=mock_emission_model,
            depths=depths,
            emission_model_key=mock_emission_model.label,
            emitter_params={"stellar": ["tau_v"]},
            param_transforms={
                "max_age": ("max_age", lambda x: cosmo.age(x["redshift"]).to_value("Myr") * Myr)
            },
            param_units={"peak_age": Myr},
        )
        assert (simulator.depths == depths).all()

        # test a single simulate with depths
        params = {
            "redshift": 0.5,
            "tau": 1.0,
            "peak_age": 200.0,
            "log10metallicity": -2.0,
            "tau_v": 1.0,
            "log_mass": 9.0,
        }
        phot = simulator.simulate(params)
        assert isinstance(phot, np.ndarray)
        assert np.isfinite(phot).all()

        simulator.out_flux_unit = "AB"
        phot = simulator(params)

    def test_simulate(self, test_simulator):
        """Test the simulate method of the GalaxySimulator."""
        simulator = test_simulator

        num_samples = 10
        thetas = {
            "redshift": np.linspace(0.1, 1.0, num_samples),
            "tau": np.random.uniform(0.1, 2.0, num_samples),
            "peak_age": np.random.uniform(0.1, 10.0, num_samples),
            "log10metallicity": np.random.uniform(-4.0, -1.4, num_samples),
            "tau_v": np.random.uniform(0.1, 2.0, num_samples),
            "log_mass": np.random.uniform(7.0, 11.0, num_samples),
        }

        for i in range(num_samples):
            simulator.output_type = ["photo_fnu"]
            simulator.out_flux_unit = nJy
            params = {key: thetas[key][i] for key in thetas}
            phot = simulator.simulate(params)

            phot = simulator(params)
            assert isinstance(phot, np.ndarray), f"Phot is {type(phot)}"
            assert np.isfinite(phot).all()

            simulator.out_flux_unit = "AB"
            phot = simulator(params)
            assert np.isfinite(phot).all()

            simulator.normalize_method = "JWST/NIRCam.F444W"
            phot = simulator(params)
            assert np.isfinite(phot).all()

            simulator.output_type = ["lnu", "sfh", "photo_fnu", "fnu"]
            phot = simulator(params)
            assert isinstance(phot, dict), f"Phot is {type(phot)}"
