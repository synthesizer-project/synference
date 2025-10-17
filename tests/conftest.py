"""Fixtures for common test objects."""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from astropy.cosmology import Planck18
from synthesizer.emission_models import PacmanEmission
from synthesizer.emission_models.attenuation import Calzetti2000
from synthesizer.grid import Grid
from synthesizer.instruments import FilterCollection, Instrument
from synthesizer.parametric import SFH, Galaxy, Stars, ZDist
from unyt import Msun, Myr, unyt_array

from synference import (  # noqa E402
    CombinedBasis,
    GalaxyBasis,
    SBI_Fitter,
    calculate_muv,
    draw_from_hypercube,
    generate_sfh_basis,
)


@pytest.fixture
def test_dir():
    """Fixture to get the test directory path."""
    return Path(__file__).resolve().parent


@pytest.fixture
def grid_dir(test_dir):
    """Fixture to get the grid directory path."""
    return test_dir / "test_grids"


@pytest.fixture
def synthesizer_grid_dir(test_dir):
    """Fixture to get the synthesizer grid directory path and set environment variable."""
    synthesizer_grid_dir = test_dir / "synthesizer_grids"
    os.environ["SYNTHESIZER_GRID_DIR"] = str(synthesizer_grid_dir)
    return synthesizer_grid_dir


@pytest.fixture
def test_sbi_grid(grid_dir):
    """Fixture to create a test SBI grid for testing synference."""
    return f"{grid_dir}/sbi_test_grid.hdf5"


@pytest.fixture
def test_grid(synthesizer_grid_dir):
    """Fixture to create a test Grid object."""
    if not os.path.exists(f"{synthesizer_grid_dir}/test_grid.hdf5"):
        subprocess.run(
            [
                "synthesizer-download",
                "--test-grids",
                "--destination",
                f"{synthesizer_grid_dir}",
            ],
            check=True,
        )

    return Grid(grid_name="test_grid", grid_dir=synthesizer_grid_dir)


@pytest.fixture
def mock_instrument():
    """Fixture to create a mock Instrument object with JWST filters."""
    filter_codes = [
        "JWST/NIRCam.F070W",
        "JWST/NIRCam.F090W",
        "JWST/NIRCam.F115W",
        "JWST/NIRCam.F200W",
        "JWST/NIRCam.F277W",
        "JWST/NIRCam.F356W",
        "JWST/NIRCam.F444W",
    ]
    filterset = FilterCollection(filter_codes=filter_codes)
    instrument = Instrument("JWST", filters=filterset)
    return instrument


@pytest.fixture
def mock_emission_model(test_grid):
    """Fixture to create a mock TotalEmission model for testing."""
    return PacmanEmission(
        grid=test_grid,
        fesc=0.1,
        fesc_ly_alpha=0.1,
        dust_curve=Calzetti2000(),
        dust_emission=None,
    )


@pytest.fixture
def simple_sfh():
    """Fixture to create a simple SFH model."""
    return SFH.LogNormal(tau=0.5, peak_age=100 * Myr, max_age=300 * Myr)


@pytest.fixture
def simple_zdist():
    """Fixture to create a simple metallicity distribution."""
    return ZDist.DeltaConstant(log10metallicity=-1.0)


@pytest.fixture
def grid_basis_params(test_grid, mock_emission_model, mock_instrument, simple_sfh, simple_zdist):
    """Fixture to create parameters for GalaxyBasis with a grid."""
    return {
        "model_name": "test_basis",
        "redshifts": np.array([6.0, 7.0, 8.0]),
        "grid": test_grid,
        "emission_model": mock_emission_model,
        "sfhs": [simple_sfh],
        "metal_dists": [simple_zdist],
        "cosmo": Planck18,
        "galaxy_params": {"tau_v": [0.2, 0.3, 0.4]},
        "instrument": mock_instrument,
        "redshift_dependent_sfh": False,
        "build_grid": True,
    }


@pytest.fixture
def lhc_grid(lhc_prior, n_params=1e2):
    """Fixture to create a Latin Hypercube sample grid for testing."""
    return draw_from_hypercube(param_ranges=lhc_prior, N=int(n_params), rng=42)


@pytest.fixture
def lhc_prior():
    """Fixture to create a mock prior distribution for testing."""
    return {
        "redshift": (0.01, 10),
        "masses": (5, 11),
        "tau_v": (0, 2),
        "peak_age": (0, 0.99),
        "tau": (0.1, 1.5),
        "log_zmet": (-3, -1.39),  # max of grid (e.g. 0.04)
    }


@pytest.fixture
def test_sfh():
    """Fixture to create a simple SFH model for testing."""
    return SFH.LogNormal


@pytest.fixture
def test_zmet():
    """Fixture to create a simple metallicity distribution for testing."""
    return ZDist.DeltaConstant


@pytest.fixture
def lhc_basis_params(
    test_grid,
    mock_emission_model,
    mock_instrument,
    lhc_prior,
    lhc_grid,
    test_zmet,
    test_sfh,
    n_params=1e2,
):
    """Fixture to create parameters for GalaxyBasis with LHC sampling."""
    all_param_dict = lhc_grid

    Z_dists = [test_zmet(log10metallicity=log_z) for log_z in all_param_dict["log_zmet"]]
    sfh_param_arrays = np.vstack((all_param_dict["tau"], all_param_dict["peak_age"])).T

    sfh_models, _ = generate_sfh_basis(
        sfh_type=test_sfh,
        sfh_param_names=["tau", "peak_age_norm"],
        sfh_param_arrays=sfh_param_arrays,
        redshifts=np.array(all_param_dict["redshift"]),
        max_redshift=20,
        cosmo=Planck18,
        sfh_param_units=[None, None],
        iterate_redshifts=False,
        calculate_min_age=False,
    )

    return {
        "model_name": "test_lhc_basis",
        "redshifts": all_param_dict["redshift"],
        "grid": test_grid,
        "emission_model": mock_emission_model,
        "metal_dists": Z_dists,
        "cosmo": Planck18,
        "instrument": mock_instrument,
        "galaxy_params": {"tau_v": all_param_dict["tau_v"]},
        "redshift_dependent_sfh": True,
        "build_grid": False,
        "sfhs": sfh_models,
    }


@pytest.fixture
def test_parametric_galaxy(test_grid, simple_sfh, simple_zdist):
    """Fixture to create a mock Galaxy object for testing."""
    return Galaxy(
        stars=Stars(
            sf_hist=simple_sfh,
            metal_dist=simple_zdist,
            metallicities=test_grid.metallicities,
            log10ages=test_grid.log10ages,
            initial_mass=unyt_array(1e9, units=Msun),
            tau_v=0.2,
        ),
        redshift=7.0,
    )


@pytest.fixture
def test_parametric_galaxies(test_parametric_galaxy, Ngalaxies=10):
    """Fixture to create a list of mock Galaxy objects for testing."""
    return [test_parametric_galaxy for _ in range(Ngalaxies)]
