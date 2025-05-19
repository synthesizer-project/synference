import os
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from unyt import unyt_array, Msun, Myr
from synthesizer.parametric import SFH, ZDist, Stars, Galaxy
from synthesizer.instruments import Instrument, FilterCollection
from synthesizer.emission_models import EmissionModel, TotalEmission, IncidentEmission
from synthesizer.grid import Grid
from astropy.cosmology import Planck18

from ltu_ili_testing import GalaxyBasis, CombinedBasis


# Fixtures for common test objects
@pytest.fixture
def mock_grid():
    grid = MagicMock(spec=Grid)
    grid.log10ages = np.linspace(6, 10, 20)
    grid.metallicity = np.array([0.0001, 0.001, 0.01, 0.02])
    grid.wavelength = np.logspace(3, 5, 100)
    return grid


@pytest.fixture
def mock_instrument():
    filter_codes = [
        "JWST/NIRCam.F070W",
        "JWST/NIRCam.F200W",
        "JWST/NIRCam.F356W",
        "JWST/NIRCam.F444W",
    ]
    filterset = MagicMock(spec=FilterCollection)
    filterset.filter_codes = filter_codes
    instrument = MagicMock(spec=Instrument)
    instrument.filters = filterset
    instrument.label = "JWST"
    return instrument


@pytest.fixture
def mock_emission_model():
    model = MagicMock(spec=EmissionModel)
    model.set_per_particle = MagicMock()
    return model


@pytest.fixture
def simple_sfh():
    return SFH.LogNormal(tau=0.5, peak_age=100 * Myr, max_age=300 * Myr)


@pytest.fixture
def simple_zdist():
    return ZDist.DeltaConstant(log10metallicity=-1.0)


@pytest.fixture
def galaxy_basis_params(mock_grid, mock_emission_model, mock_instrument, simple_sfh, simple_zdist):
    return {
        "model_name": "test_basis",
        "redshifts": np.array([6.0, 7.0, 8.0]),
        "grid": mock_grid,
        "emission_model": mock_emission_model,
        "sfhs": [simple_sfh],
        "metal_dists": [simple_zdist],
        "cosmo": Planck18,
        "instrument": mock_instrument,
        "galaxy_params": {"tau_v": 0.5},
        "stellar_masses": unyt_array([1e9, 1e10], Msun),
        "redshift_dependent_sfh": False,
    }


@pytest.fixture
def mock_pipeline():
    pipeline_mock = MagicMock()
    pipeline_mock.add_analysis_func = MagicMock()
    pipeline_mock.add_galaxies = MagicMock()
    pipeline_mock.get_spectra = MagicMock()
    pipeline_mock.get_observed_spectra = MagicMock()
    pipeline_mock.get_photometry_luminosities = MagicMock()
    pipeline_mock.get_photometry_fluxes = MagicMock()
    pipeline_mock.run = MagicMock()
    pipeline_mock.write = MagicMock()
    return pipeline_mock


class TestGalaxyBasis:
    def test_init(self, galaxy_basis_params):
        """Test that GalaxyBasis initializes correctly with valid parameters."""
        basis = GalaxyBasis(**galaxy_basis_params)
        
        assert basis.model_name == "test_basis"
        assert np.array_equal(basis.redshifts, np.array([6.0, 7.0, 8.0]))
        assert basis.grid == galaxy_basis_params["grid"]
        assert basis.emission_model == galaxy_basis_params["emission_model"]
        assert basis.per_particle is False
        
        # Test SFH dictionary creation
        for z in basis.redshifts:
            assert z in basis.sfhs
            assert len(basis.sfhs[z]) > 0
    
    def test_process_priors(self):
        """Test that process_priors correctly handles prior distributions."""
        basis = GalaxyBasis(
            model_name="test_basis",
            redshifts=np.array([7.0]),
            grid=MagicMock(),
            emission_model=MagicMock(),
        )
        
        from scipy.stats import uniform
        prior_dict = {
            "prior": uniform,
            "size": 100,
            "loc": 0.0,
            "scale": 1.0,
            "units": None
        }
        
        result = basis.process_priors(prior_dict)
        assert len(result) == 100
        assert all(0 <= x <= 1.0 for x in result)
    
    @patch("ltu_ili_testing.sample_sfzh")
    def test_create_galaxy(self, mock_sample_sfzh, galaxy_basis_params, simple_sfh, simple_zdist):
        """Test that create_galaxy correctly creates a Galaxy object."""
        basis = GalaxyBasis(**galaxy_basis_params)
        
        # Mock the sampling function
        mock_stars = MagicMock(spec=Stars)
        mock_sample_sfzh.return_value = mock_stars
        
        galaxy = basis.create_galaxy(
            sfh=simple_sfh,
            redshift=7.0,
            metal_dist=simple_zdist,
            base_mass=1e9 * Msun,
            stellar_mass=unyt_array([1e9, 1e10], Msun)
        )
        
        assert isinstance(galaxy, Galaxy)
        assert galaxy.redshift == 7.0
        assert mock_sample_sfzh.called
        
        # Test with single mass
        galaxy = basis.create_galaxy(
            sfh=simple_sfh,
            redshift=7.0,
            metal_dist=simple_zdist,
            base_mass=1e9 * Msun,
            stellar_mass=1e9 * Msun
        )
        
        assert isinstance(galaxy, Galaxy)
        assert galaxy.redshift == 7.0
        assert not mock_sample_sfzh.called  # Should use param_stars directly
    
    def test_create_galaxies(self, galaxy_basis_params):
        """Test that create_galaxies correctly creates multiple galaxies."""
        basis = GalaxyBasis(**galaxy_basis_params)
        
        # Mock create_galaxy to avoid SPS calculations
        basis.create_galaxy = MagicMock()
        mock_galaxy = MagicMock(spec=Galaxy)
        mock_galaxy.all_params = {}
        basis.create_galaxy.return_value = mock_galaxy
        
        galaxies = basis.create_galaxies(stellar_masses=unyt_array([1e9], Msun))
        
        assert len(galaxies) > 0
        assert basis.create_galaxy.called
        
        # Check that varying_param_names and fixed_param_names are populated
        assert hasattr(basis, "varying_param_names")
        assert hasattr(basis, "fixed_param_names")
    
    @patch("ltu_ili_testing.Pipeline")
    def test_process_galaxies(self, mock_pipeline_class, galaxy_basis_params, mock_pipeline):
        """Test that process_galaxies correctly processes galaxies and returns a pipeline."""
        mock_pipeline_class.return_value = mock_pipeline
        
        basis = GalaxyBasis(**galaxy_basis_params)
        basis.all_parameters = {"redshift": [6.0, 7.0], "tau": [0.1, 0.2]}
        
        mock_galaxies = [MagicMock(spec=Galaxy) for _ in range(3)]
        
        pipeline = basis.process_galaxies(
            galaxies=mock_galaxies,
            out_name="test_output.hdf5",
            n_proc=2,
            verbose=0,
            save=True
        )
        
        assert pipeline == mock_pipeline
        mock_pipeline_class.assert_called_once()
        mock_pipeline.add_galaxies.assert_called_once_with(mock_galaxies)
        assert mock_pipeline.get_spectra.called
        assert mock_pipeline.get_observed_spectra.called
        assert mock_pipeline.get_photometry_luminosities.called
        assert mock_pipeline.get_photometry_fluxes.called
        assert mock_pipeline.run.called
        assert mock_pipeline.write.called


class TestCombinedBasis:
    @pytest.fixture
    def mock_galaxy_bases(self, galaxy_basis_params, mock_emission_model):
        """Create two mock GalaxyBasis objects."""
        # Create first basis
        basis1 = MagicMock(spec=GalaxyBasis)
        basis1.model_name = "PopII"
        basis1.instrument = galaxy_basis_params["instrument"]
        basis1.varying_param_names = ["tau", "metallicity"]
        basis1.fixed_param_names = ["fesc"]
        
        # Create second basis with different emission model
        basis2 = MagicMock(spec=GalaxyBasis)
        basis2.model_name = "PopIII"
        basis2.instrument = galaxy_basis_params["instrument"]
        basis2.varying_param_names = ["min_age", "max_age"]
        basis2.fixed_param_names = ["metallicity"]
        
        return [basis1, basis2]
    
    @pytest.fixture
    def combined_basis_params(self, mock_galaxy_bases):
        """Parameters for CombinedBasis initialization."""
        return {
            "bases": mock_galaxy_bases,
            "total_stellar_masses": unyt_array([1e9, 1e10], Msun),
            "redshifts": np.array([6.0, 7.0, 8.0]),
            "base_emission_model_keys": ["total", "incident"],
            "combination_weights": np.array([[0.7, 0.3], [0.5, 0.5]]),
            "out_name": "test_combined",
            "out_dir": "./test_output/",
            "base_mass": 1e9 * Msun
        }
    
    def test_init(self, combined_basis_params):
        """Test that CombinedBasis initializes correctly."""
        combined = CombinedBasis(**combined_basis_params)
        
        assert combined.bases == combined_basis_params["bases"]
        assert np.array_equal(combined.redshifts, combined_basis_params["redshifts"])
        assert np.array_equal(combined.combination_weights, combined_basis_params["combination_weights"])
        assert combined.out_name == "test_combined"
        assert combined.base_mass == 1e9 * Msun
    
    def test_process_bases(self, combined_basis_params, tmp_path, monkeypatch):
        """Test that process_bases correctly processes all bases."""
        # Mock file path operations
        monkeypatch.setattr(os.path, "exists", lambda x: False)
        monkeypatch.setattr(os, "makedirs", lambda x: None)
        
        combined = CombinedBasis(**combined_basis_params)
        
        # Mock the create_galaxies and process_galaxies methods
        for basis in combined.bases:
            basis.create_galaxies = MagicMock(return_value=[MagicMock(spec=Galaxy)])
            basis.process_galaxies = MagicMock(return_value=MagicMock())
        
        # Mock h5py.File
        mock_h5_file = MagicMock()
        mock_h5_file.__enter__ = MagicMock(return_value=mock_h5_file)
        mock_h5_file.__exit__ = MagicMock(return_value=None)
        mock_h5_file.attrs = {}
        
        with patch("h5py.File", return_value=mock_h5_file):
            combined.process_bases(overwrite=True)
        
        # Check that create_galaxies and process_galaxies were called for each basis
        for basis in combined.bases:
            basis.create_galaxies.assert_called_once()
            basis.process_galaxies.assert_called_once()
    
    @patch("h5py.File")
    def test_load_bases(self, mock_h5py_file, combined_basis_params):
        """Test that load_bases correctly loads the pipeline outputs."""
        combined = CombinedBasis(**combined_basis_params)
        
        # Mock h5py file structure
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.__exit__.return_value = None
        mock_h5py_file.return_value = mock_file
        
        # Mock attributes in file
        mock_file.attrs = {"varying_param_names": ["tau", "metallicity"], "fixed_param_names": ["fesc"]}
        
        # Mock galaxy group
        galaxies_group = MagicMock()
        property_data = MagicMock()
        property_data.attrs = {"Units": "Msun"}
        property_data.__getitem__ = MagicMock(return_value=property_data)
        property_data.return_value = np.array([1e9, 1e10])
        
        galaxies_group.__getitem__ = MagicMock(side_effect=lambda x: {
            "redshift": property_data,
            "mass": property_data,
            "Stars": {
                "Spectra": {
                    "SpectralFluxDensities": {
                        "total": np.ones((10, 100)),
                        "incident": np.ones((10, 100))
                    }
                },
                "Photometry": {
                    "Fluxes": {
                        "total": {"JWST": {"F070W": np.ones(10)}},
                        "incident": {"JWST": {"F070W": np.ones(10)}}
                    }
                }
            }
        }[x])
        
        galaxies_group.keys = MagicMock(return_value=["redshift", "mass", "Stars"])
        mock_file.__getitem__ = MagicMock(side_effect=lambda x: {
            "Galaxies": galaxies_group,
            "Instruments": {
                "JWST": {
                    "Filters": {
                        "Header": {
                            "Wavelengths": np.logspace(3, 5, 100)
                        }
                    }
                }
            }
        }[x])
        
        # Call load_bases
        outputs = combined.load_bases()
        
        # Verify outputs
        assert "PopII" in outputs
        assert "PopIII" in outputs
        assert "properties" in outputs["PopII"]
        assert "observed_spectra" in outputs["PopII"]
        assert "wavelengths" in outputs["PopII"]
        assert "observed_photometry" in outputs["PopII"]
    
    @patch("h5py.File")
    def test_create_grid(self, mock_h5py_file, combined_basis_params):
        """Test that create_grid correctly creates a grid of SEDs."""
        combined = CombinedBasis(**combined_basis_params)
        
        # Mock load_bases
        mock_outputs = {
            "PopII": {
                "properties": {
                    "redshift": np.array([6.0, 7.0]),
                    "mass": unyt_array([1e9, 1e9], "Msun"),
                    "PopII/tau": np.array([0.1, 0.2])
                },
                "observed_spectra": np.ones((100, 2)),
                "wavelengths": np.logspace(3, 5, 100),
                "observed_photometry": {
                    "F070W": np.array([1.0, 2.0]),
                    "F200W": np.array([3.0, 4.0])
                }
            },
            "PopIII": {
                "properties": {
                    "redshift": np.array([6.0, 7.0]),
                    "mass": unyt_array([1e9, 1e9], "Msun"),
                    "PopIII/min_age": np.array([10, 20])
                },
                "observed_spectra": np.ones((100, 2)),
                "wavelengths": np.logspace(3, 5, 100),
                "observed_photometry": {
                    "F070W": np.array([5.0, 6.0]),
                    "F200W": np.array([7.0, 8.0])
                }
            }
        }
        combined.load_bases = MagicMock(return_value=mock_outputs)
        
        # Mock save_grid
        combined.save_grid = MagicMock()
        
        # Create the grid
        combined.create_grid(save=True, out_name="test_grid.hdf5")
        
        # Check that save_grid was called
        assert combined.save_grid.called
        
        # Check the structure of the grid passed to save_grid
        grid_dict = combined.save_grid.call_args[0][0]
        assert "photometry" in grid_dict
        assert "parameters" in grid_dict
        assert "parameter_names" in grid_dict
        assert "filter_codes" in grid_dict
    
    @patch("h5py.File")
    def test_save_grid(self, mock_h5py_file, combined_basis_params, tmp_path, monkeypatch):
        """Test that save_grid correctly saves the grid to a file."""
        # Mock os.path and os functions
        monkeypatch.setattr(os.path, "exists", lambda x: False)
        monkeypatch.setattr(os, "makedirs", lambda x: None)
        
        combined = CombinedBasis(**combined_basis_params)
        
        # Create test grid data
        grid_dict = {
            "photometry": np.ones((4, 100)),
            "parameters": np.ones((10, 100)),
            "parameter_names": ["redshift", "log_mass", "weight_fraction", "PopII/tau", "PopIII/min_age"],
            "filter_codes": ["F070W", "F200W", "F356W", "F444W"]
        }
        
        # Mock h5py File object
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.__exit__.return_value = None
        mock_file.create_group.return_value = MagicMock()
        mock_h5py_file.return_value = mock_file
        
        # Save the grid
        combined.save_grid(grid_dict, out_name="test_grid.hdf5")
        
        # Check that h5py.File was called with the correct parameters
        mock_h5py_file.assert_called_once()
        
        # Check that a group was created for the grid data
        mock_file.create_group.assert_called_once_with("Grid")
        
        # Check that datasets were created for photometry and parameters
        grid_group = mock_file.create_group.return_value
        assert grid_group.create_dataset.call_count == 2
        
        # Check that attributes were set
        assert mock_file.attrs.__setitem__.call_count == 2
        mock_file.attrs.__setitem__.assert_any_call("ParameterNames", grid_dict["parameter_names"])
        mock_file.attrs.__setitem__.assert_any_call("FilterCodes", grid_dict["filter_codes"])


# Integration tests for full pipeline
@pytest.mark.slow
def test_full_pipeline_integration(tmp_path):
    """Integration test for the full pipeline.
    
    This test creates a minimal version of the full pipeline to verify
    that all components work together correctly.
    """
    # Skip this test if we're not running integration tests
    pytest.skip("Integration test - skipping for normal test runs")
    
    # Create temporary directory for output
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    
    # Create minimal grid
    mock_grid = MagicMock(spec=Grid)
    mock_grid.log10ages = np.linspace(6, 10, 5)
    mock_grid.metallicity = np.array([0.0001, 0.02])
    mock_grid.wavelength = np.logspace(3, 5, 20)
    
    # Create minimal instrument
    filterset = MagicMock(spec=FilterCollection)
    filterset.filter_codes = ["JWST/NIRCam.F070W", "JWST/NIRCam.F200W"]
    instrument = MagicMock(spec=Instrument)
    instrument.filters = filterset
    instrument.label = "JWST"
    
    # Create minimal SFH and metallicity distribution
    sfh = SFH.LogNormal(tau=0.5, peak_age=100 * Myr, max_age=300 * Myr)
    zdist = ZDist.DeltaConstant(log10metallicity=-1.0)
    
    # Create minimal emission models
    emission_model1 = MagicMock(spec=TotalEmission)
    emission_model1.set_per_particle = MagicMock()
    
    emission_model2 = MagicMock(spec=IncidentEmission)
    emission_model2.set_per_particle = MagicMock()
    
    # Create bases
    basis1 = GalaxyBasis(
        model_name="PopII",
        redshifts=np.array([6.0]),
        grid=mock_grid,
        emission_model=emission_model1,
        sfhs=[sfh],
        metal_dists=[zdist],
        cosmo=Planck18,
        instrument=instrument,
        galaxy_params={"tau_v": 0.5},
        redshift_dependent_sfh=False
    )
    
    basis2 = GalaxyBasis(
        model_name="PopIII",
        redshifts=np.array([6.0]),
        grid=mock_grid,
        emission_model=emission_model2,
        sfhs=[sfh],
        metal_dists=[zdist],
        cosmo=Planck18,
        instrument=instrument,
        galaxy_params={},
        redshift_dependent_sfh=False
    )
    
    # Mock create_galaxies and process_galaxies
    basis1.create_galaxies = MagicMock(return_value=[MagicMock(spec=Galaxy)])
    basis1.process_galaxies = MagicMock()
    basis2.create_galaxies = MagicMock(return_value=[MagicMock(spec=Galaxy)])
    basis2.process_galaxies = MagicMock()
    
    # Create combined basis
    combined = CombinedBasis(
        bases=[basis1, basis2],
        total_stellar_masses=unyt_array([1e9], Msun),
        redshifts=np.array([6.0]),
        base_emission_model_keys=["total", "incident"],
        combination_weights=np.array([[0.7, 0.3]]),
        out_name="test_combined",
        out_dir=str(out_dir),
        base_mass=1e9 * Msun
    )
    
    # Mock load_bases
    mock_outputs = {
        "PopII": {
            "properties": {
                "redshift": np.array([6.0]),
                "mass": unyt_array([1e9], "Msun"),
                "PopII/tau": np.array([0.5])
            },
            "observed_spectra": np.ones((20, 1)),
            "wavelengths": np.logspace(3, 5, 20),
            "observed_photometry": {
                "F070W": np.array([1.0]),
                "F200W": np.array([2.0])
            }
        },
        "PopIII": {
            "properties": {
                "redshift": np.array([6.0]),
                "mass": unyt_array([1e9], "Msun"),
                "PopIII/metallicity": np.array([-1.0])
            },
            "observed_spectra": np.ones((20, 1)),
            "wavelengths": np.logspace(3, 5, 20),
            "observed_photometry": {
                "F070W": np.array([3.0]),
                "F200W": np.array([4.0])
            }
        }
    }
    combined.load_bases = MagicMock(return_value=mock_outputs)
    
    # Mock save_grid
    combined.save_grid = MagicMock()
    
    # Process bases and create grid
    combined.process_bases()
    combined.create_grid()
    
    # Check that all methods were called
    basis1.create_galaxies.assert_called_once()
    basis1.process_galaxies.assert_called_once()
    basis2.create_galaxies.assert_called_once()
    basis2.process_galaxies.assert_called_once()
    combined.load_bases.assert_called_once()
    combined.save_grid.assert_called_once()


# Tests for error handling and edge cases
class TestErrorHandling:
    def test_invalid_redshift_dependent_sfh(self, galaxy_basis_params):
        """Test that an error is raised when redshift_dependent_sfh is True but SFHs don't have redshift attributes."""
        params = galaxy_basis_params.copy()
        params["redshift_dependent_sfh"] = True
        
        with pytest.raises(ValueError, match="SFH must have a redshift attribute"):
            GalaxyBasis(**params)
    
    def test_missing_stellar_masses(self, galaxy_basis_params):
        """Test that an error is raised when stellar_masses is not provided."""
        params = galaxy_basis_params.copy()
        params.pop("stellar_masses")
        
        basis = GalaxyBasis(**params)
        
        with pytest.raises(ValueError, match="stellar_masses must be provided"):
            basis.create_galaxies()
    
    def test_mismatched_filters(self, combined_basis_params):
        """Test that an error is raised when bases have different filters."""
        combined = CombinedBasis(**combined_basis_params)
        
        # Change filters for the second basis
        combined.bases[1].instrument.filters.filter_codes = ["Different/Filter.1", "Different/Filter.2"]
        
        with pytest.raises(ValueError, match="has different filters"):
            combined.create_grid()
    
    def test_nonexistent_emission_model_key(self, combined_basis_params):
        """Test that an error is raised when an emission model key doesn't exist."""
        combined = CombinedBasis(**combined_basis_params)
        
        # Mock load_bases to return data with missing emission model key
        mock_outputs = {
            "PopII": {
                "properties": {"redshift": np.array([6.0]), "mass": unyt_array([1e9], "Msun")},
                "observed_spectra": np.ones((20, 1)),
                "wavelengths": np.logspace(3, 5, 20),
                "observed_photometry": {"F070W": np.array([1.0])}
            },
            "PopIII": {
                "properties": {"redshift": np.array([6.0]), "mass": unyt_array([1e9], "Msun")},
                "observed_spectra": np.ones((20, 1)),
                "wavelengths": np.logspace(3, 5, 20),
                "observed_photometry": {"F070W": np.array([3.0])}
            }
        }
        combined.load_bases = MagicMock(return_value=mock_outputs)
        
        # Change the emission model key to something that doesn't exist
        combined.base_emission_model_keys = ["nonexistent", "incident"]
        
        with pytest.raises(AssertionError):
            combined.create_grid()


# Run the tests

if __name__ == "__main__":
    pytest.main([__file__])
    # To run the tests, use the command:
    # pytest -v ltu-ili_testing/tests/basis_tests.py
