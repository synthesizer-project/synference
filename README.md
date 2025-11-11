# Synference
<img src="https://raw.githubusercontent.com/synthesizer-project/synference/main/docs/source/gfx/synference_logo.png" align="right" width="140px"/>


[![workflow](https://github.com/synthesizer-project/synference/actions/workflows/python-app.yml/badge.svg)](https://github.com/synthesizer-project/synference/actions)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/synthesizer-project/synference/blob/main/docs/CONTRIBUTING.md)
[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Documentation Status](https://github.com/synthesizer-project/synference/actions/workflows/docs.yml/badge.svg)](https://synthesizer-project.github.io/synference/)

### Overview

Synference is a Python package designed to fit to perform simulation-based inference (SBI, also known as likelihood free inference) SED fitting. It integrates with [Synthesizer](https://synthesizer-project.github.io) for flexible and fast generation of mock spectra and photometry, and uses the [LtU-ILI](https://ltu-ili.readthedocs.io/) package for fast, amortised posterior inference.

### Key Features

- **Flexible Mock Generation**: Generate mock spectra and photometry using Synthesizer, allowing for full flexibility in almost every aspect of SED generation, including a wide range of post-processed stellar population synthesis grids.
- **Flexible Training**: Mock photometry creation is separated from the training of the inference model, allowing for flexible training strategies and the use of different inference models, as well as quickly switching between different feature sets - e.g. different filtersets, different noise models, etc.
- **Fast Inference**: Leverage LtU-ILI for fast, amortised posterior inference, enabling efficient fitting of complex models to data.

### Requirements

Synference requires Python 3.10 or higher. It has the following dependencies:

- [synthesizer](https://synthesizer-project.github.io) for mock generation
- [ltu-ili](https://ltu-ili.readthedocs.io/) for inference
- [torch](https://pytorch.org/) for deep learning models
- [numpy](https://numpy.org/) for numerical operations
- [scipy](https://www.scipy.org/) for scientific computing
- [scikit-learn](https://scikit-learn.org/) for machine learning utilities
- [optuna](https://optuna.org/) for hyperparameter optimization
- [sbi](https://sbi.readthedocs.io/) for simulation-based inference
- [h5py](https://www.h5py.org/) for HDF5 file handling
- [unyt](https://unyt.readthedocs.io/) for unit handling
- [matplotlib](https://matplotlib.org/) for plotting and visualization
- [tqdm](https://tqdm.github.io/) for progress bars
- [jax](https://jax.readthedocs.io/) for GPU acceleration (optional, for some inference models)

These dependencies will be automatically installed when you install Synference using pip.

### Installation

The easiest way to currently install Synference is to clone the repository and install it in editable mode:

```bash
git clone https://www.github.com/synthesizer-project/synference.git
cd synference
pip install -e .
```

### Getting Started

To get started with Synference, you can check out the [examples](examples/) directory for usage examples and tutorials. The examples cover various aspects of using Synference, including:

- Generating mock spectra and photometry with Synthesizer
- Training inference models with LtU-ILI
- Performing inference on mock data
- Evaluating the performance of inference models
- Visualizing results
- Using training models with real data.

The most basic usage, for creating a simple mock catalogue and training a model on it looks like this:

Firstly we setup the Synthesizer based model. More details on how to set up the Synthesizer model can be found in the [Synthesizer documentation](https://synthesizer-project.github.io/). Here we use a BPASS SPS grid, a lognormal star formation history, a single stellar metallicity and a simple emission model including Cloudy nebular emission but no dust reprocessing. The photometric filters used are common JWST/NIRCam wideband filters, but any filters supported by [SVO](https://svo2.cab.inta-csic.es/theory/fps/index.php) or loaded manually can be used. The model parameters are drawn from a Latin hypercube sampling of the parameter space, but this can be done in any way independent of Synference. All we are providing to the grid generation is a set of *10,000* galaxies with a range of stellar masses, redshifts, metallicities, and star formation histories, and these can be created in any way you like.

```python
from synthesizer.grid import Grid
from synthesizer.instruments import FilterCollection, Instrument
from synthesizer.parametric import SFH, ZDist
from synthesizer.emission_models import IntrinsicEmission
from unyt import Msun, Myr
from synference import draw_from_hypercube, generate_sfh_basis

N = 10_000  # Number of galaxies in the mock catalogue

# Set some parameter ranges for our model. Any parameters can be used if they are accepted by Synthesizer.
parameter_prior_ranges = {
    "log_stellar_mass": (8.0, 12.0),  # log10(M/Msun)
    "redshift": (0.0, 10.0),  # Redshift
    "log_zmet": (-4.0, -1.4),  # log10(Z)
    "peak_age": (0.0, 1000)*Myr, # Peak age of the SFH in Myr
    "tau": (0.2, 2) # Width of Lognormal SFH
}

# Draw samples from these ranges - this could be done with any sampling method, here we use a simple Latin hypercube sampling.
parameter_samples = draw_from_hypercube(parameter_prior_ranges, N=N, unlog_keys=['log_stellar_mass'])

# Chooose photometric filters and create instrument
filter_names = ['F090W', 'F115W', 'F150W', 'F200W', 'F277W', 'F356W', 'F444W']
filter_names = [f'JWST/NIRCam.{filter}' for filter in filter_names]
instrument = Instrument('JWST', filters=FilterCollection(filter_codes=filter_names))
# Synthesizer SPS grid - BPASS 2.2.1 post-processed with Cloudy
grid = Grid("bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps.hdf5")
# Synthesizer emission model
emission_model = IntrinsicEmission(grid=grid)

# Metallicity Distributions
Z_dists = [ZDist.DeltaConstant(log10metallicity=log_z) for log_z in parameter_samples["log_zmet"]]

# Star Formation History models
sfh_models, _ = generate_sfh_basis(
    sfh_type=SFH.LogNormal,
    sfh_param_names=["tau", "peak_age"],
    sfh_param_arrays=(parameter_samples["tau"], parameter_samples["peak_age"]),
    redshifts=parameter_samples["redshift"],
)
```

Then from this model we create a **library** of photometry and parameter pairs, which we can then use to train a model on.

```python
from synference import GalaxyBasis, SBI_Fitter

basis = GalaxyBasis(
    model_name=f"sps_test",
    redshifts=parameter_samples["redshift"],
    log_stellar_masses=parameter_samples["log_stellar_mass"],
    grid=grid,
    emission_model=emission_model,
    sfhs=sfh_models,
    instrument=instrument,
    metal_dists=Z_dists,
)

basis.create_mock_library(out_name=f'library_test', emission_model_key='intrinsic', overwrite=True, out_dir="./")
```

Finally we can train a model using the `SBI_Fitter` class, which will automatically create the feature array and run the training. We have full control over the model architecture and training parameters, and can easily switch between different model types (e.g. MAF, NSF, MDN) from the lampe and sbi backends.

Here we are use a single Masked Autoregressive Flow (MAF) model, with 90 hidden features, 4 transforms, and a learning rate of 1e-4. The model will be trained on the mock catalogue we created earlier, and the results will be saved.

```python
empirical_model_fitter = SBI_Fitter.init_from_hdf5(
    hdf5_path=f"./library_test.hdf5", model_name=f"sbi_test"
)
empirical_model_fitter.create_feature_array()

empirical_model_fitter.run_single_sbi(
    learning_rate=1e-4,
    hidden_features=90,
    num_transforms=4,
    model_type="maf",
    plot=True,
)
```

Finally, we can use the trained model to perform inference on new data. Here we provide a vector of observed data, and the model will sample from the posterior distribution and recover the SED.

```python
observed_data_vector = [30.2, 28.7, 28.5, 28.0, 27.5, 26.5, 26.2]  
posterior = empirical_model_fitter.sample_posterior(observed_data_vector)
empirical_model_fitter.recover_SED(observed_data_vector)

```

This is just a basic example to get you started. Synference is highly flexible and can be adapted to a wide range of use cases in simulation-based inference for SED fitting.

### Documentation

Documentation for Synference is available at [synthesizer-project.github.io/synference](https://synthesizer-project.github.io/synference/). The documentation includes installation instructions, tutorials, API references, and examples to help you get started with using Synference for your own projects.

### Contributing

We welcome contributions to Synference! If you have suggestions, bug reports, or would like to contribute code, please open an issue or submit a pull request on GitHub. Please see our [Code of Conduct](CODE_OF_CONDUCT.md) for more details on how to contribute and interact with the community.

### License
This project is licensed under the GNU General Public License v3.0 (GPLv3). See the [LICENSE](LICENSE) file for details. This means you can use, modify, and distribute the code freely, but any derivative works must also be open source and distributed under the same license. We warn users that this software is offered "as is", without any warranty or guarantee of fitness for a particular purpose. Synference is under active development, and therefore may change in the future.
