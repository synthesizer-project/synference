# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Synference performs simulation-based inference (SBI) SED fitting of galaxy photometry and spectroscopy. It couples [Synthesizer](https://synthesizer-project.github.io) (mock spectra/photometry generation) with [LtU-ILI](https://ltu-ili.readthedocs.io/) (amortised posterior inference, wrapping the `sbi` and `lampe` backends). The core design principle: **mock library generation is decoupled from feature engineering, which is decoupled from model training** — one expensive HDF5 library can feed many trained models with different filter sets, noise models, and normalizations.

## Commands

```bash
# Install (editable). ltu-ili and synthesizer cannot come from PyPI — install from git:
pip install -e .[dev]
pip install "ltu_ili@git+https://github.com/maho3/ltu-ili.git"
pip install "cosmos-synthesizer@git+https://github.com/synthesizer-project/synthesizer.git#egg=synthesizer"

# Test data (required before running tests):
synthesizer-download --test-grids --dust-grid
python -c "from synference.utils import download_test_data; download_test_data()"

# Tests
pytest                                    # full suite
pytest tests/test_sbi.py                  # one file
pytest tests/test_sbi.py -k feature_array # one test by keyword

# Lint / format (ruff, config in pyproject.toml: line-length 100, Google docstrings, D-rules enforced except in examples/)
ruff check --fix .
ruff format .
pre-commit run --all-files                # ruff + nb-clean + large-file/merge-conflict checks

# Docs (executes all notebooks; needs pandoc and .[docs] extras)
cd docs && make html SPHINXOPTS="-j auto"
```

`sbi_runner.py` (and therefore `SBI_Fitter`) is imported inside a try/except in `__init__.py` — if `ltu-ili`/`sbi`/`lampe` are missing the package still imports but only library-generation functions are available. If `SBI_Fitter` is mysteriously undefined, check that import error printed at import time.

## Architecture: the three-stage pipeline

### Stage 1 — Library generation (`src/synference/library.py`)

Builds an HDF5 "library" of (parameters → photometry/spectra) pairs by running the Synthesizer pipeline.

- **`GalaxyBasis`** — defines the model: SPS `Grid`, `EmissionModel`, list of SFH objects, metallicity distributions, redshifts, `Instrument` (filters). Two modes: pass N samples of every parameter (e.g. from `draw_from_hypercube` Latin hypercube draws), or set `build_library=True` to take the outer product of a few basis SFHs/redshifts/ZDists. `GalaxyBasis.create_mock_library()` is the main entry point (multiprocessing via `n_proc`, optional `multi_node` MPI/SLURM operation, batched in `batch_size` chunks).
- **`CombinedBasis`** — what `create_mock_library` delegates to. Galaxies are simulated at a filler mass and renormalized to the requested stellar masses afterwards; this also lets multiple `GalaxyBasis` objects (e.g. different SPS grids) be mass-weighted into composite galaxies via `combination_weights`.
- **Supplementary parameters** — derived quantities (`calculate_muv`, `calculate_sfr`, `calculate_beta`, `calculate_line_ew`, etc.; registry in `SUPP_FUNCTIONS`) are computed per-galaxy at library time and stored alongside the free parameters, so they can later be fitted or used as features. Extra ones are passed as `**extra_analysis_functions`.
- **`GalaxySimulator`** — a self-contained parameters→photometry callable (SFH class + ZDist class + grid + instrument + emission model). Used for online/sequential SBI training, SED recovery from posterior samples, and is serialized into the library so `SBI_Fitter.recreate_simulator_from_library()` can rebuild it.
- **`alt_parametrizations` / `parameter_transforms_to_save`** — mechanism for fitting in a different parametrization than Synthesizer uses (no lambdas; must be picklable).

**HDF5 library layout** (read by `utils.load_library_from_hdf5`): datasets `Grid/Photometry`, `Grid/Parameters`, `Grid/SupplementaryParameters`, optionally `Grid/Spectra`; root attrs `FilterCodes`, `ParameterNames`, `ParameterUnits`, `PhotometryUnits`, `SupplementaryParameterNames`. Raw photometry is stored noiseless in physical units (typically nJy).

Default output dir is `library_folder` = `<repo>/libraries/` — in this checkout that is a symlink to bulk data storage.

### Stage 2 — Feature engineering + training (`src/synference/sbi_runner.py`, ~9k lines, the heart of the package)

**`SBI_Fitter`** — instantiate with `SBI_Fitter.init_from_hdf5(model_name=..., hdf5_path=...)`, then:

1. **`create_feature_array_from_raw_photometry()`** (or `..._from_raw_spectra`, or plain `create_feature_array()`): converts the raw noiseless photometry grid into the training features. This is where all observational realism is injected: flux normalization (`normalize_method`), unit choice (AB mags, asinh mags, nJy), noise scattering via depths or empirical noise models (`scatter_fluxes=N` produces N noisy realizations per galaxy), errors as extra features, simulated missing bands, extra color features (`extra_features=['F090W - F115W']`, parsed by `FilterArithmeticParser`), and parameter add/drop/transformations.
2. **`run_single_sbi()`**: trains via LtU-ILI. Key knobs: `backend` ('sbi' or 'lampe'), `engine` ('NPE'/'NLE'/'NRE' + sequential variants), `model_type` ('mdn'/'maf'/'nsf'; pass a list for an ensemble), `n_nets`, `learning_type` ('offline' from the library, or 'online' with a `GalaxySimulator`), feature/target scalers (sklearn). Priors are built from the parameter array (`create_priors`). Saves to `<repo>/models/<name>/` by default: `*_posterior.pkl` (joblib), `*_params.pkl`, `*_summary.json`, plus any noise models as `*_empirical_noise_models.h5`.
3. **Evaluation**: `evaluate_model()`, `plot_coverage()`, `calculate_TARP()`, `calculate_PIT()`, `plot_diagnostics()`, `lc2st()`, `detect_misspecification()`.

Reload a trained model with `SBI_Fitter.load_saved_model(...)` / `load_model_from_pkl(...)`.

- **`optimize_sbi()`** runs Optuna hyperparameter searches via **`SBICustomRunner`** (`custom_runner.py`), a subclass of LtU-ILI's `SBIRunner` adding Optuna studies (optionally SQLite/MySQL-backed for multi-node searches), custom training loops, and `CustomUniform` torch priors with bijections. YAML study configs live in `examples/sbi/configs/`.
- **`Simformer_Fitter`** (subclass of `SBI_Fitter`, backed by the pure-PyTorch `src/synference/simformer/` subpackage) — score-diffusion transformer (Gloeckler et al. 2024) trained on the joint `[theta, x]` with per-example condition masks. Key difference from NPE: one trained model can condition on arbitrary subsets of features/parameters (boolean condition masks over nodes `[theta..., x...]`, True = observed), so it natively handles missing bands, acts as likelihood or posterior, and supports interval-constrained sampling via guidance (`sample_posterior_intervals`). The subpackage split: `nn.py` (tokenizer + transformer score net), `sde.py` (VE/VP SDEs; VE default), `masks.py` (condition/attention masks), `sampling.py` (fixed-step integrators + guidance + PF-ODE log-prob), `train.py` (`train_simformer`, config defaults), `model.py` (`SimformerModel` wrapper with `sample`/`sample_batched`/`sample_intervals`/`log_prob`/`save`/`load`).
- **`MissingPhotometryHandler`** — NPE alternative for missing data: KDE-imputes missing bands from nearest library neighbours (chi²-matched), then marginalizes posteriors over imputations.

### Stage 3 — Inference on real data

- `sample_posterior(obs_vector)` — posterior samples for one observation.
- `fit_catalogue(observations, columns_to_feature_names=..., flux_units=...)` — batch-fits an astropy Table / pandas DataFrame: builds features from catalogue columns (`create_features_from_observations` handles unit conversion and missing-data flags), runs out-of-distribution checks (pyod outlier ensembles + `test_in_distribution`), samples posteriors with per-row timeouts, returns quantiles appended to the input table, optionally recovers SEDs via the simulator.
- `recover_SED(obs_vector)` — pushes posterior samples back through the `GalaxySimulator` to reconstruct the SED implied by the fit.

### Noise models (`src/synference/noise_models.py`)

Class hierarchy under abstract `UncertaintyModel` (all serialize to/from HDF5 groups; `save_unc_model_to_hdf5` / `load_unc_model_from_hdf5`):

- `DepthUncertaintyModel` — analytic scatter from n-sigma depths.
- `EmpiricalUncertaintyModel` / `AsinhEmpiricalUncertaintyModel` / `GeneralEmpiricalUncertaintyModel` — flux-dependent error distributions binned from real catalogues (e.g. `create_uncertainty_models_from_EPOCHS_cat`); the General variant also models upper limits and SNR-dependent behaviour.
- `SpectralUncertaintyModel` — error kernels for spectra.

Noise models used in training are saved with the model so `fit_catalogue` applies consistent scattering at inference time. Pre-built models for COSMOS2020/COSMOS2025/JADES live in `priv/` (private, not shipped).

## Repository layout notes

- `src/synference/` — the whole package is 6 modules; `library.py` and `sbi_runner.py` contain nearly everything.
- `examples/` — the real documentation of usage patterns: `library_generation/scripts/` (incl. SLURM multi-node library builds), `sbi/` (training scripts, Optuna configs, SLURM), `online/`, `simformer/`, `paper/` (analysis notebooks).
- `models/` — trained model outputs (gitignored artifacts); `libraries/` — symlink to bulk library storage; `priv/` — private data/notebooks not for distribution.
- `tests/` — pytest suite; fixtures in `conftest.py` build small libraries from the Synthesizer test grid and expect the downloaded test data (see Commands).
- Docs are notebook-driven (`docs/source/`, nbsphinx); the docs CI build executes every notebook, so broken notebooks fail CI.

## Conventions

- Ruff enforces pydocstyle (Google convention) on `src/` — new public functions/classes need docstrings or CI fails. `examples/` is exempt from D-rules.
- Units are handled with `unyt` throughout; photometry conversions (AB ↔ Jy ↔ asinh) go through helpers in `utils.py` / `UncertaintyModel` staticmethods rather than ad-hoc math.
- Logging goes through the package logger from `utils.setup_mpi_named_logger("synference")` (MPI-rank aware); use `from . import logger`, not `print`.
- Anything serialized into libraries/models (parameter transforms, simulators) must be picklable — no lambdas.
