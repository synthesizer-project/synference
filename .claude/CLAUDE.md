# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Synference is a Python package for **simulation-based inference (SBI) SED fitting** of galaxy photometry/spectroscopy. It integrates [Synthesizer](https://synthesizer-project.github.io) (mock generation) with [LtU-ILI](https://ltu-ili.readthedocs.io/) (amortised posterior inference).

The three-stage workflow is:
1. **Library generation** — use `GalaxyBasis` to generate mock photometry/spectra with Synthesizer and save as HDF5
2. **Training** — use `SBI_Fitter` to load an HDF5 library, create feature arrays, and train a normalising flow
3. **Inference** — call `sample_posterior` / `recover_SED` on observed data

## Development Setup

```bash
pip install -e ".[dev,test]"
pre-commit install
# LtU-ILI must be installed separately (PyPI restriction):
pip install "ltu_ili@git+https://github.com/maho3/ltu-ili.git"
# Test data (small HDF5 library used by the test suite):
python -c "from synference.utils import download_test_data; download_test_data()"
```

## Commands

```bash
# Run full test suite
pytest

# Run a single test file
pytest tests/test_library.py

# Run a single test
pytest tests/test_library.py::test_name

# Lint (ruff) — auto-fix
ruff check --fix src/
ruff format src/

# Build docs (from docs/)
cd docs && make html
```

## Code Architecture

All source lives in `src/synference/`. The package exports everything from `__init__.py`.

| Module | Responsibility |
|---|---|
| `library.py` | Synthesizer-based mock generation. Key classes: `GalaxyBasis` (single-component library), `CombinedBasis` (merge multiple bases), `LibraryCreator` (low-level HDF5 builder), `GalaxySimulator` (on-the-fly simulator for online learning). Helper functions: `generate_sfh_basis`, `draw_from_hypercube`, `calculate_*` derived quantities (SFR, Muv, colour, β, D4000, etc.). |
| `sbi_runner.py` | Inference layer. `SBI_Fitter` wraps LtU-ILI: loads HDF5 library → `create_feature_array()` → `run_single_sbi()` / `run_ensemble_sbi()` → `sample_posterior()` / `recover_SED()`. Also contains `MissingPhotometryHandler` and `Simformer_Fitter`. Hyperparameter search uses Optuna via `SBICustomRunner`. |
| `noise_models.py` | Abstract base `UncertaintyModel` plus concrete implementations: `EmpiricalUncertaintyModel`, `AsinhEmpiricalUncertaintyModel`, `DepthUncertaintyModel`, `GeneralEmpiricalUncertaintyModel`. Models are serializable to/from HDF5. |
| `custom_runner.py` | `SBICustomRunner` — subclass of LtU-ILI's `SBIRunner` with an Optuna-based training loop and `CustomIndependentUniform` prior. |
| `utils.py` | Photometric utilities (`f_jy_to_asinh`, `f_jy_err_to_asinh`, etc.), HDF5 loading (`load_library_from_hdf5`), outlier detection, feature importance, MPI-aware logging, data download. |
| `simformer.py` | Experimental Simformer support (partially commented out). |

## Key Conventions

- **Linter/formatter**: `ruff` (line length 100, Google-style docstrings). `__init__.py` files are excluded from ruff. Run `ruff check --fix` before committing.
- **Notebooks**: committed clean (outputs stripped) via `nb-clean` pre-commit hook. Run `nb-clean clean notebook.ipynb` manually if needed.
- **HDF5 is the interchange format** between library generation and training — everything flows through `*.hdf5` files.
- **Units**: `unyt` throughout. Pay attention to flux units (Jy, nJy, uJy) and the asinh-magnitude transform used for features.
- **Pre-commit hooks** enforce merge-conflict checks, case-conflict checks, large-file prevention, ruff lint+format, and nb-clean.
- `synthesizer` and `ltu_ili` imports are guarded by `try/except` so the package can be partially imported when only one dependency is available.
- Tests require the Synthesizer test grid (auto-downloaded via `synthesizer-download --test-grids`) and the synference test data library (`download_test_data()`).
