[![workflow](https://github.com/synthesizer-project/sbifitter/actions/workflows/python-app.yml/badge.svg)](https://github.com/synthesizer-project/sbifitter/actions)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


Current features:

Can generate a grid of SEDs drawn from prior ranges on SFH parameters, emission models (dust laws, dust emission, Lyman-alpha emission etc), metallicity distributions. Can combine models drawn from different grids (e.g. Pop III and standard Pop I/Pop II stars and generate photometry and spectroscopy. Can draw priors on grid or LatinHypercube.

Can generate feature arrays and perform flexible fitting using LtU-ILI SBI fitting, and plot standard validation plots. Can optimize parameters with Optuna. Supports all model types offered by LtU-ILI (only those available through pytorch tested) including NPE, NLE, NRE as well as the sequential variants. Supports online learning with on-the-fly model generation through sequential learning. 
