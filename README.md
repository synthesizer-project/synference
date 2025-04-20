Project aiming to implement some basic Synthesizer-based SBI SED fitting. Specifically aiming to do model comparison through direct comparison of Bayesian evidence using LtU-ILI. Early work in progress.


Current features:

Can generate a grid of SEDs drawn from prior ranges on SFH parameters, emission models (dust laws, dust emission, Lyman-alpha emission etc), metallicity distributions. Can combine models drawn from different grids (e.g. Pop III and standard Pop I/Pop II stars and generate photometry and spectroscopy.

To Do:

1. Saving of grid to HDF5, including parameter ranges and SEDs etc.
2. More complex model generation (e.g. parameters dependent on other parameters. This is currently only supported to prevent stellar populations older than the age of the Universe.).
3. All the SBI stuff - splitting into test/train, selecting features, normalizing models, generating tensor priors for fitted parameters etc.
4. Variety of ways to implement the SBI model and hyperparameter optimization using Optuna.
5. Assesing performance of SBI model, both from test dataset and classifcation of out of domain photometry. Typical plots to assess model performance.
6. Comparison of models with e.g. different grids, IMFs etc or ensembling over them.
7. More Pop III models e.g. Nakajima & Maiolino 2023. 

