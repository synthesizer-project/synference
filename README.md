Repo trakcing progress of basic Synthesizer-based SBI SED fitting. Specifically aiming to do model comparison through direct comparison of Bayesian evidence using LtU-ILI. Early work in progress.

Current features:

Can generate a grid of SEDs drawn from prior ranges on SFH parameters, emission models (dust laws, dust emission, Lyman-alpha emission etc), metallicity distributions. Can combine models drawn from different grids (e.g. Pop III and standard Pop I/Pop II stars and generate photometry and spectroscopy.

Can generate feature arrays and perform flexible fitting using LtU-ILI SBI fitting, and plot standard validation plots. 

To Do's:

1. More complex model generation (e.g. parameters dependent on other parameters. This is currently only supported to prevent stellar populations older than the age of the Universe.).
2. Variety of ways to implement the SBI model and hyperparameter optimization using Optuna.
3. Assesing performance of SBI model, both from test dataset and classifcation of out of domain photometry. Typical plots to assess model performance.
4. Comparison of models with e.g. different grids, IMFs etc or ensembling over them.
5. More Pop III models e.g. Nakajima & Maiolino 2023.
6. Generealization to other types of compiste models/inference of standard SED fitting parameters. 

