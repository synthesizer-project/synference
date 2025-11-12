Posterior Inference
*******************

In this section, we provide an overview of the various methods for performing posterior inference in Synference. Posterior inference is a crucial step in the simulation-based inference workflow, once an SBI model has been trained. It involves using the trained model to infer the posterior distribution of model parameters given observed data.

We can also use the trained SBI model alongside our simulator to recover the predicted observables for our posterior samples, allowing us to assess the quality of our inference, and assess how well our model is able to reproduce the observed data.

.. toctree::
   :maxdepth: 1

   catalogue_fitting
   sed_recovery
