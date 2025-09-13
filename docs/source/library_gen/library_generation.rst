Library Generation
******************

The principle of simulation-based inference relies on a 'simulator', which takes in a set of parameters and produces a synthetic observation. In traditional SED fitting, this is often a forward model that generates a synthetic SED from a set of physical parameters, and typically computed on-the-fly during inference. Whilst SBI models can be trained using such a simulator (called 'online' or 'active' learning), this is computationally wasteful if you intend to train multiple SBI models with the same simulator configuration.

synference therefore provides tools to generate a 'library' of pre-computed simulations, which can then be used to train multiple SBI models. This is particularly useful for computationally expensive simulators, where generating a large library of simulations can take hours or days, but training an SBI model on that library takes only minutes to hours.

Normal use of synference uses the 'Synthesizer' package to generate synthetic observables from a set of input parameters. The Synthesizer package provides a high-level interface to a variety of underlying astrophysical simulation codes, and can generate a wide range of synthetic observables including photometric fluxes, spectra, and images. The full Synthesizer documentation can be found `here <https://synthesizer-project.github.io/synthesizer/>`_, but we provide a brief crash course below.

.. toctree::
   :maxdepth: 2

   synthesizer_crash_course
   basic_library_generation
   complex_library_generation
   multithreading_library_generation

