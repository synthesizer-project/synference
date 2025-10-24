Synference
^^^^^^^^^^^

Synference is an open-source python package for SED fitting of photometric and spectroscopic data using Simulation-Based Inference (SBI) methods. It is part of the broader Synthesizer project, which aims to provide tools for generating and analyzing synthetic astronomical observables.

For the Synthesizer project homepage, see https://synthesizer-project.github.io/.

This documentation provides a broad overview of the various components in synference and how they interact.
The `getting started guide <getting_started/getting_started>`_ contains download and installation instructions, as well as an overview of the code.

For detailed examples of what synference can do, take a look at the `examples <auto_examples/index>`_ page.
A full description of the code base is provided in the `API <API>`_.

Contents
^^^^^^^^

.. toctree::
   :maxdepth: 2
   
   getting_started/getting_started
   sbi/introduction_to_sbi
   library_gen/library_generation
   sbi_train/intro_sbi
   noise_modelling/creating_noise_model
   posterior_inference/intro
   advanced_topics/advanced_topics
   FAQ/FAQ
   notebook_examples/cookbook
   auto_examples/index
   API

Citation & Acknowledgement
--------------------------

Please cite **all** of the following papers (Harvey et al. 2025 (in prep.), `Lovell et al. 2025 <https://ui.adsabs.harvard.edu/abs/2025arXiv250803888L/abstract>`_, `Roper et al. 2025 <https://ui.adsabs.harvard.edu/abs/2025arXiv250615811R/abstract>`_, `Ho et al. 2024 <https://ui.adsabs.harvard.edu/abs/2024OJAp....7E..54H/abstract>`_) if you use synference in your research:

.. code-block:: bibtex

      @ARTICLE{2025arXiv250803888L,
             author = {{Lovell}, Christopher C. and {Roper}, William J. and {Vijayan}, Aswin P. and {Wilkins}, Stephen M. and {Newman}, Sophie and {Seeyave}, Louise},
              title = "{Synthesizer: a Software Package for Synthetic Astronomical Observables}",
            journal = {arXiv e-prints},
           keywords = {Instrumentation and Methods for Astrophysics, Cosmology and Nongalactic Astrophysics, Astrophysics of Galaxies},
               year = 2025,
              month = aug,
                eid = {arXiv:2508.03888},
              pages = {arXiv:2508.03888},
      archivePrefix = {arXiv},
             eprint = {2508.03888},
       primaryClass = {astro-ph.IM},
             adsurl = {https://ui.adsabs.harvard.edu/abs/2025arXiv250803888L},
            adsnote = {Provided by the SAO/NASA Astrophysics Data System}
      }

      @ARTICLE{2025arXiv250615811R,
         author = {{Roper}, Will J. and {Lovell}, Christopher and {Vijayan}, Aswin and {Wilkins}, Stephen and {Akins}, Hollis and {Berger}, Sabrina and {Sant Fournier}, Connor and {Harvey}, Thomas and {Iyer}, Kartheik and {Leonardi}, Marco and {Newman}, Sophie and {Pautasso}, Borja and {Perry}, Ashley and {Seeyave}, Louise and {Sommovigo}, Laura},
          title = "{Synthesizer: Synthetic Observables For Modern Astronomy}",
        journal = {arXiv e-prints},
       keywords = {Instrumentation and Methods for Astrophysics, Astrophysics of Galaxies},
           year = 2025,
          month = jun,
            eid = {arXiv:2506.15811},
          pages = {arXiv:2506.15811},
      archivePrefix = {arXiv},
             eprint = {2506.15811},
       primaryClass = {astro-ph.IM},
             adsurl = {https://ui.adsabs.harvard.edu/abs/2025arXiv250615811R},
            adsnote = {Provided by the SAO/NASA Astrophysics Data System}
      }

      @ARTICLE{2024OJAp....7E..54H,
      author = {{Ho}, Matthew and {Bartlett}, Deaglan J. and {Chartier}, Nicolas and {Cuesta-Lazaro}, Carolina and {Ding}, Simon and {Lapel}, Axel and {Lemos}, Pablo and {Lovell}, Christopher C. and {Makinen}, T. Lucas and {Modi}, Chirag and {Pandya}, Viraj and {Pandey}, Shivam and {Perez}, Lucia A. and {Wandelt}, Benjamin and {Bryan}, Greg L.},
      title = "{LtU-ILI: An All-in-One Framework for Implicit Inference in Astrophysics and Cosmology}",
      journal = {The Open Journal of Astrophysics},
      keywords = {Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Astrophysics of Galaxies, Computer Science - Machine Learning},
         year = 2024,
      month = jul,
      volume = {7},
         eid = {54},
      pages = {54},
         doi = {10.33232/001c.120559},
      archivePrefix = {arXiv},
            eprint = {2402.05137},
      primaryClass = {astro-ph.IM},
            adsurl = {https://ui.adsabs.harvard.edu/abs/2024OJAp....7E..54H},
            adsnote = {Provided by the SAO/NASA Astrophysics Data System}

Contributing
------------

Please see `here <https://github.com/synthesizer-project/synference/blob/main/docs/CONTRIBUTING.md>`_ for contribution guidelines.

Primary Contributors
---------------------

.. include:: ../../AUTHORS.rst

License
-------

synference is free software made available under the GNU General Public License v3.0. For details see the `LICENSE <https://github.com/synthesizer-project/synference/blob/main/LICENSE.md>`_.