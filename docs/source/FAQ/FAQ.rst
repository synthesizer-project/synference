Frequently Asked Questions
**************************

Here are some frequently asked questions about synference. If you have a question that is not answered here, please open an issue on the `GitHub repository <https://github.com/synthesizer-project/synference/issues>`_.

Q: How do I move a model trained with GPU to CPU for inference?
----------------------------------------------------------------

A: You can move a trained model to CPU by using the `to` method provided by PyTorch. Here is an example:

.. code-block:: python

    import torch

    # Assuming `trained_model` is your trained SBI model
    trained_model.to(torch.device('cpu'))

This will move all the model parameters to the CPU, allowing you to perform inference without a GPU.


How do I deal with low sampling acceptance?
-------------------------------------------

A: You will be warned if your sampling acceptance is low during inference. Low sampling acceptance means that the model is predicting posterior samples which are outside the prior proposal range. To deal with this, you can try the following:
* Increase the prior proposal range to ensure that the true parameters are within the prior support. For some parameters this can be reasonable, for example setting a slightly wider range for stellar mass.
* Train a better model - low sampling acceptance can indicate that the model is not accurately capturing the posterior distribution. You can try training a more complex model, or using more training data to improve the model's performance.
* You can check if a specific parameter is causing low sampling acceptance by looking at the acceptance per parameter during inference using the custom torch prior implemented in Synference. You can enable this by running `SBI_Fitter.create_priors(debug_sample_acceptance=True)` before inference. This will log the acceptance rate for each parameter, allowing you to identify any parameters that may be causing issues.

How do I load a previously trained model for inference?
-------------------------------------------------------

A: You can load a previously trained model using the `SBI_Fitter.load_saved_model` method. Here is an example:

.. code-block:: python

    from synference import SBI_Fitter

    # Load the saved model
    trained_model = SBI_Fitter.load_saved_model('path/to/saved/model', device='cpu')

This will load the model from the specified path and move it to the CPU for inference. You can then use this model to perform inference on new data. The model can be trained on GPU and then loaded on CPU for inference as shown above.

What other SBI tools are available for SED fitting and more generally for galaxy inference?
--------------------------------------------------------------------------------------------

There are a number of great tools out there for specific applications of SBI. This is not an exhaustive list (and feel free to suggest more), but below we've collated some of the useful tools we've found. Please cite the relevant papers if you use these tools. 

1. `SBIPIX <https://github.com/patriglesias/SBIPIX/>`_ - Published
   `here <https://ui.adsabs.harvard.edu/abs/2025arXiv250604336I/abstract>`_.
   Specifically for pixel-level inference from galaxy images, this
   package provides pretrained models as well as the ability to train
   your own models.
2. `SBI++ <https://github.com/wangbingjie/sbi_pp>`_ - Published
   `here <https://ui.adsabs.harvard.edu/abs/2023ApJ...952L..10W/abstract>`_,
   this tool demonstrates a SBI workflow for SED fitting which can deal
   with out of distribution noise and missing data.
3. `SBI-SFHs <https://github.com/patriglesias/SBI_SFHs>`_ - Published
   `here <https://www.aanda.org/articles/aa/abs/2024/09/aa49909-24/aa49909-24.html>`_
   this tool recovers galaxy SFHS from spectroscopy.


