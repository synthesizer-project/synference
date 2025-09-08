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

