Installation
************


Creating a Python Environment
#############################

For best results, we recommend installing Synference in a virtual environment. You can use `venv <https://docs.python.org/3/library/venv.html>`_ or `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ to create a virtual environment.

Synference requires Python 3.10 or higher. It is recommended to use the latest stable version of Python.

.. code-block:: bash

    python3 -m venv /path_to_new_virtual_environment
    source /path_to_new_virtual_environment/bin/activate


Installing from PyPI (Recommended)
##################################

To install Synference, you can use install directly from the PyPI repository:

.. code-block:: bash

   pip install synference

Additionally you will currently have to install `LtU-ILI https://ltu-ili.readthedocs.io/en/latest/install.html` separately, as it can't be included as a dependency. This can be achieved simply as follows:

.. code-block:: bash

    git clone https://github.com/maho3/ltu-ili.git
    cd ltu-ili
    pip install ".[pytorch]"


Installing from Source 
###################### 

Alternatively, you can clone the repository and install it manually:

.. code-block:: bash

   git clone https://github.com/synthesizer-project/synference.git
   cd Synference
   pip install .

To install in editable mode, allowing you to make changes to the code if needed, use the following pip command:

.. code-block:: bash

    pip install -e .


Optional Dependencies
##################### 

Synference provides several optional dependency groups to cater to different use cases. These groups can be installed directly from PyPI using the following syntax:

.. code-block:: bash

    pip install synference[<group>] 

Or when installing from source: 

.. code-block:: bash

    pip install .[<group>]

The available groups are:

- **Development** (``dev``): Tools to help developing including linting and formatting.
- **Testing** (``test``): Frameworks and utilities for running tests.
- **Documentation** (``docs``): Packages required to build the project documentation.
- **simformer** (``simformer``): Dependencies for using the Simformer model within Synference.


For example, to install with development dependencies, run:

.. code-block:: bash

    pip install synference[dev]

Multiple optional dependency groups can be installed in one command. For instance, to install both the testing and documentation dependencies, run:

.. code-block:: bash

    pip install synference[test,docs]


Help
~~~~

If you wish to report an issue or get help with Synference, please `raise an issue <https://github.com/synthesizer-project/synference/issues>`_ on Github.
