############
Installation
############

pip2conda can be installed from conda-forge or PyPI.

From conda-forge (recommended)
===============================

For best results, please install from `conda-forge <https://conda-forge.org/>`_:

.. code-block:: shell

   conda install -c conda-forge pip2conda

From PyPI
=========

You can also install directly from PyPI:

.. code-block:: shell

   python -m pip install pip2conda

If you want to use the experimental rattler backend,
install the ``rattler`` optional extra as well:

.. code-block:: shell

   python -m pip install "pip2conda[rattler]"

Requirements
============

pip2conda supports Python 3.11 and later versions.
