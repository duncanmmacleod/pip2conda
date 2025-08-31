###########
Basic Usage
###########

pip2conda translates pip-style requirements into conda requirements by reading
or generating metadata for a project, evaluating build and runtime requirements,
and translating each requirement into a conda-forge compatible format.

Simple Project Analysis
=======================

Run ``pip2conda`` from the base of your project directory:

.. code-block:: shell

    pip2conda

.. admonition:: Collecting requirements for the current project

    .. code-block:: console

        $ pip2conda
        grayskull>=1.0.0
        packaging>=25.0
        python-build>=1.0.0
        python>=3.10
        requests>=2.32.5
        ruamel.yaml>=0.18.15
        setuptools-scm>=3.4.3
        setuptools>=61.0
        wheel
        wheel>=0.45.1

The tool will analyze your project's metadata and output the conda-compatible
requirements.

Analyzing Wheel Files
=====================

You can also point pip2conda at a wheel file for any project:

.. code-block:: shell

   pip2conda path/to/package.whl

Converting requirements.txt
===========================

To convert an existing ``requirements.txt`` file into a conda ``environment.yml`` 
file, use the ``-r/--requirements`` option:

.. code-block:: shell

   pip2conda -r ./requirements.txt

This will read the requirements from the specified file and output the
conda-compatible equivalents.

Output Formats
==============

By default, pip2conda outputs requirements in a simple list format suitable
for use with conda. The tool can also generate YAML format output for
conda environment files.


.. admonition:: Example: YAML output

    .. code-block:: shell
        :caption: Write output in YAML format
        
        pip2conda -o requirements.yaml

    .. code-block:: yaml
        :caption: requirements.yaml

        channels:
        - conda-forge
        dependencies:
        - grayskull>=1.0.0
        - packaging>=25.0
        - python-build>=1.0.0
        - python>=3.10
        - requests>=2.32.5
        - ruamel.yaml>=0.18.15
        - setuptools-scm>=3.4.3
        - setuptools>=61.0
        - wheel
        - wheel>=0.45.1

Command Line Options
====================

For a full list of available options, run:

.. code-block:: shell

   pip2conda --help
