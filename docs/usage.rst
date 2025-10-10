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

Working with Dependency Groups
===============================

`pip2conda` supports both standard `PEP 735 <https://peps.python.org/pep-0735/>`__
dependency groups and custom `pip2conda`-specific dependency groups.

Standard Dependency Groups
---------------------------

Standard dependency groups are defined in the ``[dependency-groups]`` table
of your ``pyproject.toml``:

.. code-block:: toml

    [dependency-groups]
    test = ["pytest", "coverage"]
    docs = ["sphinx", "myst-parser"]

You can install specific groups using the ``-g/--dependency-group`` option:

.. code-block:: shell

    pip2conda -g test
    pip2conda --dependency-group docs

Or install all groups at once:

.. code-block:: shell

    pip2conda --all-groups

Custom Dependency Groups
-------------------------

For projects that use tools like ``uv`` which may fail when dependency groups
contain non-pip-installable packages, you can define custom dependency groups
that are only recognized by `pip2conda`:

.. code-block:: toml

    [tool.pip2conda.dependency-groups]
    conda = ["my-conda-only-package", "another-conda-package"]
    custom-test = ["pytest-conda", "conda-coverage"]

These custom groups work exactly like standard groups:

.. code-block:: shell

    pip2conda -g conda
    pip2conda -g custom-test

Mixing Standard and Custom Groups
----------------------------------

You can use both standard and custom dependency groups in the same project.
Custom groups take precedence when both define the same group name:

.. code-block:: toml

    [dependency-groups]
    test = ["pytest", "coverage"]

    [tool.pip2conda.dependency-groups]
    test = ["pytest-conda", "conda-coverage"]  # This overrides the standard test group
    conda = ["my-conda-only-package"]

Groups can also reference each other using ``include-group``:

.. code-block:: toml

    [dependency-groups]
    base = ["requests", "numpy"]

    [tool.pip2conda.dependency-groups]
    extended = [
        {"include-group" = "base"},
        "additional-conda-package",
    ]

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
