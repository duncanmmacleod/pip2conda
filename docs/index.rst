.. include:: ../README.md
   :parser: myst_parser.sphinx_

.. #########
.. pip2conda
.. #########

.. .. image:: https://badge.fury.io/py/pip2conda.svg
   :target: https://badge.fury.io/py/pip2conda
   :alt: pip2conda PyPI release badge
.. .. image:: https://img.shields.io/pypi/l/pip2conda.svg
   :target: https://choosealicense.com/licenses/gpl-3.0/
   :alt: pip2conda license

.. ``pip2conda`` is a tool to translate `pip` requirements into `conda` requirements.

.. It parses build requirements from ``pyproject.toml`` files, then runtime and
.. extra requirements from ``setup.cfg``, including unpackging extras and
.. evaluating environment markers, before matching translating each requirement
.. into a conda-forge requirement suitable for installation with `conda/mamba`.

.. ============
.. Installation
.. ============

.. .. code-block:: bash

    python -m pip install pip2conda
