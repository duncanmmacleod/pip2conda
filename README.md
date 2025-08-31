# pip2conda

`pip2conda` is a tool to translate `pip`-style requirements into `conda`
requirements.

`pip2conda2` reads or generates the metadata for a project,
evaluating the build (if possible) and runtime requirements - including unpackging
extras and evaluating environment markers - before translating each requirement
into a conda-forge requirement suitable for installation with `conda/mamba`.

[![PyPI version](https://badge.fury.io/py/pip2conda.svg)](http://badge.fury.io/py/pip2conda)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/pip2conda.svg)](https://anaconda.org/conda-forge/pip2conda/)
[![License](https://img.shields.io/pypi/l/pip2conda.svg)](https://choosealicense.com/licenses/gpl-3.0/)
![Supported Python versions](https://img.shields.io/pypi/pyversions/pip2conda.svg)

## Installation

For best results, please install from [conda-forge](https://conda-forge.org/):

```shell
conda install -c conda-forge pip2conda
```

You can also install directly from PyPI:

```shell
python -m pip install pip2conda
```

## Basic Usage

Just execute `pip2conda` from the base of your project directory, or point
it at a wheel file for any project.
For example, running `pip2conda` in the base directory for its own
project repository does this:

```console
$ pip2conda
build
grayskull>=1.0.0
packaging
python>=3.10
requests
ruamel.yaml
setuptools-scm>=3.4.3
setuptools>=61.0
wheel
```

For more details, see the online documentation at

<https://pip2conda.readthedocs.io>
