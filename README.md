# pip2conda

`pip2conda` is a tool to translate `pip`-style requirements into conda
requirements, mainly for use in seeding test environments (e.g. in CI).

[![PyPI version](https://badge.fury.io/py/pip2conda.svg)](http://badge.fury.io/py/pip2conda)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/pip2conda.svg)](https://anaconda.org/conda-forge/pip2conda/)
[![License](https://img.shields.io/pypi/l/pip2conda.svg)](https://choosealicense.com/licenses/gpl-3.0/)
![Supported Python versions](https://img.shields.io/pypi/pyversions/pip2conda.svg)

[![Build status](https://github.com/duncanmmacleod/pip2conda/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/duncanmmacleod/pip2conda/actions/workflows/test.yml)
[![Documentation status](https://readthedocs.org/projects/pip2conda/badge/?version=latest)](https://pip2conda.readthedocs.io/en/latest/?badge=latest)
[![Coverage status](https://codecov.io/gh/duncanmmacleod/pip2conda/branch/main/graph/badge.svg)](https://codecov.io/gh/duncanmmacleod/pip2conda)

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

Just execute `pip2conda` from the base of your project directory.
For example, running `pip2conda` in the base directory for its own
project repository does this:

```console
$ pip2conda
grayskull
packaging>=20.0
python=3.9.*
requests
setuptools
setuptools>=42
setuptools_scm>=3.4
toml
tomli>=1.0.0
wheel
```
