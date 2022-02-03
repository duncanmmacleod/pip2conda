# -*- coding: utf-8 -*-
# Copyright (C) Cardiff University (2022)
# SPDX-License-Identifier: GPL-3.0-or-later

"""Sphinx configuration for pip2conda
"""

import re

from pip2conda import __version__ as pip2conda_version

# General information about the project.
project = 'pip2conda'
copyright = '2022, Cardiff University'
author = 'Duncan Macleod'
version = re.split(r"[\w-]", pip2conda_version)[0]
release = pip2conda_version

# input file support
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

# extra options
default_role = "obj"

# html options
html_theme = "sphinx_rtd_theme"

# -- extensions

extensions = [
    "myst_parser",
]
