# Copyright (c) 2022-2025 Cardiff University
# SPDX-License-Identifier: GPL-3.0-or-later

"""Sphinx configuration for pip2conda
"""

import re

from pip2conda import __version__ as pip2conda_version

# General information about the project.
project = "pip2conda"
copyright = "2022-2025, Cardiff University"
author = "Duncan Macleod"
version = re.split(r"[\w-]", pip2conda_version)[0]
release = pip2conda_version

# input file support
source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

# extra options
default_role = "obj"

# -- HTML formatting --------

html_theme = "furo"
html_title = f"{project} {version}"
html_css_files = [
    # add fontawesome for icons
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.2/css/fontawesome.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.2/css/solid.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.2/css/brands.min.css",
]
html_theme_options = {
    # add gitlab link in footer
    "footer_icons": [
        {
            "name": "GitLab",
            "url": "https://gitlab.com/gwpy/pip2conda",
            "class": "fa-brands fa-gitlab",
        },
    ],
    # colours
    "dark_css_variables": {
        "color-brand-content": (gwpy_dark_mode_colour := "#ededed"),
        "color-brand-primary": gwpy_dark_mode_colour,
        "color-brand-visited": gwpy_dark_mode_colour,
    },
    "light_css_variables": {
        "color-brand-content": (gwpy_light_mode_colour := "#29435f"),
        "color-brand-primary": gwpy_light_mode_colour,
        "color-brand-visited": gwpy_light_mode_colour,
    },
}


# -- extensions

extensions = [
    "myst_parser",
    "sphinx_copybutton",
]

# myst_parser
myst_enable_extensions = [
    "attrs_block",
]
