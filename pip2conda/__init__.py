# -*- coding: utf-8 -*-
# Copyright (C) Cardiff University (2022)
# SPDX-License-Identifier: GPL-3.0-or-later

"""Translate pip requirements into conda requirements
"""

__author__ = "Duncan Macleod <duncanmmacleod@gmail.com>"

try:
    from ._version import version as __version__
except ModuleNotFoundError:  # development mode
    __version__ = "dev"

from .pip2conda import pip2conda
