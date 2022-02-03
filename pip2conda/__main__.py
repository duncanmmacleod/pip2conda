# -*- coding: utf-8 -*-
# Copyright (C) Cardiff University (2022)
# SPDX-License-Identifier: GPL-3.0-or-later

"""Module execution entry point for pip2conda
"""

import sys

from .pip2conda import main


if __name__ == "__main__":
    sys.exit(main())
