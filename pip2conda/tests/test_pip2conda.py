# -*- coding: utf-8 -*-
# Copyright (C) Cardiff University (2022)
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for pip2conda
"""

import json
import subprocess
from shutil import which
from unittest import mock

import pytest

from ..pip2conda import (
    main as pip2conda_main,
)

__author__ = "Duncan Macleod <duncanmmacleod@gmail.com>"


def mock_proc(returncode=0, data=dict()):
    proc = mock.create_autospec(subprocess.CompletedProcess)
    proc.returncode = returncode
    proc.stdout = json.dumps(data)
    return proc


def test_end2end_mock(tmp_path):
    # write package information
    (tmp_path / "pyproject.toml").write_text("""[build-system]
requires = [
    'a',
    "b ; sys.platform == 'test'",
    'c >= 2.0',
]
""")
    (tmp_path / "setup.cfg").write_text("""[options]
install_requires =
    d
    e >= 2.0
""")

    # run the tool (mocking out the call to conda)
    out = tmp_path / "out.txt"
    with mock.patch("subprocess.run") as _run:
        _run.return_value = mock_proc(
            returncode=1,
            data={
                "exception_name": "PackagesNotFoundError",
                "packages": [
                    "d",
                ],
            },
        )
        pip2conda_main(args=[
            "--python-version", "9.9",
            "--project-dir", str(tmp_path),
            "--output", str(out),
            "--disable-mamba",
        ])

        # validate that --disable-mamba worked
        _run.assert_called_once()

    # validate the result
    assert out.read_text().splitlines() == [
        "a",
        "c>=2.0",
        "e>=2.0",
        "python=9.9.*",
    ]


@pytest.mark.skipif(
    not which("conda"),
    reason="cannot find conda",
)
def test_end2end_real(tmp_path):
    # write package information (using exact pins for reproducibility)
    (tmp_path / "setup.cfg").write_text("""[options]
setup_requires =
    setuptools ==62.1.0
    setuptools_scm[toml] ==6.4.2
install_requires =
    gwpy ==2.1.3
    igwn-auth-utils[requests] ==0.2.2
    oldest-supported-numpy

[options.extras_require]
test =
    pytest
    pytest-cov
""")

    # run the tool (mocking out the call to conda)
    out = tmp_path / "out.txt"
    try:
        pip2conda_main(args=[
            "--project-dir", str(tmp_path),
            "--output", str(out),
            "--all",
        ])
    except subprocess.CalledProcessError as exc:
        pytest.skip(str(exc))

    # validate the result
    result = sorted(out.read_text().splitlines())
    expected = {
        "gwpy==2.1.3",
        "igwn-auth-utils==0.2.2",
        "safe-netrc>=1.0.0",
        "setuptools>=42",
    }
    assert expected.issubset(result)
    assert "oldest-supported-numpy" not in result


def test_end2end_requirements(tmp_path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("""
mock ; python_version < '3.0'
numpy
scipy >= 1.4.0
""")
    out = tmp_path / "out.txt"
    pip2conda_main(args=[
        "--no-build-requires",
        "--output", str(out),
        "--project-dir", str(tmp_path),
        "--requirements", str(requirements),
        "--skip-conda-forge-check",
    ])

    # validate the result
    result = sorted(out.read_text().splitlines())
    expected = {
        "numpy",
        "scipy>=1.4.0",
    }
    assert expected.issubset(result)
