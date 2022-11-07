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


def test_setuptools_mock(tmp_path):
    """Test parsing requirements of a mixed setuptools project
    while mocking out a conda error.
    """
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


def test_setuptools_pyproject_toml(tmp_path):
    """Test parsing requirements of a setuptools project using
    only pyproject.toml, without checking with conda."
    """
    # write package information
    (tmp_path / "pyproject.toml").write_text("""
[build-system]
requires = [ "setuptools" ]
build-backend = "setuptools.build_meta"

[project]
name = "pip2conda-tests"
version = "0.0.0"
requires-python = ">=3.9"
dependencies = [
  "numpy >= 1.20.0",
  "scipy",
]

[project.optional-dependencies]
test = [
  "pytest",
  "pytest-cov < 2.0.0; python_version < '3.11'",
  "pytest-cov >= 2.0.0; python_version >= '3.11'",
]
""")

    # run the tool
    out = tmp_path / "out.txt"
    pip2conda_main(args=[
        "--project-dir", str(tmp_path),
        "--python-version", "3.11",
        "--output", str(out),
        "--skip-conda-forge-check",
        "test",
    ])

    # assert that we get what we should
    assert set(out.read_text().splitlines()) == {
        "numpy>=1.20.0",
        "pytest",
        "pytest-cov>=2.0.0",
        "python=3.11.*",
        "scipy",
        "setuptools",
    }


@pytest.mark.skipif(
    not which("conda"),
    reason="cannot find conda",
)
def test_setuptools_setup_cfg(tmp_path):
    """Test parsing requirements of a setuptools project that doesn't
    have a pyproject.toml file at all.
    """
    # write package information (using exact pins for reproducibility)
    (tmp_path / "setup.cfg").write_text("""
[metadata]
name = test
version = 0.0.0

[options]
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
    (tmp_path / "setup.py").write_text("""
from setuptools import setup
setup()
""")

    # run the tool (mocking out the call to conda)
    out = tmp_path / "out.txt"
    try:
        pip2conda_main(args=[
            "--project-dir", str(tmp_path),
            "--output", str(out),
            "--all",
            "--verbose",
            "--verbose",
        ])
    except subprocess.CalledProcessError as exc:
        pytest.skip(str(exc))

    # validate the result
    result = sorted(out.read_text().splitlines())
    expected = {
        "gwpy==2.1.3",
        "igwn-auth-utils==0.2.2",
        "safe-netrc>=1.0.0",
        "setuptools==62.1.0",
    }
    assert expected.issubset(result)
    assert "oldest-supported-numpy" not in result


def test_requirements_txt(tmp_path):
    """Test parsing requirements from requirements.txt files.
    """
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


def test_poetry(tmp_path):
    """Test parsing requirements from a poetry package.
    """
    (tmp_path / "pyproject.toml").write_text("""
[build-system]
requires = [ "poetry-core>=1.0.0" ]
build-backend = "poetry.core.masonry.api"

[tool.poetry]
name = "test"
version = "0.0.0"
description = "test project"
authors = [ "Duncan Macleod <duncanmmacleod@gmail.com>" ]

[tool.poetry.dependencies]
python = "^3.10"
h5py = ">=2.10.0"
# for tests
pytest = {version="*", optional=true}

[tool.poetry.extras]
test = [ "pytest" ]
""")
    (tmp_path / "test.py").touch()

    # run the tool
    out = tmp_path / "out.txt"
    pip2conda_main(args=[
        "--output", str(out),
        "--project-dir", str(tmp_path),
        "--skip-conda-forge-check",
        "--verbose",
        "--verbose",
        "test",
    ])

    # assert that we get what we should
    assert set(out.read_text().splitlines()) == {
        "h5py>=2.10.0",
        "poetry-core>=1.0.0",
        "pytest",
        "python>=3.10,<4.0",
    }
