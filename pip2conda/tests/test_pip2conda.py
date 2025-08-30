# Copyright (c) 2022-2025 Cardiff University
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for pip2conda."""

import json
import subprocess
from pathlib import Path
from shutil import which
from unittest import mock

import pytest
import requests
from build import BuildException

from ..pip2conda import (
    _normalize_dependency_groups,
    _normalize_group_name,
    _resolve_dependency_group,
    main as pip2conda_main,
    parse_dependency_groups,
    parse_requirements,
)

__author__ = "Duncan Macleod <duncanmmacleod@gmail.com>"

PIP2CONDA_WHL = (
    "https://files.pythonhosted.org/packages/py3/p/pip2conda/"
    "pip2conda-0.4.2-py3-none-any.whl"
)


# -- utilities --------------

def mock_proc(
    returncode: int = 0,
    data: dict | None = None,
) -> mock.MagicMock:
    """Mock a `subprocess.CompletedProcess`."""
    proc = mock.create_autospec(subprocess.CompletedProcess)
    proc.returncode = returncode
    proc.stdout = json.dumps(data or {})
    return proc


@pytest.fixture
def whl(tmp_path: Path) -> Path:
    """Download the pip2conda wheel file."""
    try:
        resp = requests.get(PIP2CONDA_WHL, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover
        pytest.skip(str(exc))
    whl = tmp_path / Path(PIP2CONDA_WHL).name
    with whl.open("wb") as file:
        file.write(resp.content)
    return whl


# -- unit tests -------------

@pytest.mark.parametrize(("reqs", "environment", "extras", "result"), [
    # simple list of packages, no environmment, no extras
    pytest.param(
        ["a", "b"],
        None,
        None,
        ["a", "b"],
        id="simple",
    ),
    # environment marker with negative match
    pytest.param(
        ["a ; python_version < '2.0'", "b"],
        {"python_version": "2.0"},
        None,
        ["b"],
        id="env_marker_neg",
    ),
    # environment marker and extra with negative match
    pytest.param(
        ["a ; python_version >= '2.0' and extra == 'test'", "b"],
        {"python_version": "2.0"},
        None,
        ["b"],
        id="env_marker_and_extra_neg",
    ),
    # no environment marker and extra with negative match
    pytest.param(
        ["a ; python_version >= '2.0' and extra == 'test'", "b"],
        None,
        ["dev"],
        ["b"],
        id="env_marker_and_extra_neg_2",
    ),
    # environment marker and extra with positive match
    pytest.param(
        ["a ; python_version >= '2.0' and extra == 'test'", "b"],
        {"python_version": "2.0"},
        ["test"],
        ["a", "b"],
        id="env_marker_and_extra_pos",
    ),
    # environment marker and extra with positive and negative matches
    pytest.param(
        [
            "a ; python_version >= '2.0' and extra == 'test'",
            "b ; extra == 'dev'",
        ],
        {"python_version": "2.0"},
        ["test", "doc"],
        ["a"],
        id="env_marker_and_extra_pos_neg",
    ),
    # environment marker and extras with multiple positive matches
    pytest.param(
        [
            "a ; python_version >= '2.0' and extra == 'test'",
            "b ; extra == 'dev'",
        ],
        {"python_version": "2.0"},
        ["test", "dev", "doc"],
        ["a", "b"],
        id="env_marker_and_extra_pos_neg_2",
    ),
])
def test_parse_requirements(reqs, environment, extras, result):
    """Test that :func:`parse_requirements` correctly evaluates markers."""
    assert list(parse_requirements(
        reqs,
        environment=environment,
        extras=extras,
    )) == result


# -- end-to-end tests -------

def test_setuptools_mock(tmp_path):
    """Test parsing a mixed setuptools project.

    This test also includes a mocked conda error indicating a missing package.
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
        ])

    # validate the result
    assert out.read_text().splitlines() == [
        "a",
        "c>=2.0",
        "e>=2.0",
        "python==9.9.*",
    ]


@pytest.mark.parametrize(("extras", "groups"), [
    ([], []),
    ([], ["test"]),
    (["astropy"], []),
    (["astropy"], ["test"]),
])
def test_setuptools_pyproject_toml(tmp_path, extras, groups):
    """Test parsing a setuptools project using only pyproject.toml."""
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
astropy = [
  "astropy",
]

[dependency-groups]
test = [
  "pytest",
  "pytest-cov < 2.0.0; python_version < '3.11'",
  "pytest-cov >= 2.0.0; python_version >= '3.11'",
]
""")

    # run the tool
    out = tmp_path / "out.txt"
    args = [
        "--project-dir", str(tmp_path),
        "--python-version", "3.11",
        "--output", str(out),
        "--skip-conda-forge-check",
    ]
    for extra in (extras or []):
        args.extend(["--extra", extra])
    for group in (groups or []):
        args.extend(["--group", group])
    pip2conda_main(args=args)

    # assert that we get what we should
    expected = {
        "numpy>=1.20.0",
        "python==3.11.*",
        "scipy",
        "setuptools",
    }
    if "astropy" in extras or []:
        expected.update({
            "astropy",
        })
    if "test" in groups or []:
        expected.update({
            "pytest",
            "pytest-cov>=2.0.0",
        })
    assert set(out.read_text().splitlines()) == expected


@pytest.mark.skipif(
    not which("conda"),
    reason="cannot find conda",
)
def test_setuptools_setup_cfg(tmp_path):
    """Test parsing a setuptools project that doesn't have a pyproject.toml file."""
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
        "safe-netrc>=1.0",
        "setuptools==62.1.0",
    }
    assert expected.issubset(result)
    assert "oldest-supported-numpy" not in result


def test_requirements_txt(tmp_path):
    """Test parsing requirements from requirements.txt files."""
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
    """Test parsing requirements from a poetry package."""
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
        "--extra", "test",
    ])

    # assert that we get what we should
    assert set(out.read_text().splitlines()) == {
        "h5py>=2.10.0",
        "poetry-core>=1.0.0",
        "pytest",
        "python>=3.10",
        "python<4.0",
    }


def test_wheel(tmp_path, whl):
    """Test parsing requirements from a wheel file."""
    # run the tool
    out = tmp_path / "out.txt"
    pip2conda_main(args=[
        "--output", str(out),
        "--wheel", str(whl),
        "--skip-conda-forge-check",
    ])

    # assert that we get what we should
    assert set(out.read_text().splitlines()) == {
        "grayskull>=1.0.0",
        "packaging",
        "python>=3.10",
        "python-build",
        "requests",
        "ruamel.yaml",
    }


def test_error(tmp_path):
    """Test that giving no valid input results in an error."""
    with pytest.raises(BuildException):
        pip2conda_main(args=[
            "--project-dir", str(tmp_path),
        ])


# -- dependency groups tests --------

def test_normalize_group_name():
    """Test dependency group name normalization."""
    assert _normalize_group_name("test") == "test"
    assert _normalize_group_name("Test") == "test"
    assert _normalize_group_name("test_group") == "test-group"
    assert _normalize_group_name("test-group") == "test-group"
    assert _normalize_group_name("test.group") == "test-group"
    assert _normalize_group_name("Test_Group.Name") == "test-group-name"


def test_normalize_dependency_groups():
    """Test dependency group normalization and duplicate detection."""
    groups = {
        "test": ["pytest"],
        "docs": ["sphinx"],
    }
    normalized = _normalize_dependency_groups(groups)
    assert normalized == {"test": ["pytest"], "docs": ["sphinx"]}

    # Test duplicate detection
    groups_with_duplicates = {
        "test": ["pytest"],
        "Test": ["coverage"],
    }
    with pytest.raises(ValueError, match="Duplicate dependency group names"):
        _normalize_dependency_groups(groups_with_duplicates)


def test_resolve_dependency_group_simple():
    """Test resolving simple dependency groups."""
    groups = {
        "test": ["pytest", "coverage"],
        "docs": ["sphinx"],
    }

    result = _resolve_dependency_group(groups, "test")
    assert result == ["pytest", "coverage"]

    result = _resolve_dependency_group(groups, "docs")
    assert result == ["sphinx"]


def test_resolve_dependency_group_with_includes():
    """Test resolving dependency groups with includes."""
    groups = {
        "base": ["requests"],
        "test": ["pytest", {"include-group": "base"}],
        "dev": [{"include-group": "test"}, "black"],
    }

    result = _resolve_dependency_group(groups, "test")  # type: ignore[arg-type]
    assert result == ["pytest", "requests"]

    result = _resolve_dependency_group(groups, "dev")  # type: ignore[arg-type]
    assert result == ["pytest", "requests", "black"]


def test_resolve_dependency_group_cycle_detection():
    """Test cycle detection in dependency group includes."""
    groups = {
        "a": [{"include-group": "b"}],
        "b": [{"include-group": "a"}],
    }

    with pytest.raises(ValueError, match="Cyclic dependency group include"):
        _resolve_dependency_group(groups, "a") # type: ignore[arg-type]


def test_resolve_dependency_group_missing():
    """Test error handling for missing dependency groups."""
    groups = {
        "test": ["pytest"],
    }

    with pytest.raises(LookupError, match="Dependency group 'missing' not found"):
        _resolve_dependency_group(groups, "missing")


def test_resolve_dependency_group_invalid_format():
    """Test error handling for invalid dependency group formats."""
    # Invalid group (not a list)
    groups = {
        "test": "not-a-list",
    }
    with pytest.raises(
        TypeError,
        match="Dependency group 'test' is not a list",
    ):
        _resolve_dependency_group(groups, "test")  # type: ignore[arg-type]


def test_resolve_dependency_group_invalid_table():
    """Test error handling for invalid dependency-group table."""
    groups = {
        "test": [{"wrong-key": "value"}],
    }
    with pytest.raises(
        ValueError,
        match="Invalid dependency group item",
    ):
        _resolve_dependency_group(groups, "test")  # type: ignore[arg-type]


def test_parse_dependency_groups(tmp_path):
    """Test parsing dependency groups from pyproject.toml."""
    # Create a test pyproject.toml
    pyproject_content = """
[dependency-groups]
test = ["pytest", "coverage"]
docs = ["sphinx", "myst-parser"]
dev = [
    {include-group = "test"},
    {include-group = "docs"},
    "pre-commit",
]
"""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(pyproject_content)

    # Test parsing individual groups
    result = parse_dependency_groups(tmp_path, ["test"])
    assert result == ["pytest", "coverage"]

    result = parse_dependency_groups(tmp_path, ["docs"])
    assert result == ["sphinx", "myst-parser"]

    # Test parsing group with includes
    result = parse_dependency_groups(tmp_path, ["dev"])
    assert result == ["pytest", "coverage", "sphinx", "myst-parser", "pre-commit"]

    # Test parsing multiple groups
    result = parse_dependency_groups(tmp_path, ["test", "docs"])
    assert result == ["pytest", "coverage", "sphinx", "myst-parser"]

    # Test parsing all groups
    result = parse_dependency_groups(tmp_path, "ALL")
    expected = [
        "pytest",
        "coverage",
        "sphinx",
        "myst-parser",
        "pytest",
        "coverage",
        "sphinx",
        "myst-parser",
        "pre-commit",
    ]
    assert result == expected


def test_parse_dependency_groups_no_file(tmp_path):
    """Test parsing dependency groups when pyproject.toml doesn't exist."""
    result = parse_dependency_groups(tmp_path, ["test"])
    assert result == []


def test_parse_dependency_groups_no_groups(tmp_path):
    """Test parsing dependency groups when no dependency-groups table exists."""
    pyproject_content = """
[project]
name = "test-project"
version = "1.0.0"
"""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(pyproject_content)

    result = parse_dependency_groups(tmp_path, ["test"])
    assert result == []


def test_pip2conda_main_with_dependency_groups(tmp_path):
    """Test the main function with dependency groups."""
    # Create test project
    pyproject_content = """
[project]
name = "test-project"
version = "1.0.0"
dependencies = ["requests"]

[dependency-groups]
test = ["pytest", "coverage"]
docs = ["sphinx"]
"""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(pyproject_content)

    # Test with dependency groups
    out = tmp_path / "out.txt"
    pip2conda_main(args=[
        "--project-dir", str(tmp_path),
        "--dependency-group", "test",
        "--output", str(out),
        "--skip-conda-forge-check",
        "--no-build-requires",
    ])

    output_lines = set(out.read_text().splitlines())
    assert "requests" in output_lines
    assert "pytest" in output_lines
    assert "coverage" in output_lines
    assert "sphinx" not in output_lines
