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
from packaging.requirements import InvalidRequirement

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
        _run.side_effect = [
            # Fail attempt 1 with a missing package
            mock_proc(
                returncode=1,
                data={
                    "exception_name": "PackagesNotFoundError",
                    "packages": [
                        "d",
                    ],
                },
            ),
            # Succeed attempt 2 with d removed
            mock_proc(
                returncode=0,
                data={},
            ),
        ]
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
    """Test dependency group normalization."""
    groups = {
        "test": ["pytest"],
        "docs": ["sphinx"],
    }
    normalized = _normalize_dependency_groups(groups)
    assert normalized == {"test": ["pytest"], "docs": ["sphinx"]}


def test_normalize_dependency_groups_duplicate():
    """Test dependency group duplicate detection."""
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


def test_parse_custom_dependency_groups(tmp_path):
    """Test parsing custom dependency groups from [tool.pip2conda.dependency-groups]."""
    pyproject_content = """
[tool.pip2conda.dependency-groups]
conda = ["my-conda-only-package", "another-conda-package"]
custom-test = ["pytest-conda", "conda-coverage"]
"""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(pyproject_content)

    # Test parsing custom groups
    result = parse_dependency_groups(tmp_path, ["conda"])
    assert result == ["my-conda-only-package", "another-conda-package"]

    result = parse_dependency_groups(tmp_path, ["custom-test"])
    assert result == ["pytest-conda", "conda-coverage"]

    # Test parsing all custom groups
    result = parse_dependency_groups(tmp_path, "ALL")
    expected = [
        "my-conda-only-package",
        "another-conda-package",
        "pytest-conda",
        "conda-coverage",
    ]
    assert result == expected


def test_parse_custom_dependency_groups_with_includes(tmp_path):
    """Test parsing custom dependency groups with include-group references."""
    pyproject_content = """
[tool.pip2conda.dependency-groups]
base = ["requests", "numpy"]
conda = ["my-conda-only-package"]
extended = [
    {include-group = "base"},
    {include-group = "conda"},
    "additional-package",
]
"""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(pyproject_content)

    result = parse_dependency_groups(tmp_path, ["extended"])
    expected = ["requests", "numpy", "my-conda-only-package", "additional-package"]
    assert result == expected


def test_parse_custom_dependency_groups_normalization(tmp_path):
    """Test that custom dependency group names are properly normalized."""
    pyproject_content = """
[tool.pip2conda.dependency-groups]
"Test_Group" = ["pytest"]
"docs.build" = ["sphinx"]
"""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(pyproject_content)

    # Test normalized access
    result = parse_dependency_groups(tmp_path, ["test-group"])
    assert result == ["pytest"]

    result = parse_dependency_groups(tmp_path, ["docs-build"])
    assert result == ["sphinx"]



def test_parse_custom_dependency_groups_invalid_format(tmp_path):
    """Test error handling for invalid custom dependency group formats."""
    pyproject_content = """
[tool.pip2conda.dependency-groups]
invalid = "not-a-list"
"""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(pyproject_content)

    with pytest.raises(TypeError, match="Dependency group 'invalid' is not a list"):
        parse_dependency_groups(tmp_path, ["invalid"])


def test_parse_custom_dependency_groups_missing_group(tmp_path):
    """Test error handling when requesting non-existent custom groups."""
    pyproject_content = """
[tool.pip2conda.dependency-groups]
existing = ["package"]
"""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(pyproject_content)

    with pytest.raises(LookupError, match="Dependency group 'missing' not found"):
        parse_dependency_groups(tmp_path, ["missing"])


def test_parse_custom_dependency_groups_cycle_detection(tmp_path):
    """Test cycle detection in custom dependency groups."""
    pyproject_content = """
[tool.pip2conda.dependency-groups]
a = [{include-group = "b"}]
b = [{include-group = "a"}]
"""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(pyproject_content)

    with pytest.raises(ValueError, match="Cyclic dependency group include"):
        parse_dependency_groups(tmp_path, ["a"])


def test_parse_custom_dependency_groups_empty_table(tmp_path):
    """Test parsing when custom dependency groups table is empty."""
    pyproject_content = """
[project]
name = "test"

[tool.pip2conda.dependency-groups]
"""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(pyproject_content)

    result = parse_dependency_groups(tmp_path, ["test"])
    assert result == []

    result = parse_dependency_groups(tmp_path, "ALL")
    assert result == []


def test_merge_standard_and_custom_dependency_groups(tmp_path):
    """Test merging behavior between standard and custom dependency groups."""
    pyproject_content = """
[dependency-groups]
test = ["pytest", "coverage"]
docs = ["sphinx"]

[tool.pip2conda.dependency-groups]
conda = ["my-conda-only-package"]
custom-test = ["pytest-conda"]
"""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(pyproject_content)

    # Test accessing standard groups
    result = parse_dependency_groups(tmp_path, ["test"])
    assert result == ["pytest", "coverage"]

    result = parse_dependency_groups(tmp_path, ["docs"])
    assert result == ["sphinx"]

    # Test accessing custom groups
    result = parse_dependency_groups(tmp_path, ["conda"])
    assert result == ["my-conda-only-package"]

    result = parse_dependency_groups(tmp_path, ["custom-test"])
    assert result == ["pytest-conda"]

    # Test accessing multiple groups from both sources
    result = parse_dependency_groups(tmp_path, ["test", "conda"])
    expected = ["pytest", "coverage", "my-conda-only-package"]
    assert result == expected

    # Test ALL groups includes both standard and custom
    result = parse_dependency_groups(tmp_path, "ALL")
    expected = [
        "pytest", "coverage",  # test
        "sphinx",  # docs
        "my-conda-only-package",  # conda
        "pytest-conda",  # custom-test
    ]
    assert result == expected


def test_custom_groups_cross_reference_standard_groups(tmp_path):
    """Test custom groups that reference standard groups via include-group."""
    pyproject_content = """
[dependency-groups]
base = ["requests", "numpy"]
test = ["pytest"]

[tool.pip2conda.dependency-groups]
conda = ["my-conda-only-package"]
extended = [
    {include-group = "base"},
    {include-group = "test"},
    {include-group = "conda"},
    "additional-package",
]
"""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(pyproject_content)

    result = parse_dependency_groups(tmp_path, ["extended"])
    expected = [
        "requests", "numpy",  # from base
        "pytest",  # from test
        "my-conda-only-package",  # from conda
        "additional-package",  # direct
    ]
    assert result == expected


def test_standard_groups_reference_custom_groups(tmp_path):
    """Test standard groups that reference custom groups via include-group."""
    pyproject_content = """
[dependency-groups]
dev = [
    {include-group = "test"},
    {include-group = "conda"},
    "dev-package",
]
test = ["pytest"]

[tool.pip2conda.dependency-groups]
conda = ["my-conda-only-package"]
"""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(pyproject_content)

    result = parse_dependency_groups(tmp_path, ["dev"])
    expected = [
        "pytest",  # from test
        "my-conda-only-package",  # from conda
        "dev-package",  # direct
    ]
    assert result == expected


def test_mixed_group_references_both_directions(tmp_path):
    """Test complex group references.

    Complex scenario with bidirectional references between standard and custom groups.
    """
    pyproject_content = """
[dependency-groups]
base = ["requests"]
standard-extended = [
    {include-group = "base"},
    {include-group = "custom-base"},
]

[tool.pip2conda.dependency-groups]
custom-base = ["conda-package"]
custom-extended = [
    {include-group = "standard-extended"},
    "final-package",
]
"""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(pyproject_content)

    result = parse_dependency_groups(tmp_path, ["custom-extended"])
    expected = [
        "requests",  # base -> standard-extended -> custom-extended
        "conda-package",  # custom-base -> standard-extended -> custom-extended
        "final-package",  # direct in custom-extended
    ]
    assert result == expected


def test_custom_groups_override_standard_groups(tmp_path):
    """Test that custom groups take precedence over standard groups with same name."""
    pyproject_content = """
[dependency-groups]
test = ["pytest", "coverage"]
docs = ["sphinx"]

[tool.pip2conda.dependency-groups]
test = ["pytest-conda", "conda-coverage"]
conda = ["my-conda-only-package"]
"""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(pyproject_content)

    # Custom 'test' group should override standard 'test' group
    result = parse_dependency_groups(tmp_path, ["test"])
    assert result == ["pytest-conda", "conda-coverage"]
    # Should not contain standard test packages
    assert "pytest" not in result
    assert "coverage" not in result

    # Standard 'docs' group should still be accessible
    result = parse_dependency_groups(tmp_path, ["docs"])
    assert result == ["sphinx"]

    # Custom-only group should work
    result = parse_dependency_groups(tmp_path, ["conda"])
    assert result == ["my-conda-only-package"]


def test_precedence_with_normalized_names(tmp_path):
    """Test precedence behavior with normalized group names."""
    pyproject_content = """
[dependency-groups]
"test_group" = ["standard-package"]

[tool.pip2conda.dependency-groups]
"test-group" = ["custom-package"]
"""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(pyproject_content)

    # Both normalize to "test-group", custom should take precedence
    result = parse_dependency_groups(tmp_path, ["test-group"])
    assert result == ["custom-package"]

    result = parse_dependency_groups(tmp_path, ["test_group"])
    assert result == ["custom-package"]


def test_precedence_with_includes_referencing_overridden_groups(tmp_path):
    """Test precedence when includes reference groups that are overridden."""
    pyproject_content = """
[dependency-groups]
base = ["standard-base"]
extended = [
    {include-group = "base"},
    "standard-extended",
]

[tool.pip2conda.dependency-groups]
base = ["custom-base"]
"""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(pyproject_content)

    # When 'extended' includes 'base', it should get the custom version
    result = parse_dependency_groups(tmp_path, ["extended"])
    expected = ["custom-base", "standard-extended"]
    assert result == expected
    assert "standard-base" not in result


def test_precedence_all_groups_includes_both_standard_and_custom(tmp_path):
    """Test that ALL groups includes both standard and custom.

    Assert custom taking precedence.
    """
    pyproject_content = """
[dependency-groups]
test = ["standard-test"]
docs = ["sphinx"]
unique-standard = ["standard-unique"]

[tool.pip2conda.dependency-groups]
test = ["custom-test"]
conda = ["conda-package"]
unique-custom = ["custom-unique"]
"""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(pyproject_content)

    result = parse_dependency_groups(tmp_path, "ALL")

    # Should contain custom test (not standard)
    assert "custom-test" in result
    assert "standard-test" not in result

    # Should contain standard docs (no custom override)
    assert "sphinx" in result

    # Should contain unique groups from both
    assert "standard-unique" in result
    assert "custom-unique" in result
    assert "conda-package" in result


def test_precedence_error_handling_with_overrides(tmp_path):
    """Test error handling when overridden groups have different validation issues."""
    pyproject_content = """
[dependency-groups]
test = ["valid-package"]

[tool.pip2conda.dependency-groups]
test = "invalid-not-a-list"
"""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(pyproject_content)

    # Should fail with custom group validation error, not fall back to standard
    with pytest.raises(TypeError, match="Dependency group 'test' is not a list"):
        parse_dependency_groups(tmp_path, ["test"])


def test_malformed_custom_groups_invalid_toml_structure(tmp_path):
    """Test handling of custom dependency groups with invalid TOML."""
    # Test missing tool section
    pyproject_content = """
[dependency-groups]
test = ["pytest"]

[pip2conda.dependency-groups]  # Missing 'tool.' prefix
conda = ["package"]
"""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(pyproject_content)

    # Should only find standard groups, custom groups should be ignored
    result = parse_dependency_groups(tmp_path, ["test"])
    assert result == ["pytest"]

    # Custom group should not be found
    with pytest.raises(LookupError, match="Dependency group 'conda' not found"):
        parse_dependency_groups(tmp_path, ["conda"])


def test_malformed_custom_groups_invalid_include_syntax(tmp_path):
    """Test handling of malformed include-group syntax in custom groups."""
    pyproject_content = """
[tool.pip2conda.dependency-groups]
invalid-include = [
    {"wrong-key" = "value"},
    "valid-package",
]
"""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(pyproject_content)

    with pytest.raises(ValueError, match="Invalid dependency group item"):
        parse_dependency_groups(tmp_path, ["invalid-include"])


def test_malformed_custom_groups_invalid_requirement_specs(tmp_path):
    """Test handling of invalid PEP 508 requirement specs in custom groups."""
    pyproject_content = """
[tool.pip2conda.dependency-groups]
invalid-specs = [
    "valid-package",
    "invalid requirement spec with spaces and no quotes",
    "another-valid-package",
]
"""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(pyproject_content)

    # Should fail when trying to validate the invalid requirement spec
    with pytest.raises(
        InvalidRequirement,
        match="Expected end or semicolon",
    ):
        parse_dependency_groups(tmp_path, ["invalid-specs"])


def test_malformed_custom_groups_mixed_types_in_list(tmp_path):
    """Test handling of mixed invalid types in custom dependency group lists."""
    pyproject_content = """
[tool.pip2conda.dependency-groups]
mixed-types = [
    "valid-package",
    123,  # Invalid: number
    {"include-group" = "valid"},
    true,  # Invalid: boolean
]
"""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(pyproject_content)

    with pytest.raises(ValueError, match="Invalid dependency group item"):
        parse_dependency_groups(tmp_path, ["mixed-types"])


def test_empty_custom_groups_section(tmp_path):
    """Test handling of empty [tool.pip2conda] section."""
    pyproject_content = """
[dependency-groups]
test = ["pytest"]

[tool.pip2conda]
# Empty section, no dependency-groups
"""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(pyproject_content)

    # Should work with standard groups only
    result = parse_dependency_groups(tmp_path, ["test"])
    assert result == ["pytest"]

    result = parse_dependency_groups(tmp_path, "ALL")
    assert result == ["pytest"]


def test_missing_tool_section_entirely(tmp_path):
    """Test handling when [tool] section doesn't exist at all."""
    pyproject_content = """
[dependency-groups]
test = ["pytest"]

[project]
name = "test"
"""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(pyproject_content)

    # Should work with standard groups only
    result = parse_dependency_groups(tmp_path, ["test"])
    assert result == ["pytest"]

    result = parse_dependency_groups(tmp_path, "ALL")
    assert result == ["pytest"]


def test_custom_groups_with_empty_lists(tmp_path):
    """Test handling of custom groups with empty dependency lists."""
    pyproject_content = """
[tool.pip2conda.dependency-groups]
empty = []
non-empty = ["package"]
"""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(pyproject_content)

    result = parse_dependency_groups(tmp_path, ["empty"])
    assert result == []

    result = parse_dependency_groups(tmp_path, ["non-empty"])
    assert result == ["package"]

    result = parse_dependency_groups(tmp_path, "ALL")
    assert result == ["package"]  # Empty group contributes nothing


def test_custom_groups_with_only_includes(tmp_path):
    """Test custom groups that contain only include-group references."""
    pyproject_content = """
[dependency-groups]
base = ["requests"]

[tool.pip2conda.dependency-groups]
only-includes = [
    {"include-group" = "base"},
]
"""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(pyproject_content)

    result = parse_dependency_groups(tmp_path, ["only-includes"])
    assert result == ["requests"]


def test_custom_groups_circular_includes_across_standard_custom(tmp_path):
    """Test circular includes between standard and custom groups."""
    pyproject_content = """
[dependency-groups]
standard = [
    {"include-group" = "custom"},
    "standard-package",
]

[tool.pip2conda.dependency-groups]
custom = [
    {"include-group" = "standard"},
    "custom-package",
]
"""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(pyproject_content)

    with pytest.raises(ValueError, match="Cyclic dependency group include"):
        parse_dependency_groups(tmp_path, ["standard"])

    with pytest.raises(ValueError, match="Cyclic dependency group include"):
        parse_dependency_groups(tmp_path, ["custom"])


def test_cli_with_custom_dependency_groups(tmp_path):
    """Test CLI with custom dependency groups using --dependency-group flag."""
    pyproject_content = """
[project]
name = "test-project"
version = "1.0.0"
dependencies = ["requests"]

[dependency-groups]
test = ["pytest"]

[tool.pip2conda.dependency-groups]
conda = ["my-conda-only-package"]
custom-test = ["pytest-conda", "conda-coverage"]
"""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(pyproject_content)

    # Test with custom group
    out = tmp_path / "out.txt"
    pip2conda_main(args=[
        "--project-dir", str(tmp_path),
        "--dependency-group", "conda",
        "--output", str(out),
        "--skip-conda-forge-check",
        "--no-build-requires",
    ])

    output_lines = set(out.read_text().splitlines())
    assert "requests" in output_lines
    assert "my-conda-only-package" in output_lines
    assert "pytest" not in output_lines
    assert "pytest-conda" not in output_lines

    # Test with multiple groups including both standard and custom
    out2 = tmp_path / "out2.txt"
    pip2conda_main(args=[
        "--project-dir", str(tmp_path),
        "--dependency-group", "test",
        "--dependency-group", "custom-test",
        "--output", str(out2),
        "--skip-conda-forge-check",
        "--no-build-requires",
    ])

    output_lines2 = set(out2.read_text().splitlines())
    assert "requests" in output_lines2
    assert "pytest" in output_lines2  # from standard test group
    assert "pytest-conda" in output_lines2  # from custom-test group
    assert "conda-coverage" in output_lines2  # from custom-test group


def test_cli_with_all_groups_including_custom(tmp_path):
    """Test CLI with --all-groups flag including custom dependency groups."""
    pyproject_content = """
[project]
name = "test-project"
version = "1.0.0"
dependencies = ["requests"]

[dependency-groups]
test = ["pytest"]
docs = ["sphinx"]

[tool.pip2conda.dependency-groups]
conda = ["my-conda-only-package"]
custom-test = ["pytest-conda"]
"""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(pyproject_content)

    out = tmp_path / "out.txt"
    pip2conda_main(args=[
        "--project-dir", str(tmp_path),
        "--all-groups",
        "--output", str(out),
        "--skip-conda-forge-check",
        "--no-build-requires",
    ])

    output_lines = set(out.read_text().splitlines())
    # Should include base dependencies
    assert "requests" in output_lines
    # Should include standard groups
    assert "pytest" in output_lines
    assert "sphinx" in output_lines
    # Should include custom groups
    assert "my-conda-only-package" in output_lines
    assert "pytest-conda" in output_lines


def test_cli_with_custom_groups_precedence(tmp_path):
    """Test CLI behavior when custom groups override standard groups."""
    pyproject_content = """
[project]
name = "test-project"
version = "1.0.0"
dependencies = ["requests"]

[dependency-groups]
test = ["pytest", "coverage"]

[tool.pip2conda.dependency-groups]
test = ["pytest-conda", "conda-coverage"]
"""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(pyproject_content)

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
    # Should use custom test group, not standard
    assert "pytest-conda" in output_lines
    assert "conda-coverage" in output_lines
    assert "pytest" not in output_lines
    assert "coverage" not in output_lines


def test_cli_with_custom_groups_includes(tmp_path):
    """Test CLI with custom groups that include other groups."""
    pyproject_content = """
[project]
name = "test-project"
version = "1.0.0"
dependencies = ["requests"]

[dependency-groups]
base = ["numpy"]

[tool.pip2conda.dependency-groups]
conda = ["my-conda-only-package"]
extended = [
    {"include-group" = "base"},
    {"include-group" = "conda"},
    "additional-package",
]
"""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(pyproject_content)

    out = tmp_path / "out.txt"
    pip2conda_main(args=[
        "--project-dir", str(tmp_path),
        "--dependency-group", "extended",
        "--output", str(out),
        "--skip-conda-forge-check",
        "--no-build-requires",
    ])

    output_lines = set(out.read_text().splitlines())
    assert "requests" in output_lines
    assert "numpy" in output_lines  # from base
    assert "my-conda-only-package" in output_lines  # from conda
    assert "additional-package" in output_lines  # direct


def test_cli_error_handling_with_custom_groups(tmp_path):
    """Test CLI error handling when custom groups have issues."""
    pyproject_content = """
[project]
name = "test-project"
version = "1.0.0"

[tool.pip2conda.dependency-groups]
invalid = "not-a-list"
"""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(pyproject_content)

    out = tmp_path / "out.txt"
    with pytest.raises(TypeError, match="Dependency group 'invalid' is not a list"):
        pip2conda_main(args=[
            "--project-dir", str(tmp_path),
            "--dependency-group", "invalid",
            "--output", str(out),
            "--skip-conda-forge-check",
            "--no-build-requires",
        ])


def test_cli_with_nonexistent_custom_group(tmp_path):
    """Test CLI error handling when requesting non-existent custom group."""
    pyproject_content = """
[project]
name = "test-project"
version = "1.0.0"

[tool.pip2conda.dependency-groups]
existing = ["package"]
"""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(pyproject_content)

    out = tmp_path / "out.txt"
    with pytest.raises(LookupError, match="Dependency group 'nonexistent' not found"):
        pip2conda_main(args=[
            "--project-dir", str(tmp_path),
            "--dependency-group", "nonexistent",
            "--output", str(out),
            "--skip-conda-forge-check",
            "--no-build-requires",
        ])


def test_cli_yaml_output_with_custom_groups(tmp_path):
    """Test CLI YAML output format with custom dependency groups."""
    pyproject_content = """
[project]
name = "test-project"
version = "1.0.0"
dependencies = ["requests"]

[tool.pip2conda.dependency-groups]
conda = ["my-conda-only-package"]
"""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(pyproject_content)

    out = tmp_path / "environment.yml"
    pip2conda_main(args=[
        "--project-dir", str(tmp_path),
        "--dependency-group", "conda",
        "--output", str(out),
        "--skip-conda-forge-check",
        "--no-build-requires",
    ])

    # Verify YAML format
    assert out.exists()
    content = out.read_text()
    assert "channels:" in content
    assert "conda-forge" in content
    assert "dependencies:" in content
    assert "requests" in content
    assert "my-conda-only-package" in content
