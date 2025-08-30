# Copyright (c) 2022-2025 Cardiff University
# SPDX-License-Identifier: GPL-3.0-or-later

"""Parse requirements for a Python project and convert for installing with conda.

This project parses the Python metadata for a project, including optional-dependencies
and dependency-groups, and returns a list of packages that can be installed using
Conda from the conda-forge channel.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import subprocess
import tempfile
import warnings
from collections import defaultdict
from importlib.metadata import PathDistribution
from pathlib import Path
from shutil import which
from typing import TYPE_CHECKING

try:
    from tomllib import load as toml_load
except ModuleNotFoundError:  # python < 3.11
    from tomli import load as toml_load  # type: ignore[assignment,no-redef]

import requests
from build import (
    BuildBackendException,
    BuildException,
    ProjectBuilder,
)
from build.env import DefaultIsolatedEnv
from grayskull.strategy.pypi import PYPI_CONFIG
from packaging.requirements import Requirement
from ruamel.yaml import YAML
from wheel.wheelfile import WheelFile

try:
    from coloredlogs import ColoredFormatter as Formatter
except ImportError:
    Formatter = logging.Formatter

if TYPE_CHECKING:
    from collections.abc import (
        Collection,
        Iterable,
        Iterator,
        Mapping,
        Sequence,
    )
    from typing import (
        Literal,
        TextIO,
    )

    from packaging.markers import Marker

    DependencyGroupsType = Mapping[
        str,
        Sequence[str | Mapping[Literal["include-group"], str]],
    ]

log = logging.getLogger(__package__)

yaml = YAML()

# conda config
CONDA = (
    which("conda", mode=os.X_OK)
    or os.environ.get("CONDA_EXE")
    or "conda"
)

# default timeout for an HTTP GET request
REQUESTS_TIMEOUT = 60

# regex to match version spec characters
VERSION_OPERATOR = re.compile("[><=!]")


# -- conda utilities --------

def load_conda_forge_name_map() -> dict[str, str]:
    """Load the PyPI <-> conda-forge package name map from grayskull.

    See https://github.com/conda-incubator/grayskull/blob/main/grayskull/pypi/config.yaml
    """
    # parse the config file and return (pypi_name: conda_forge_name) pairs
    with Path(PYPI_CONFIG).open() as conf:
        return {
            x: y["conda_forge"]
            for x, y in yaml.load(conf).items()
        }


def format_requirement(
    requirement: Requirement,
    conda_forge_map: dict[str, str] | None = None,
) -> Iterator[str]:
    """Format a (pip) Requirement as a conda dependency.

    Complicated specifiers (with multiple conditions) are separated into
    individual requirements which are yielded individually.

    Parameters
    ----------
    requirement : `packaging.requirements.Requirement`
        The requirement to format.

    conda_forge_map : `dict`
        `(pypi_name, conda_forge_name)` mapping dictionary.

    Yields
    ------
    formatted : `str`
        A formatted conda requirement.

    Examples
    --------
    >>> import packaging.requirements
    >>> req = packaging.requirements.Requirement.parse("htcondor >= 9.0.0")
    >>> print(list(format_requirement(req)))
    ['python-htcondor>=9.0.0']
    >>> req = packaging.requirements.Requirement.parse("python-framel>=8.40.1,!=8.46.0")
    >>> print(list(format_requirement(req)))
    ['python-framel!=8.46.0', 'python-framel>=8.40.1']
    """
    if conda_forge_map is None:
        conda_forge_map = {}
    name = conda_forge_map.get(requirement.name, requirement.name.lower())
    if requirement.specifier:
        for spec in requirement.specifier:
            yield name + str(spec)
    else:
        yield name


def _normalize_group_name(name: str) -> str:
    """Normalize a dependency group name according to PEP 735.

    Parameters
    ----------
    name : str
        The dependency group name to normalize.

    Returns
    -------
    str
        The normalized group name.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def _normalize_dependency_groups(
    dependency_groups: DependencyGroupsType,
) -> DependencyGroupsType:
    """Normalize dependency group names and detect duplicates.

    Parameters
    ----------
    dependency_groups : `dict`
        Dictionary mapping group names to their dependency lists.

    Returns
    -------
    dict
        Dictionary with normalized group names as keys.

    Raises
    ------
    ValueError
        If duplicate normalized names are detected.
    """
    original_names = defaultdict(list)
    normalized_groups: DependencyGroupsType = {}

    for group_name, value in dependency_groups.items():
        normed_group_name = _normalize_group_name(group_name)
        original_names[normed_group_name].append(group_name)
        normalized_groups[normed_group_name] = value

    errors = []
    for normed_name, names in original_names.items():
        if len(names) > 1:
            errors.append(f"{normed_name} ({', '.join(names)})")
    if errors:
        msg = f"Duplicate dependency group names: {', '.join(errors)}"
        raise ValueError(msg)

    return normalized_groups


def _resolve_dependency_group(
    dependency_groups: DependencyGroupsType,
    group: str,
    past_groups: tuple[str, ...] = (),
) -> list[str]:
    """Resolve a single dependency group, expanding includes recursively.

    Parameters
    ----------
    dependency_groups : `dict`
        Dictionary of all dependency groups.
    group : str
        The name of the group to resolve.
    past_groups : `tuple`
        Groups already being resolved (for cycle detection).

    Returns
    -------
    list
        List of resolved requirement strings.

    Raises
    ------
    ValueError
        If a cycle is detected or invalid data is found.
    LookupError
        If a referenced group doesn't exist.
    """
    if group in past_groups:
        msg = f"Cyclic dependency group include: {group} -> {past_groups}"
        raise ValueError(msg)

    if group not in dependency_groups:
        msg = f"Dependency group '{group}' not found"
        raise LookupError(msg)

    raw_group = dependency_groups[group]
    if not isinstance(raw_group, list):
        msg = f"Dependency group '{group}' is not a list"
        raise TypeError(msg)

    realized_group = []
    for item in raw_group:
        if isinstance(item, str):
            # Validate as PEP 508 dependency specifier
            Requirement(item)
            realized_group.append(item)
        elif isinstance(item, dict):
            if tuple(item.keys()) != ("include-group",):
                msg = f"Invalid dependency group item: {item}"
                raise ValueError(msg)

            include_group = _normalize_group_name(next(iter(item.values())))
            realized_group.extend(
                _resolve_dependency_group(
                    dependency_groups,
                    include_group,
                    (*past_groups, group),
                ),
            )
        else:
            msg = f"Invalid dependency group item: {item}"
            raise ValueError(msg)  # noqa: TRY004

    return realized_group


def parse_dependency_groups(
    project_dir: Path,
    groups_to_parse: Iterable[str] | str,
) -> list[str]:
    """Parse dependency groups from pyproject.toml.

    Parameters
    ----------
    project_dir : `pathlib.Path`
        Path to the project directory containing pyproject.toml.
    groups_to_parse : `list` of `str` or `str`
        List of dependency group names to parse, or "ALL" for all groups.

    Returns
    -------
    list
        List of requirement strings from the specified dependency groups.

    Raises
    ------
    FileNotFoundError
        If pyproject.toml doesn't exist.
    ValueError
        If dependency groups data is invalid.
    LookupError
        If a requested group doesn't exist.
    """
    pyproject_path = project_dir / "pyproject.toml"
    if not pyproject_path.exists():
        return []

    with pyproject_path.open("rb") as f:
        pyproject_data = toml_load(f)

    dependency_groups_raw = pyproject_data.get("dependency-groups", {})
    if not dependency_groups_raw:
        return []

    dependency_groups = _normalize_dependency_groups(dependency_groups_raw)

    if groups_to_parse == "ALL":
        groups_to_parse = list(dependency_groups.keys())
    elif isinstance(groups_to_parse, str):
        groups_to_parse = [groups_to_parse]

    # Normalize requested group names
    groups_to_parse = [_normalize_group_name(g) for g in groups_to_parse]

    requirements = []
    for group in groups_to_parse:
        requirements.extend(_resolve_dependency_group(dependency_groups, group))

    return requirements


# -- python metadata parsing

def parse_setup_requires(project_dir: Path) -> list[str]:
    """Parse the list of `setup_requires` packages from a setuptools dist.

    Parameters
    ----------
    project_dir : `pathlib.Path`
        The path to the project to be parsed.

    Returns
    -------
    setup_requires : `list`
        The list of build requirements.
    """
    from setuptools import Distribution  # noqa: PLC0415

    origin = Path().cwd()
    os.chdir(project_dir)
    try:
        dist = Distribution()
        dist.parse_config_files()
    finally:
        os.chdir(origin)
    return dist.setup_requires


def read_wheel_metadata(path: Path | str) -> dict[str, str | list[str]]:
    """Read the metadata for a project from a wheel."""
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        WheelFile(path, "r") as whl,
    ):
        tmppath = Path(tmpdir)
        # extract only the dist_info directory
        distinfo = [
            name for name in whl.namelist()
            if name.startswith(f"{whl.dist_info_path}/")
        ]
        whl.extractall(members=distinfo, path=tmppath)
        # return the metadata as JSON
        return PathDistribution(
            tmppath / whl.dist_info_path,
        ).metadata.json


def build_project_metadata(project_dir: Path) -> dict[str, str | list[str]]:
    """Build the metadata for a project.

    This function is basically a stripped down version of
    the python-build interface, which only generates the metadata
    and then stops.

    This function may generated a temporary environment in which to
    install the backend, if required.

    Parameters
    ----------
    project_dir : `pathlib.Path`
        The project to build.

    Returns
    -------
    meta : `dict`
        The package metadata as parsed by
        `importlib.metadata.Distribution.metadata.json`.
    """
    log.info("building metadata for %s", project_dir)

    # use python-build to generate the build metadata
    builder = ProjectBuilder(project_dir)
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            metadir = builder.prepare("wheel", tmpdir)
        except BuildBackendException as exc:
            log.debug("preparing wheel failed: '%s'", str(exc))
            log.debug("building isolated environment...")
            # the backend is missing, so we need to
            # install it on-the-fly
            with DefaultIsolatedEnv() as env:
                builder = ProjectBuilder.from_isolated_env(
                    env,
                    project_dir,
                )
                env.install(builder.build_system_requires)
                metadir = builder.prepare("wheel", tmpdir)
        if metadir is None:
            msg = f"Failed to prepare wheel for {project_dir}"
            raise RuntimeError(msg)
        dist = PathDistribution(Path(metadir))
        meta = dist.metadata.json

    # inject the build system requirements into the metadata
    build_requires: Iterable[str]
    if (project_dir / "pyproject.toml").is_file():
        build_requires = builder.build_system_requires
    else:
        # not given in pyproject.toml, so need to parse
        # manually from setup.cfg
        build_requires = parse_setup_requires(project_dir)
    meta["build_system_requires"] = list(build_requires)

    return meta


def parse_req_extras(
    req: Requirement,
    environment: dict[str, str] | None = None,
    conda_forge_map: dict[str, str] | None = None,
) -> Iterator[str]:
    """Parse the extras for a requirement.

    This unpackes a requirement like `package[extra]` into the list of
    actual packages that are required, and yields formatted conda
    dependencies.

    Parameters
    ----------
    req : `packaging.requirements.Requirement`
        the requirement to format
    environment : `dict`, optional
        the environment against which to evaluate markers
    conda_forge_map : `dict`
        `(pypi_name, conda_forge_name)` mapping dictionary
    """
    if not req.extras:
        return

    # query pypi for metadata
    resp = requests.get(
        f"https://pypi.org/pypi/{req.name}/json",
        timeout=REQUESTS_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()

    # parse the requirements that match the requested extras
    yield from parse_requirements(
        data["info"]["requires_dist"] or [],
        environment=environment,
        conda_forge_map=conda_forge_map,
        extras=req.extras,
        depth=1,
    )


def _evaluate_marker(
    marker: Marker | None,
    environment: dict[str, str] | None = None,
    extras: Iterable[str] | None = None,
) -> bool:
    """Evaluate whether an environment marker matches this environment."""
    if not marker:  # no marker, always True
        return True

    if environment is None:
        environment = {}

    # loop over all extras (including 'no extra') and see if there's a match
    for extra in {""} | set(extras or []):
        environment["extra"] = extra
        if marker.evaluate(environment):
            return True
    return False


def parse_requirements(
    requirements: Iterable[str],
    conda_forge_map: dict[str, str] | None = None,
    environment: dict[str, str] | None = None,
    extras: Iterable[str] | None = None,
    depth: int = 0,
) -> Iterator[str]:
    """Parse requirement specs from a list of lines.

    Parameters
    ----------
    requirements : `list` of `packaging.requirements.Requirement`
        The list of requirements to parse.

    conda_forge_map : `dict`
        `(pypi_name, conda_forge_name)` mapping dictionary

    environment : `dict`
        the environment against which to evaluate the marker

    extras : `list` of `str`
        list of extras to include in the environment marker evaluation

    depth : `int`
        internal variable that indicates the depth of this parsing,
        only used to help with logging

    Yields
    ------
    spec : `packaging.requirements.Requirement`
        a formatted requirement for each line
    """
    for entry in requirements:
        if not depth:  # print top-level requirements
            log.debug("  parsing %s", entry)
        req = Requirement(entry)
        # if environment markers don't pass, skip
        if not _evaluate_marker(
            req.marker,
            environment=environment,
            extras=extras,
        ):
            continue
        # if requirement is a URL, skip
        if req.url:
            continue
        # if requirement includes extras, parse those recursively
        yield from parse_req_extras(
            req,
            environment=environment,
            conda_forge_map=conda_forge_map,
        )
        # format as 'name{>=version}'
        yield from format_requirement(
            req,
            conda_forge_map=conda_forge_map,
        )


# -- requirements.txt -------

def parse_requirements_file(
    file: str | os.PathLike[str] | TextIO,
    **kwargs,
) -> Iterator[str]:
    """Parse a requirements.txt-format file."""
    if isinstance(file, str | os.PathLike):
        with Path(file).open() as fileobj:
            yield from parse_requirements_file(fileobj, **kwargs)
            return

    for line in map(str.strip, file):
        if (
            not line  # blank line
            or line.startswith("#")  # comment
            or "://" in line  # URL
        ):
            continue
        if line.startswith("-r "):
            yield from parse_requirements_file(line[3:].strip(), **kwargs)
        else:
            yield from parse_requirements([line], **kwargs)


def parse_all_requirements(
    project: Path,
    python_version: str | None = None,
    extras: Iterable[str] | str | None = None,
    dependency_groups: Iterable[str] | str | None = None,
    requirements_files: Iterable[Path | str] | None = None,
    *,
    skip_build_requires: bool = False,
) -> Iterator[str]:
    """Parse all requirements for a project.

    Parameters
    ----------
    project : `pathlib.Path`
        the base path of the project, or the path to a wheel file

    python_version : `str`, optional
        the ``'X.Y'`` python version to use

    extras : `list` of `str` or ``'ALL'``
        the list of extras to parse from the ``'options.extras_require'``
        key, or ``'ALL'`` to read all of them

    dependency_groups : `list` of `str` or ``'ALL'``
        the list of PEP 735 dependency groups to parse, or ``'ALL'``
        to read all of them

    requirements_files : `list` of `str`
        list of paths to Pip requirements.txt-format files that list
        package requirements.

    skip_build_requires : `bool`, optional
        if `True` skip parsing `build-requires` from `pyproject.toml` or
        `setup.cfg`

    Yields
    ------
    requirements : `str`
        a requirement spec str compatible with conda
    """
    # load the map from grayskull
    conda_forge_map = load_conda_forge_name_map()

    # parse project metadata
    if project.suffix == ".whl":
        meta = read_wheel_metadata(project)
    else:
        try:
            meta = build_project_metadata(project)
        except BuildException:
            if not requirements_files:
                # we need _something_ to work with
                raise
            meta = {}

    # generate environment for markers
    environment = {}

    # parse python version
    if python_version:
        # use user-given Python version to seed the marker environment
        parts = python_version.split(".")
        while len(parts) < 3:
            parts.append("0")
        environment["python_version"] = ".".join(parts[:2])
        environment["python_full_version"] = ".".join(parts)
    elif "requires_python" in meta:
        python_version = str(meta["requires_python"])
    if python_version:
        log.info("Using Python %s", python_version)
        if not python_version.startswith((">", "<", "=")):
            python_version = f"=={python_version}.*"
        yield from format_requirement(Requirement(f"python{python_version}"))

    # then build requirements
    if not skip_build_requires:
        log.info("Processing build-system/requires")
        for req in parse_requirements(
            meta.get("build_system_requires", []),
            environment=environment,
            conda_forge_map=conda_forge_map,
        ):
            log.debug("    parsed %s", req)
            yield req

    # then runtime requirements
    log.info("Processing requires_dist")
    if extras == "ALL":
        extras = meta["provides_extra"]
    for req in parse_requirements(
        meta.get("requires_dist", []),
        environment=environment,
        extras=extras,
        conda_forge_map=conda_forge_map,
    ):
        log.debug("    parsed %s", req)
        yield req

    # then requirements.txt files
    for reqfile in requirements_files or []:
        log.info("Processing %s", reqfile)
        for req in parse_requirements_file(
            reqfile,
            environment=environment,
            conda_forge_map=conda_forge_map,
        ):
            log.debug("    parsed %s", req)
            yield req

    # then dependency groups (PEP 735)
    if dependency_groups and project.suffix == ".whl":
        msg = "Cannot process dependency groups from a wheel file"
        raise ValueError(msg)
    if dependency_groups:
        log.info("Processing dependency groups")
        group_reqs = parse_dependency_groups(project, dependency_groups)
        for req in parse_requirements(
            group_reqs,
            environment=environment,
            conda_forge_map=conda_forge_map,
        ):
            log.debug("    parsed %s", req)
            yield req


# -- conda ------------------

def find_packages(
    requirements: set[str] | list[str],
    conda: str | Path = CONDA,
) -> subprocess.CompletedProcess[str]:
    """Run conda/mamba to resolve an environment.

    This does not actually create an environment, but is called so
    that if it fails because packages are missing, they can be
    identified.
    """
    conda = str(conda)
    with tempfile.TemporaryDirectory(prefix=Path(__file__).stem) as prefix:
        cmd = [
            str(conda),
            "create",  # solve for a new environment
            "--dry-run",  # don't actually do anything but solve
            "--json",  # print JSON-format output
            "--quiet",  # don't print logging info
            "--yes",  # don't ask questions
            "--override-channels",  # ignore user's conda config
            "--channel", "conda-forge",  # only look at conda-forge
            "--prefix", prefix,  # don't overwrite existing env by mistake!
            *sorted(requirements),
        ]

        if conda.lower().endswith(".bat"):
            # Windows (batch?) does weird things with angle brackets,
            # so we need to escape them in a weird way
            cmd = [re.sub("([><])", r"^^^\1", arg) for arg in cmd]
            # remove quotes from batch script name, powershell doesn't understand
            cmdstr = str(conda) + " " + shlex.join(cmd[1:])
        else:
            cmdstr = shlex.join(cmd)

        log.debug("$ %s", cmdstr)
        return subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            text=True,
        )


def filter_requirements(
    requirements: set[str] | list[str],
    conda: str | Path = CONDA,
) -> set[str]:
    """Filter requirements by running conda/mamba to see what is missing."""
    requirements = set(requirements)

    # find all packages with conda
    exe = Path(conda).stem
    log.info("Finding packages with %s", exe)
    pfind = find_packages(requirements, conda=conda)

    if pfind.returncode < 0:  # killed with signal
        pfind.check_returncode()  # raises
    if pfind.returncode:  # something went wrong
        # parse the JSON report
        report = json.loads(pfind.stdout)

        # report isn't a simple 'missing package' error
        if report.get("exception_name", None) != "PackagesNotFoundError":
            log.critical("\n".join((
                f"{exe} failed to resolve packages:",
                report.get("error", report.get("solver_problems", "unknown")),
            )))
            pfind.check_returncode()  # raises exception

        # one or more packages are missing
        log.warning(
            "%s failed to find some packages, "
            "attempting to parse what's missing",
            exe,
        )
        missing = {
            pkg.split("[", 1)[0].lower()  # strip out build info
            for pkg in report["packages"]
        }

        # filter out the missing packages
        for req in list(requirements):
            guesses = {
                # name with version (no whitespace)
                req.replace(" ", ""),
                # name only
                VERSION_OPERATOR.split(req)[0].strip().lower(),
            }
            if guesses & missing:  # package is missing
                log.warning("  removing '%s'", req)
                requirements.remove(req)

    return requirements


# -- output formatting ------

def write_yaml(path: Path, packages: Collection[str]) -> None:
    """Write the given ``packages`` as a conda environment YAML file."""
    env = {
        "channels": ["conda-forge"],
        "dependencies": packages,
    }
    with path.open("w") as file:
        yaml.dump(env, file)


# -- pip2conda main func ----

def pip2conda(
    project: Path,
    python_version: str | None = None,
    extras: list[str] | str | None = None,
    dependency_groups: list[str] | str | None = None,
    requirements_files: list[Path] | None = None,
    *,
    skip_build_requires: bool = False,
    skip_conda_forge_check: bool = False,
    conda: str | Path = CONDA,
) -> set[str]:
    """Parse requirements for a project and return conda packages."""
    # parse requirements
    requirements = set(parse_all_requirements(
        project,
        python_version=python_version,
        extras=extras,
        dependency_groups=dependency_groups,
        requirements_files=requirements_files,
        skip_build_requires=skip_build_requires,
    ))

    if skip_conda_forge_check:
        return requirements

    # filter out requirements that aren't available in conda-forge
    return filter_requirements(
        requirements,
        conda=conda,
    )


# -- command line operation -

def _get_prog() -> str:
    """Get the program name for the usage text."""
    if __name__ == "__main__":
        return Path(__file__).stem
    return __name__.rsplit(".", 1)[-1]


def create_parser() -> argparse.ArgumentParser:
    """Create a command-line `ArgumentParser` for this tool."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        prog=_get_prog(),
    )

    # DEPRECATED positional argument
    parser.add_argument(
        "positional_extras",
        metavar="EXTRA",
        nargs="*",
        default=[],
        help=(
            "name of setuptools 'extras' to parse [DEPRECATED, use -e/--extra instead]"
        ),
    )

    parser.add_argument(
        "-e",
        "--extra",
        metavar="EXTRA",
        dest="extras",
        action="append",
        default=[],
        help=(
            "include optional dependencies from the specified extra name; may be"
            "provided more than once"
        ),
    )
    parser.add_argument(
        "-g",
        "--group",
        "--dependency-group",
        dest="dependency_groups",
        metavar="GROUP",
        default=[],
        action="append",
        help=(
            "Install the specified dependency group from a "
            "`pylock.toml` or `pyproject.toml`"
        ),
    )
    parser.add_argument(
        "-a",
        "--all-extras",
        "--all",
        action="store_true",
        default=False,
        help="include all extras",
    )
    parser.add_argument(
        "-G",
        "--all-groups",
        action="store_true",
        default=False,
        help="include all dependency groups",
    )
    parser.add_argument(
        "-b",
        "--no-build-requires",
        action="store_true",
        default=False,
        help="skip parsing of build-requires from pyproject.toml or setup.cfg",
    )
    parser.add_argument(
        "-r",
        "--requirements",
        type=Path,
        default=[],
        action="extend",
        help="path of Pip requirements file to parse",
        nargs="*",
    )
    parser.add_argument(
        "-d",
        "--project",
        "--project-dir",
        "--wheel",
        default=Path.cwd(),
        type=Path,
        help="project directory, or path to wheel",
    )
    parser.add_argument(
        "-p",
        "--python-version",
        default=None,
        help="python X.Y version to use",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "path of output file, defaults to stdout; if the --output "
            "argument ends with .yml or .yaml, output will be written in "
            "as a conda environment YAML file, otherwise a simple "
            "requirements.txt-style text file will be written"
        ),
    )
    parser.add_argument(
        "-C",
        "--conda",
        default=CONDA,
        type=Path,
        help="Conda/mamba executable to call",
    )
    parser.add_argument(
        "-s",
        "--skip-conda-forge-check",
        action="store_true",
        default=False,
        help="skip checking that packages exist in conda-forge",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="print verbose logging",
    )
    return parser


def main(args: list[str] | None = None) -> None:
    """Run the thing."""
    parser = create_parser()
    opts = parser.parse_args(args=args)

    # configure logging
    root_logger = logging.getLogger()
    if not root_logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setFormatter(Formatter(
            fmt="%(asctime)s:%(name)s[%(process)d]:%(levelname)+8s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        ))
        root_logger.addHandler(handler)
        root_logger.setLevel(max(3 - opts.verbose, 0) * 10)

    # handle extras
    if opts.positional_extras:
        warnings.warn(
            "Positional extras are deprecated, use -e/--extra instead",
            DeprecationWarning,
            stacklevel=2,
        )
        opts.extras.extend(opts.positional_extras)

    # run the thing
    requirements = sorted(pip2conda(
        opts.project,
        python_version=opts.python_version,
        extras="ALL" if opts.all_extras else opts.extras,
        dependency_groups="ALL" if opts.all_groups else opts.dependency_groups,
        requirements_files=opts.requirements,
        skip_build_requires=opts.no_build_requires,
        skip_conda_forge_check=opts.skip_conda_forge_check,
        conda=opts.conda,
    ))
    log.info("Package finding complete")

    # print output to file or stdout
    out = "\n".join(requirements)
    if opts.output and opts.output.suffix in {".yml", ".yaml"}:
        write_yaml(opts.output, requirements)
    elif opts.output:
        opts.output.write_text(out + "\n")
    else:
        print(out)
