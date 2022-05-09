# -*- coding: utf-8 -*-
# Copyright (C) Cardiff University (2022)
# SPDX-License-Identifier: GPL-3.0-or-later

"""Parse setup.cfg for package requirements and print out a list of
packages that can be installed using conda from the conda-forge channel.
"""

import argparse
import json
import logging
import os
import re
import subprocess
import tempfile
from configparser import ConfigParser
from pathlib import Path
from shutil import which

import requests

import tomli

import pkg_resources

from grayskull.strategy.pypi import PYPI_CONFIG
from ruamel.yaml import YAML

yaml = YAML()

# conda config
CONDA = (
    which("conda")
    or os.environ.get("CONDA_EXE", "conda")
)
CONDA_OR_MAMBA = which("mamba") or CONDA

# configure logging
LOGGER = logging.getLogger(__name__.rsplit(".", 1)[-1])
try:
    from coloredlogs import ColoredFormatter as _Formatter
except ImportError:
    _Formatter = logging.Formatter
if not LOGGER.hasHandlers():
    _LOG_HANDLER = logging.StreamHandler()
    _LOG_HANDLER.setFormatter(_Formatter(
        fmt="[%(asctime)s] %(levelname)+8s: %(message)s",
    ))
    LOGGER.addHandler(_LOG_HANDLER)

# regex to match version spec characters
VERSION_OPERATOR = re.compile("[><=!]")


def load_conda_forge_name_map():
    """Load the PyPI <-> conda-forge package name map from grayskull

    See https://github.com/conda-incubator/grayskull/blob/main/grayskull/pypi/config.yaml
    """  # noqa: E501
    # parse the config file and return (pypi_name: conda_forge_name) pairs
    with open(PYPI_CONFIG, "r") as conf:
        return {
            x: y["conda_forge"]
            for x, y in yaml.load(conf).items()
        }


def format_requirement(requirement, conda_forge_map=dict()):
    """Format a (pip) Requirement as a conda dependency

    Parameters
    ----------
    requirement : `pkg_resources.Requirement`
        the requirement to format

    conda_forge_map : `dict`
        `(pypi_name, conda_forge_name)` mapping dictionary

    Returns
    -------
    formatted : `str`
        the formatted conda requirement

    Examples
    --------
    >>> import pkg_resources
    >>> req = pkg_resources.Requirement.parse("htcondor >= 9.0.0")
    >>> print(format_requirement(req))
    'python-htcondor>=9.0.0'
    """
    return (
        conda_forge_map.get(requirement.name, requirement.name.lower())
        + str(requirement.specifier)
    ).strip()


def parse_extras(req, conda_forge_map=dict()):
    """Parse the extras for a requirement

    This unpackes a requirement like `package[extra]` into the list of
    actual packages that are required, and yields formatted conda
    dependencies.

    Parameters
    ----------
    req : `pkg_resources.Requirement`
        the requirement to format

    conda_forge_map : `dict`
        `(pypi_name, conda_forge_name)` mapping dictionary
    """
    if not req.extras:
        return

    # query pypi for metadata
    resp = requests.get(f"https://pypi.org/pypi/{req.name}/json")
    resp.raise_for_status()
    data = resp.json()

    # parse the requirements that match the requested extras
    yield from parse_requirements(
        data["info"]["requires_dist"],
        conda_forge_map=conda_forge_map,
        extras=req.extras,
        depth=1,
    )


def _evaluate_marker(marker, extras=None):
    """Evaluate whether an environment marker matches this environment
    """
    if not marker:  # no marker, always True
        return True
    try:
        return marker.evaluate()  # built-in environment
    except ValueError:
        extras = extras or []
        # marker includes extras (probably), evaluate for any of the given
        # extras
        return any(marker.evaluate({"extra": extra}) for extra in extras)


def parse_requirements(lines, conda_forge_map=dict(), extras=None, depth=0):
    """Parse requirement specs from a list of lines

    Parameters
    ----------
    lines : iterable of `str`
        input to iterate over, e.g. a `list` of package names, or
        a `file` object

    conda_forge_map : `dict`
        `(pypi_name, conda_forge_name)` mapping dictionary

    extras : `list` of `str`
        list of extras to include in the environment marker evaluation

    depth : `int`
        internal variable that indicates the depth of this parsing,
        only used to help with logging

    Yields
    ------
    spec : `pkg_resources.Requirement`
        a formatted requirement for each line
    """
    for item in pkg_resources.parse_requirements(lines):
        if not depth:  # print top-level requirements
            LOGGER.debug(f"  parsing {item}")
        # if environment markers don't pass, skip
        if not _evaluate_marker(item.marker, extras=extras):
            continue
        # if requirement is a URL, skip
        if item.url:
            continue
        # if requirement includes extras, parse those recursively
        yield from parse_extras(item, conda_forge_map=conda_forge_map)
        # format as 'name{>=version}'
        yield format_requirement(item, conda_forge_map=conda_forge_map)


def parse_build_requires(project_dir):
    """Parse the build requirements for a project

    This reads from either pyproject.toml (preferred) or setup.cfg.
    """
    project_dir = Path(project_dir)
    pyproject_toml = project_dir / "pyproject.toml"
    try:
        with open(pyproject_toml, "rb") as file:
            ppt = tomli.load(file)
    except FileNotFoundError:
        LOGGER.debug("  failed to read pyproject.toml, reading setup.cfg")
        setup_cfg = project_dir / "setup.cfg"
        conf = ConfigParser()
        conf.read(setup_cfg)
        return conf["options"]["setup_requires"].strip().splitlines()
    else:
        return ppt["build-system"]["requires"]


def parse_requirements_file(file, **kwargs):
    """Parse a requirements.txt-format file
    """
    print(type(file), file)

    if isinstance(file, (str, os.PathLike)):
        with open(file, "r") as fileobj:
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
            yield from parse_requirements(line, **kwargs)


def parse_all_requirements(
    project_dir,
    python_version=None,
    extras=[],
    requirements_files=[],
    skip_build_requires=False,
):
    """Parse all requirements for a project

    Parameters
    ----------
    project_dir : `pathlib.Path`
        the base path of the project

    python_version : `str`, optional
        the ``'X.Y'`` python version to use

    extras : `list` of `str` or ``'ALL'``
        the list of extras to parse from the ``'options.extras_require'``
        key, or ``'ALL'`` to read all of them

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

    # add python first
    if python_version:
        LOGGER.info(f"Using Python {python_version}")
        yield f"python={python_version}.*"

    # then build requirements
    if not skip_build_requires:
        LOGGER.info("Processing build-requires")
        for req in parse_requirements(
            parse_build_requires(project_dir),
            conda_forge_map=conda_forge_map,
        ):
            LOGGER.debug(f"    parsed {req}")
            yield req

    # then setup.cfg options
    setup_cfg = project_dir / "setup.cfg"
    if setup_cfg.exists():
        LOGGER.info(f"Processing {setup_cfg}")
        for req in parse_setup_cfg(
            setup_cfg,
            extras=extras,
            conda_forge_map=conda_forge_map,
        ):
            LOGGER.debug(f"    parsed {req}")
            yield req

    # then requirements.txt files
    for reqfile in requirements_files:
        LOGGER.info(f"Processing {reqfile}")
        for req in parse_requirements_file(
            reqfile,
            conda_forge_map=conda_forge_map,
        ):
            LOGGER.debug(f"    parsed {req}")
            yield req


def parse_setup_cfg(path, extras=None, conda_forge_map=dict()):
    conf = ConfigParser()
    with open(path, "r") as file:
        conf.read_file(file)

    if extras == "ALL":  # use all extras
        extras = conf["options.extras_require"].keys()

    options = [
        ("options", "install_requires"),
    ] + [("options.extras_require", extra) for extra in extras or []]

    for sect, opt in options:
        LOGGER.info(f"Processing {sect}/{opt}")
        lines = conf[sect][opt].strip().splitlines()
        for req in parse_requirements(lines, conda_forge_map=conda_forge_map):
            LOGGER.debug(f"    parsed {req}")
            yield req


def find_packages(requirements, use_mamba=True):
    """Run conda/mamba to resolve an environment

    This does not actually create an environment, but is called so
    that if it fails because packages are missing, they can be
    identified.
    """
    prefix = tempfile.mktemp(prefix=Path(__file__).stem)
    EXE = CONDA_OR_MAMBA if use_mamba else CONDA
    use_mamba = "mamba" in os.path.basename(EXE)
    cmd = [
        EXE,
        "create",  # solve for a new environment
        "--dry-run",  # don't actually do anything but solve
        "--json",  # print JSON-format output
        "--quiet",  # don't print logging info
        "--yes",  # don't ask questions
        "--override-channels",  # ignore user's conda config
        "--channel", "conda-forge",  # only look at conda-forge
        "--prefix", prefix,  # don't overwrite existing env by mistake!
    ]

    # we use weird quoting here so that when the command is printed
    # to the log, PowerShell users can copy it and run it verbatim
    # without ps seeing '>' and piping output
    cmd.extend((f'"""{req}"""' for req in requirements))

    LOGGER.debug(f"$ {' '.join(cmd)}")
    pfind = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        text=True,
    )

    if pfind.returncode:
        # search failed; if we can't use the output to parse missing
        # packages because we're using mamba, we need to try again
        # with conda, which definitely outputs json...
        try:
            json.loads(pfind.stdout)
        except json.JSONDecodeError:
            if not use_mamba:
                raise
            LOGGER.debug(
                "mamba search failed and didn't report JSON:\n"
                f"{pfind.stdout}".rstrip()
            )
            LOGGER.debug("trying again with conda")
            return(find_packages(requirements, use_mamba=False))

    return pfind


def filter_requirements(requirements, use_mamba=True):
    """Filter requirements by running conda/mamba to see what is missing
    """
    requirements = set(requirements)

    # find all packages with conda
    LOGGER.info("Finding packages with conda/mamba")
    pfind = find_packages(requirements, use_mamba=use_mamba)

    if pfind.returncode:  # something went wrong
        # parse the JSON report
        report = json.loads(pfind.stdout)

        # report isn't a simple 'missing package' error
        if report["exception_name"] != "PackagesNotFoundError":
            LOGGER.critical("\n".join((
                "conda/mamba failed to resolve packages:",
                report["error"],
            )))
            pfind.check_returncode()  # raises exception

        # one or more packages are missing
        LOGGER.warning(
            "conda/mamba failed to find some packages, "
            "attempting to parse what's missing",
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
                LOGGER.warning(f"  removing {req!r}")
                requirements.remove(req)

    return requirements


def write_yaml(path, packages):
    """Write the given ``packages`` as a conda environment YAML file
    """
    env = {
        "channels": ["conda-forge"],
        "dependencies": packages,
    }
    with open(path, "w") as file:
        yaml.dump(env, file)


def create_parser():
    """Create a command-line `ArgumentParser` for this tool
    """
    if __name__ == "__main__":
        prog = __module__.__name__  # noqa: F821
    else:
        prog = __name__.rsplit(".", 1)[-1]
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        prog=prog,
    )
    parser.add_argument(
        "extras_name",
        nargs="*",
        default=[],
        help="name of setuptools 'extras' to parse",
    )
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        default=False,
        help="include all extras",
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
        "--project-dir",
        default=os.getcwd(),
        type=Path,
        help="project base directory",
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
        "-M",
        "--disable-mamba",
        action="store_true",
        default=False,
        help="don't use mamba, even if it is available",
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


def pip2conda(
        project_dir,
        python_version=None,
        extras=[],
        requirements_files=[],
        skip_build_requires=False,
        skip_conda_forge_check=False,
        use_mamba=True,
):
    # parse requirements
    requirements = parse_all_requirements(
        project_dir,
        python_version=python_version,
        extras=extras,
        requirements_files=requirements_files,
        skip_build_requires=skip_build_requires,
    )

    if skip_conda_forge_check:
        return requirements

    # filter out requirements that aren't available in conda-forge
    return filter_requirements(
        requirements,
        use_mamba=use_mamba,
    )


def main(args=None):
    """Run the thing
    """
    parser = create_parser()
    args = parser.parse_args(args=args)

    # set verbose logging
    LOGGER.setLevel(max(3 - args.verbose, 0) * 10)

    # show what conda/mamba we found
    LOGGER.debug(f"found conda in {CONDA}")
    if CONDA_OR_MAMBA != CONDA and not args.disable_mamba:
        LOGGER.debug(f"found mamba in {CONDA_OR_MAMBA}")

    # run the thing
    requirements = sorted(pip2conda(
        args.project_dir,
        python_version=args.python_version,
        extras="ALL" if args.all else args.extras_name,
        requirements_files=args.requirements,
        skip_build_requires=args.no_build_requires,
        skip_conda_forge_check=args.skip_conda_forge_check,
        use_mamba=not args.disable_mamba,
    ))
    LOGGER.info("Package finding complete")

    # print output to file or stdout
    out = "\n".join(requirements)
    if args.output and args.output.suffix in {".yml", ".yaml"}:
        write_yaml(args.output, requirements)
    elif args.output:
        args.output.write_text(out + "\n")
    else:
        print(out)
