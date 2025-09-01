<!-- markdownlint-disable MD024 -->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

(v0.8.0)=

## 0.8.0 - 2025-09-01

-   Fix safe-netrc version reference
    ([!42](https://gitlab.com/gwpy/pip2conda/-/merge_requests/42))
-   Improve use of logging
    ([!43](https://gitlab.com/gwpy/pip2conda/-/merge_requests/43))
-   Add support for dependency groups
    ([!49](https://gitlab.com/gwpy/pip2conda/-/merge_requests/49))
-   Add type annotations
    ([!51](https://gitlab.com/gwpy/pip2conda/-/merge_requests/51))
-   Add changelog and rebuild sphinx docs
    ([!53](https://gitlab.com/gwpy/pip2conda/-/merge_requests/53))

[Release details](https://gitlab.com/gwpy/pip2conda/-/releases/0.8.0)

(v0.7.1)=

## 0.7.1 - 2025-02-14

-   Remove unused call to json.loads
    ([!40](https://gitlab.com/gwpy/pip2conda/-/merge_requests/40))
-   Handle posix signal termination for conda create
    ([!41](https://gitlab.com/gwpy/pip2conda/-/merge_requests/41))

[Release details](https://gitlab.com/gwpy/pip2conda/-/releases/0.7.1)

(v0.7.0)=

## 0.7.0 - 2025-01-07

-   Separate compound version specifiers
    ([!39](https://gitlab.com/gwpy/pip2conda/-/merge_requests/39))

[Release details](https://gitlab.com/gwpy/pip2conda/-/releases/0.7.0)

(v0.6.1)=

## 0.6.1 - 2025-01-06

-   Fix character escaping on Windows
    ([!38](https://gitlab.com/gwpy/pip2conda/-/merge_requests/38))

[Release details](https://gitlab.com/gwpy/pip2conda/-/releases/0.6.1)

(v0.6.0)=

## 0.6.0 - 2025-01-06

-   PyPI trusted publishing workflow
    ([!37](https://gitlab.com/gwpy/pip2conda/-/merge_requests/37))

[Release details](https://gitlab.com/gwpy/pip2conda/-/releases/0.6.0)

(v0.5.1)=

## 0.5.1 - 2023-09-06

-   Add wheel as a project requirement
    ([!25](https://gitlab.com/gwpy/pip2conda/-/merge_requests/25))
-   Update isolated environment builds for python-build 1.0.0
    ([!26](https://gitlab.com/gwpy/pip2conda/-/merge_requests/26))

[Release details](https://gitlab.com/gwpy/pip2conda/-/releases/0.5.1)

(v0.5.0)=

## 0.5.0 - 2023-08-23

-   Support for evaluating requirements from a wheel file
    ([!24](https://gitlab.com/gwpy/pip2conda/-/merge_requests/24))

[Release details](https://gitlab.com/gwpy/pip2conda/-/releases/0.5.0)

(v0.4.2)=

## 0.4.2 - 2023-01-12

-   Fix evaluating markers with extras
    ([!22](https://gitlab.com/gwpy/pip2conda/-/merge_requests/22))

[Release details](https://gitlab.com/gwpy/pip2conda/-/releases/0.4.2)

(v0.4.1)=

## 0.4.1 - 2022-11-08

-   Add support for python_version in marker environments
    ([!20](https://gitlab.com/gwpy/pip2conda/-/merge_requests/20))
-   Formal support for Python 3.11
    ([!21](https://gitlab.com/gwpy/pip2conda/-/merge_requests/21))

[Release details](https://gitlab.com/gwpy/pip2conda/-/releases/0.4.1)

(v0.4.0)=

## 0.4.0 - 2022-10-07

-   Refactor to use `build` to generate metadata, rather than parsing myriad
    configuration files
    ([!17](https://gitlab.com/gwpy/pip2conda/-/merge_requests/17))

[Release details](https://gitlab.com/gwpy/pip2conda/-/releases/0.4.0)

(v0.3.2)=

## 0.3.2 - 2022-05-09

-   Remove leftover debug print call
    ([!14](https://gitlab.com/gwpy/pip2conda/-/merge_requests/14))

**Note:** This is a corrected release of 0.3.1 which didn't actually include
the necessary fixes.

[Release details](https://gitlab.com/gwpy/pip2conda/-/releases/0.3.2)

(v0.3.1)=

## 0.3.1 - 2022-05-09

-   Remove leftover debug print call
    ([!14](https://gitlab.com/gwpy/pip2conda/-/merge_requests/14))

[Release details](https://gitlab.com/gwpy/pip2conda/-/releases/0.3.1)

(v0.3.0)=

## 0.3.0 - 2022-05-09

-   Enable parsing of requirements.txt files
    ([!13](https://gitlab.com/gwpy/pip2conda/-/merge_requests/13))
-   Add `--no-build-requires` command-line option
    ([!13](https://gitlab.com/gwpy/pip2conda/-/merge_requests/13))

[Release details](https://gitlab.com/gwpy/pip2conda/-/releases/0.3.0)

(v0.2.0)=

## 0.2.0 - 2022-02-03

-   Support for YAML-format output files
    ([!7](https://gitlab.com/gwpy/pip2conda/-/merge_requests/7))
-   New `--disable-mamba` command-line option
    ([!9](https://gitlab.com/gwpy/pip2conda/-/merge_requests/9))

[Release details](https://gitlab.com/gwpy/pip2conda/-/releases/0.2.0)

(v0.1.0)=

## 0.1.0 - 2022-02-01

- Initial release of pip2conda
- Extracted from GWpy source repository
- Core functionality for translating pip requirements into conda requirements

[Release details](https://gitlab.com/gwpy/pip2conda/-/releases/0.1.0)

**Note:** This is the first release of this project, which was extracted from
the source repository for GWpy by the author, and released under the same license.
