# PUMA: Plotting UMami Api

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PUMA Docs](https://img.shields.io/badge/info-documentation-informational)](https://umami-hep.github.io/puma/)
[![PyPI version](https://badge.fury.io/py/puma-hep.svg)](https://badge.fury.io/py/puma-hep)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6607414.svg)](https://doi.org/10.5281/zenodo.6607414)

[![codecov](https://codecov.io/gh/umami-hep/puma/branch/main/graph/badge.svg)](https://codecov.io/gh/umami-hep/puma)
![Testing workflow](https://github.com/umami-hep/puma/actions/workflows/testing.yml/badge.svg)
![Linting workflow](https://github.com/umami-hep/puma/actions/workflows/linting.yml/badge.svg)
![Pages workflow](https://github.com/umami-hep/puma/actions/workflows/pages.yml/badge.svg)
![Docker build workflow](https://github.com/umami-hep/puma/actions/workflows/docker_build.yml/badge.svg)

The Python package `puma` provides a plotting API for commonly used plots in flavour tagging.

|                                     ROC curves                                      |                                            Histogram plots                                             |                                    Variable vs efficiency                                    |
| :---------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------: |
| <img src=https://github.com/umami-hep/puma/raw/examples-material/roc.png width=200> | <img src=https://github.com/umami-hep/puma/raw/examples-material/histogram_discriminant.png width=220> | <img src=https://github.com/umami-hep/puma/raw/examples-material/pt_light_rej.png width=220> |

## Installation

`puma` can be installed from [PyPI](https://pypi.org/project/puma-hep/) or using the latest code from this repository.

### Install latest release from PyPI

```bash
pip install puma-hep
```

The installation from PyPI only allows to install tagged releases, meaning you can not
install the latest code from this repo using the above command.
If you just want to use a stable release of `puma`, this is the way to go.

### Install latest version from GitHub

```bash
pip install https://github.com/umami-hep/puma/archive/main.tar.gz
```

This will install the latest version of `puma`, i.e. the current version
from the `main` branch (no matter if it is a release/tagged commit).
If you plan on contributing to `puma` and/or want the latest version possible, this
is what you want.

### Install for development with `uv` (recommended)

For development, we recommend using [`uv`](https://docs.astral.sh/uv/), a fast Python package installer and resolver. First, install `uv`:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip (If installing from PyPI, we recommend installing uv into an isolated environment)
pip install uv
```

Then clone the repository and install `puma` with development dependencies:

```bash
git clone https://github.com/umami-hep/puma.git
cd puma
uv sync --extra dev
```

This will install `puma` in editable mode along with all development tools (testing, linting, etc.).

> [!TIP]
> In order to use locally installed version of `puma` in other `uv`-managed projects, you can add the following to the `pyproject.toml` of the other project:
> ```toml
> [tool.uv.sources]
> puma-hep = { path = "path_to/puma" }
> ```

## Docker images

The Docker images are built on GitHub and contain the latest version from the `main` branch.

The container registry with all available tags can be found
[here](https://gitlab.cern.ch/aft/training-images/puma-images/container_registry/13727).

The `puma:latest` image is based on `python:3.11.10-bullseye` and is meant for users who want to use the latest version of `puma`. For each release, there is a corresponding tagged image.
You can start an interactive shell in a container with your current working directory
mounted into the container by using one of the commands provided below.

On a machine with Docker installed:

```bash
docker run -it --rm -v $PWD:/puma_container -w /puma_container gitlab-registry.cern.ch/aft/training-images/puma-images/puma:latest bash
```

On a machine/cluster with singularity installed:

```bash
singularity shell -B $PWD docker://gitlab-registry.cern.ch/aft/training-images/puma-images/puma:latest
```

**The images are automatically updated via GitHub and pushed to this [repository registry](https://gitlab.cern.ch/aft/training-images/puma-images/container_registry).**