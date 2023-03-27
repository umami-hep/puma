# puma - Plotting UMami Api

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Umami docs](https://img.shields.io/badge/info-documentation-informational)](https://umami-hep.github.io/puma/)
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

`puma` can be installed from PyPI or using the latest code from this repository.

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

## Docker images

The Docker images are built on GitHub and contain the latest version from the `main` branch.

The container registry with all available tags can be found
[here](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-images/puma-images/container_registry/13727).

The `puma:latest` image is based on `python:3.8.15-slim` and is meant for users who want to use the latest version of `puma`. For each release, there is a corresponding tagged image.
You can start an interactive shell in a container with your current working directory
mounted into the container by using one of the commands provided below.

On a machine with Docker installed:

```bash
docker run -it --rm -v $PWD:/puma_container -w /puma_container gitlab-registry.cern.ch/atlas-flavor-tagging-tools/training-images/puma-images/puma:latest bash
```

On a machine/cluster with singularity installed:

```bash
singularity shell -B $PWD docker://gitlab-registry.cern.ch/atlas-flavor-tagging-tools/training-images/puma-images/puma:latest
```

### Extended image for development

_For development, just replace the tag of the image_:

`latest` -> `latest-dev`

In addition to the minimal requirements that are required to use `puma`, the
`puma:latest-dev` image has the `requirements.txt` from the `puma` repo installed as
well.
This means that packages like `pytest`, `black`, `pylint`, etc. are installed as well.
However, note that `puma` itself is not installed in that image such that the dev-version
on your machine can be used/tested.

**The images are automatically updated via GitHub and pushed to this [repository registry](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-images/puma-images/container_registry).**
