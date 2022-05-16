# puma - Plotting UMami Api

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Umami docs](https://img.shields.io/badge/info-documentation-informational)](https://umami-hep.github.io/puma/)
[![PyPI version](https://badge.fury.io/py/puma-hep.svg)](https://badge.fury.io/py/puma-hep)

![Testing workflow](https://github.com/umami-hep/puma/actions/workflows/testing.yml/badge.svg)
![Linting workflow](https://github.com/umami-hep/puma/actions/workflows/linting.yml/badge.svg)
![Pages workflow](https://github.com/umami-hep/puma/actions/workflows/pages.yml/badge.svg)

The Python package `puma` provides a plotting API for commonly used plots in flavour tagging.

|                              ROC curves                              |                                     Histogram plots                                     |                            Variable vs efficiency                             |
| :------------------------------------------------------------------: | :-------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------: |
| <img src=https://github.com/umami-hep/puma/raw/examples-material/roc.png width=200> | <img src=https://github.com/umami-hep/puma/raw/examples-material/histogram_discriminant.png width=220> | <img src=https://github.com/umami-hep/puma/raw/examples-material/pt_light_rej.png width=220> |


## Installation

The `puma` package is currently under construction. It can be installed from PyPI or
using the latest code from this repository.

### Install latest release from PyPI

```bash   
pip install puma-hep
```

The installation from PyPI only allows to install tagged releases, meaning you can not
install the latest code from this repo using the above command.
If you just want to use a stable release of `puma`, this is the way to go.

### Install latest version from GitHub
```bash   
pip install https://github.com/umami-hep/puma/archive/master.tar.gz
```

This will install the latest version of `puma`, i.e. the current version
from the `main` branch (no matter if it is a release/tagged commit).
If you plan on contributing to `puma` and/or want the latest version possible, this
is what you want.
