# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
from subprocess import check_output

import requests

import puma

sys.path.insert(0, os.path.abspath("../../puma"))


# -- Project information -----------------------------------------------------

project = "puma"
copyright = "2022, puma developers"
author = "puma developers"

# The full version, including alpha/beta/rc tags
release = ""

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "myst_parser",  # to include markdown files in the documentation
    "sphinx.ext.napoleon",
    "autoapi.extension",  # generates the API section of our documentation
    "sphinx_copybutton",  # adds a copy-button to each code cell
]

# -- sphinx-autoapi extension -----------------------------------------------
# https://sphinx-autoapi.readthedocs.io/en/latest/reference/config.html#
autoapi_type = "python"
autoapi_dirs = ["../../puma"]
autoapi_python_use_implicit_namespaces = True
autoapi_python_class_content = "both"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_static_path = ["_static"]
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

default_role = "code"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
json_url = "https://umami-hep.github.io/puma/main/_static/switcher.json"
release = puma.__version__

# get git hash we are currently on (when building the docs)
current_hash = check_output(["git", "rev-parse", "HEAD"]).decode("ascii").split("\n")[0]
# get git hash of latest commit on main branch
commits = requests.get("https://api.github.com/repos/umami-hep/puma/commits/main")
latest_commit_hash = commits.json()["sha"]

if current_hash == latest_commit_hash:
    version_match = "latest"
else:
    version_match = f"v{release}"

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "external_links": [
        {
            "url": "https://github.com/umami-hep/puma/blob/main/changelog.md",
            "name": "Changelog",
        },
    ],
    "logo": {
        "text": "puma documentation",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/umami-hep/puma",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        },
    ],
    "switcher": {
        "json_url": json_url,
        "version_match": version_match,
    },
    "navbar_end": ["theme-switcher", "version-switcher", "navbar-icon-links"],
}
html_context = {
    "default_mode": "light",  # default for light/dark theme
}

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "friendly"
