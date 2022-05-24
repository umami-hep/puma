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
    "myst_parser",
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "sphinx_multiversion",
]

# Configuration of sphinx-autoapi extension
# https://sphinx-autoapi.readthedocs.io/en/latest/reference/config.html#
autoapi_type = "python"
autoapi_dirs = ["../../puma"]
autoapi_python_use_implicit_namespaces = True
autoapi_python_class_content = "both"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
html_sidebars = {
    "**": [
        "versioning.html",
    ],
}


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

default_role = "code"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/umami-hep/puma",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        },
    ],
}

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "friendly"
