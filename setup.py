"""
Install the package and all its dependencies.

Run

    python -m pip install .

to install the package.
"""

# read the contents of the README.md file
from __future__ import annotations

from pathlib import Path

from setuptools import setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    long_description=long_description,
    long_description_content_type="text/markdown",
)
