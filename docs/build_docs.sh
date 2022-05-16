#!/bin/bash

# install requirements and puma
pip install .

# install requirements for sphinx
pip install -r docs/requirements.txt

# build the documentation
cd docs
rm -rf _build _static _templates
sphinx-build -b html source _build/html

# we have to create an empty .nojekyll file in order to make the html theme work
touch _build/html/.nojekyll