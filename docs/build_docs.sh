#!/bin/bash

# install requirements and puma
pip install .

# install requirements for sphinx
pip install -r docs/requirements.txt

# build the documentation
cd docs
rm -rf _build _static _templates
python sphinx_build_multiversion.py
# copy the redirect_index.html that redirects to the main/latest version
cp source/redirect_index.html _build/html/index.html

# we have to create an empty .nojekyll file in order to make the html theme work
touch _build/html/.nojekyll
