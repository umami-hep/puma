#!/bin/bash

# install requirements and puma
# pip install .
pip install -r requirements.txt

# install requirements for sphinx
pip install -r docs/requirements.txt

# build the documentation
rm -rf docs/_*
python docs/sphinx_build_multiversion.py
# copy the redirect_index.html that redirects to the main/latest version
cp docs/source/redirect_index.html docs/_build/html/index.html

# we have to create an empty .nojekyll file in order to make the html theme work
touch docs/_build/html/.nojekyll
