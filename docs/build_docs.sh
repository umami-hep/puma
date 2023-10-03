#!/bin/bash

# install requirements
pip install -r requirements.txt

# install docs requirements
pip install -r docs/requirements.txt

# add current working directory to PYTHONPATH such that package is found
export PYTHONPATH=$PWD:$PYTHONPATH

# build the documentation
rm -rf docs/_*
python docs/sphinx_build_multiversion.py
# copy the redirect_index.html that redirects to the main/latest version

mkdir -p docs/_build/html/
cp docs/source/redirect_index.html docs/_build/html/index.html

# we have to create an empty .nojekyll file in order to make the html theme work
touch docs/_build/html/.nojekyll
