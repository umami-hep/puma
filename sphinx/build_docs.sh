#!/bin/bash

# install requirements and puma
pip install -r requirements.txt
source run_setup.sh

# install requirements for sphinx
pip install -r sphinx/requirements.txt

# build the documentation
cd sphinx
rm -rf _build _static _templates source
mkdir source
sphinx-apidoc -f -o . ../puma
make html

# copy html files to docs folder
mkdir -p ../docs
mv html/* ../docs

# we have to create an empty .nojekyll file in order to make the html theme work
touch ../docs/.nojekyll
