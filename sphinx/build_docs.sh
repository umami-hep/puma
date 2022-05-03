#!/bin/bash

# install requirements and puma
pip install -r requirements.txt
source run_setup.sh
# install requirements for sphinx
pip install -r sphinx/requirements.txt
cd sphinx
rm -rf _build _static _templates source
mkdir source
sphinx-apidoc -f -o . ../puma
make html
mkdir -p ../docs
mv html/* ../docs
# we have to create an empty .nojekyll file in order to make the html theme work
touch ../docs/.nojekyll
