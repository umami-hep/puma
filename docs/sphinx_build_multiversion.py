"""Script to loop over a number of git branches/tags, check them out and build the
the sphinx docs"""

import json
import os
from shutil import copy
from subprocess import run

import pygit2  # pylint: disable=E0401

with open("docs/source/_static/switcher.json", "r") as f:  # pylint: disable=W1514
    version_switcher = json.load(f)

# initial_branch = pygit2.Repository(".").head.shorthand
initial_branch = "jobirk-docs-add-version-button"
copy("docs/source/conf.py", "./conf_latest.py")

for entry in version_switcher:
    # build the documentation
    print(f"Building docs for branch/tag {entry['version']}")

    # checkout the version/tag and obtain latest conf.py
    run(f"git checkout {entry['version']}", shell=True, check=True)
    if os.path.isfile("docs/source/conf.py"):
        print("removing the old conf.py file")
        os.remove("docs/source/conf.py")
    copy("conf_latest.py", "docs/source/conf.py")

    # run librep on markdown files (render placeholders with sytax §§§filename§§§)
    run(
        "librep --ref_dir $PWD --input 'docs/**/*.md' --no_backup",
        shell=True,
        check=True,
    )

    # build the docs for this version
    run(
        f"sphinx-build -b html docs/source docs/_build/html/{entry['version']}",
        shell=True,
        check=True,
    )
    run("git stash", shell=True, check=True)

# checkout initial branch for following steps
run(f"git checkout {initial_branch}", shell=True, check=True)
