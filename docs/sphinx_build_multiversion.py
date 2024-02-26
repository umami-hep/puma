"""Script to loop over a number of git branches/tags, check them out and build the
the sphinx docs.
The branches/tags for which the docs are generated are defined in the file
"docs/source/_static/switcher.json" (note that the docs do not use the local version
by default, but instead the version from the latest commit on GH-Pages. This is due to
the fact that this button prefers urls instead of local filenames).

This script has to be executed in the root of the repository!
"""

from __future__ import annotations

import os
from shutil import copy
from subprocess import run


def build_docs_version(version):
    """Builds the docs for a specific version. The latest conf.py is used no matter
    if it differs from the version from back then.
    This function expects the file "conf_latest.py" to exist in the current working
    directory.

    Parameters
    ----------
    version : str
        Branch or tag name for which the docs are built
    """
    # checkout the version/tag and obtain latest conf.py
    run(f"git checkout {version}", shell=True, check=True)
    if os.path.isfile("docs/source/conf.py"):
        # removing the old conf.py file to make room for the latest one
        os.remove("docs/source/conf.py")
    copy("conf_latest.py", "docs/source/conf.py")

    # run librep on markdown files (render placeholders with sytax §§§filename§§§)
    run(
        "librep --ref_dir $PWD --input 'docs/**/*.md' --no_backup",  # noqa: S607
        shell=True,
        check=True,
    )

    # build the docs for this version
    run(
        f" python -m sphinx.cmd.build -b html docs/source docs/_build/html/{version}",
        shell=True,
        check=True,
    )
    run("git stash", shell=True, check=True)  # noqa: S607


def main():
    """Main function that is executed when the script is called."""
    # get currently active branch
    command = "git rev-parse --abbrev-ref HEAD".split()
    initial_branch = run(command, capture_output=True, check=True).stdout.strip().decode("utf-8")

    # copy the latest conf.py, since we want to use that configuration for all the
    # docs versions that are built
    copy("docs/source/conf.py", "./conf_latest.py")

    # build docs for main branch
    build_docs_version("main")

    # checkout initial branch for following steps
    run(f"git checkout {initial_branch}", shell=True, check=True)

    # remove temporary copy of latest conf.py
    os.remove("./conf_latest.py")


if __name__ == "__main__":
    main()
