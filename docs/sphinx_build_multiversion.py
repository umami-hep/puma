"""Script to loop over a number of git branches/tags, check them out and build the
the sphinx docs"""

from subprocess import run, PIPE
import pygit2

versions = [
    "main",
    # "v0.1.0",
]

initial_branch = pygit2.Repository('.').head.shorthand

for version in versions:
    # build the documentation
    print(f"Building docs for branch/tag {version}")
    command = f"git checkout {version} "  # && sphinx-build -b html source _build/html/{version}"
    run(command, shell=True, check=True)

run(f"git checkout {initial_branch}", shell=True, check=True)
