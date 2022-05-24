"""Script to loop over a number of git branches/tags, check them out and build the
the sphinx docs"""

from subprocess import run

versions = [
    # "main",
    "v0.1.0",
]

proc = run("git rev-parse --abbrev-ref HEAD", shell=True, check=True)
initial_branch = proc.stdout

for version in versions:
    # build the documentation
    print(f"Building docs for branch/tag {version}")
    command = (
        f"git checkout {version} && sphinx-build -b html source _build/html/{version}"
    )
    run(command, shell=True, check=True)

run(f"git checkout {initial_branch}", shell=True, check=True)
