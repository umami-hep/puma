# Docker images

The Docker images are built on GitHub and contain the latest version from the `main` branch.

The container registry with all available tags can be found
[here](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-images/puma-images/container_registry/13727).

## Extended image for development

_For development, you should use the `latest-dev` tagged image_:

In addition to the minimal requirements that are required to use `puma`, the
`puma:latest-dev` image has the `requirements.txt` from the `puma` repo installed as
well.
This means that packages like `pytest`, `black`, `pylint`, etc. are installed as well.
However, note that `puma` itself is not installed in that image such that the dev-version
on your machine can be used/tested.

On a machine with Docker installed:

```bash
docker run -it --rm -v $PWD:/puma_container -w /puma_container gitlab-registry.cern.ch/atlas-flavor-tagging-tools/training-images/puma-images/puma:latest-dev bash
```

On a machine/cluster with singularity installed:

```bash
singularity shell -B $PWD docker://gitlab-registry.cern.ch/atlas-flavor-tagging-tools/training-images/puma-images/puma:latest-dev
```
