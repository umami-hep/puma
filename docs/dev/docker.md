# Docker images

The Docker images are built on GitHub and contain the latest version from the `main` branch.

The container registry with all available tags can be found
[here](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-images/puma-images/container_registry/13727).

On a machine with Docker installed:

```bash
docker run -it --rm -v $PWD:/puma_container -w /puma_container gitlab-registry.cern.ch/atlas-flavor-tagging-tools/training-images/puma-images/puma:latest bash
```

On a machine/cluster with singularity installed:

```bash
singularity shell -B $PWD docker://gitlab-registry.cern.ch/atlas-flavor-tagging-tools/training-images/puma-images/puma:latest
```
