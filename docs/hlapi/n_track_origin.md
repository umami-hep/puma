# Number of Tracks per Track Origin

To understand the amount of tracks per jet as a function of $p_\mathrm{T}$, PUMA
provides a high-level function named `n_tracks_per_origin`. This function can load an
`.h5` from the
[Training-Dataset-Dumper](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper/)
and extract the number of tracks per jet (and per track origin) as a function of $p_\mathrm{T}$.

An example of the usage of this function can be found here:

```py
--8<-- "examples/plot_n_track_origin.py"
```