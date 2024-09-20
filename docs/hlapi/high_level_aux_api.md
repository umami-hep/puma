# High level API for aux task plots

To set up the inputs for the plots, have a look [here](./index.md). In general, the input structure is the same
for aux task plots, but a separate container with track-level aux task outputs is required to be present in the
files (by default "tracks"). For now, vertexing and track origin prediction aux tasks are supported.

The following examples use the dummy data which is described [here](./dummy_data.md)

The high level API for aux tasks matches the general high level API in terms of structure. It can be used to
produce vertexing performance plots.


## Initialising the taggers

Compared to the `Results` object, an `AuxResults` object is initialized without definition of a signal class (rather
this is passed to the plotting functions directly). Otherwise initialization proceeds in the same fashion. Taggers
can be added to these objects in an analogous way to `Results`, except that each tagger should be initialized with
a list of available aux tasks (by default "vertexing" and "track_origin"). Relevant information for these aux tasks
will then be read in from the provided file, assuming the information is available and the specific aux task is
properly supported within puma. The `AuxResults` object also allows for track variables (such as pT, eta, etc.) to
be loaded in for each tagger. These are listed out via the `aux_perf_vars` attribute and allow for calculation of
vertex masses. If (track) `deta` is specified in `aux_perf_vars` and (jet) `eta` in `perf_vars`, (track) `eta` is
automatically calculated and added to `aux_perf_vars`.

```py
--8<-- "examples/high_level_aux_plots.py:1:35"
```


## Vertexing performance

Vertexing performance plots can be produced for a specified jet flavour as shown in
```py
--8<-- "examples/high_level_aux_plots.py:41:43"
```
Here ```vtx_flavours``` defines a list of flavours for which secondary vertices are expected (e.g. b-jets)
and ```no_vertex_flavours``` defines a list where secondary vertices are not expected (e.g. l-jets). Different
plots are produced in each case (see below). In general, this plotting function handles all considerations
for vertexing performance. This includes processing truth vertex indices by removing vertices containing tracks
not from HF and reco vertices by removing the vertex most consistent with the reconstructed PV (if a tagger has
the capability to identify tracks from a PV via track origin classification). See
[here](https://ftag.docs.cern.ch/algorithms/labelling/track_labels/) for more information about truth track
origin and vertex definitions. If inclusive vertexing is enabled, all HF vertices are merged into a single
truth vertex. For reconstructed vertices in a tagger with track origin classification, all vertices with at
least one HF track are merged and all others are removed. If track origin classification is not available, but
inclusive vertexing is enabled, then all vertices are merged. After this cleaning procedure, a 1:1 greedy matching
procedure between truth and reconstructed vertices is performed, with which all the relevant performance metrics
for plots are calculated. In total, 4 plots are produced for each jet flavour with expected SVs and 1 is produced
for each flavour with no expected SVs (all plotted against specific performance variable):

* Vertexing efficiency: defined as number of vertices matched divided by number of true vertices (expected SVs)
* Vertexing purity: defined as number of vertices matched divided by number of reconstructed vertices (expected SVs)
* Track-vertex association efficiency: defined as number of tracks in matched vertex common between truth and
reco vertices divided by number of tracks in true vertex (expected SVs)
* Track-vertex association purity: defined as number of tracks in matched vertex common between truth and reco
vertices divided by number of tracks in reco vertex (expected SVs)
* Vertexing fake rate: fraction of jets where at least one SV is found (no expected SVs)

Note that by default the vertex matching algorithm enforces purity criteria requiring track association
efficiency > 0.65 and purity > 0.5


## Vertex mass reconstruction

Secondary vertex mass histograms can be produced for a specified jet flavour as shown in
```py
--8<-- "examples/high_level_aux_plots.py:45:47"
```
Vertex masses are calculated using a pion hypothesis for all tracks, both for truth and reconstructed vertices with
the given values for pT, eta and phi. These plots can be produced for inclusive or exclusive vertexing as defined in the
vertexing section. For inclusive vertexing, difference plots are also produced, which show differences in the reco vs.
truth mass estimates on a vertex by vertex basis.


## Track Origin Performances

The Track Origin auxiliary task is a multiclass classification task, in which each track is associated with its most probable belonging origin, chosen between:

- Pileup;
- Fake;
- Primary;
- FromB;
- FromBC;
- FromC;
- FromTau;
- OtherSecondary;

A good metric to evaluate the performances of the classifier is the [Confusion Matrix](../examples/confusion_matrix.md), which can be plotted using the method `plot_track_origin_confmat`.

The normalization of the tagger's confusion matrix can be chosen among the ones allowed by the `confusion_matrix` [possible normalizations](../examples/confusion_matrix.md#normalization), by specifying the argument `normalize` in the `plot_track_origin_confmat` function. By default, the matrix's rows are normalized.
