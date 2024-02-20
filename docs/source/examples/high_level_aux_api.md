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
properly supported within puma.

```py
§§§examples/high_level_aux_plots.py:1:35§§§
```


## Vertexing performance

Vertexing performance plots can be produced for a specified jet flavour (all jets by default) as shown in
```py
§§§examples/high_level_aux_plots.py:39:42§§§
```
This function handles all considerations for vertexing performance, including removing PV, pileup and fake
tracks from truth vertices and from reco vertices if a tagger has available track origin classification
information. See [here](https://ftag.docs.cern.ch/algorithms/labelling/track_labels/) for more information
about truth track origin and vertex definitions. It also performs a 1:1 greedy matching procedure between
truth and reconstructed vertices, with which it calculates all relevant performance metrics for plots. There
is also a flag to turn on inclusive vertexing, which merges all tracks from HF into a single truth vertex
per jet. For reconstructed vertices, this merges all tracks from HF if track origin classification
information is available and all tracks marked as being contained in a secondary vertex if not. In total, 5
plots are produced (plotted against a specified performance variable):

* Vertexing efficiency: defined as number of vertices matched divided by number of true vertices
* Vertexing purity: defined as number of vertices matched divided by number of reconstructed vertices
* Number of reconstructed vertices: self-explanatory, useful to examine e.g. vertexing performance in l-jets
* Track-vertex association efficiency: defined as number of tracks in matched vertex common between truth and
reco vertices divided by number of tracks in true vertex
* Track-vertex association purity: defined as number of tracks in matched vertex common between truth and reco
vertices divided by number of tracks in reco vertex

Note that by default the vertex matching algorithm enforces purity criteria requiring track association
efficiency > 0.65 and purity > 0.5
