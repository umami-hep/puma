# High level API for aux task plots

To set up the inputs for the plots, have a look [here](./index.md). In general, the input structure is the same
for aux task plots, but the "tracks" container is also required since this is where the aux task outputs are
located. In particular, the truth and reco vertex indices (by default `truthVertexIndex` and `VertexIndex`)
are necessary for vertexing plots.

The following examples use the dummy data which is described [here](./dummy_data.md)

The high level API for aux tasks matches the general high level API in terms of structure. It can be used to
produce vertexing performance plots.


## Initialising the taggers

The `AuxResults` object is initialised with the signal class in the same way as the `Results` class. Any vertexing
performance plots produced will reflect performance on this signal class only. Vertex matching is automatically done
within the `AuxResults` class once a tagger is added and all relevant metrics for plots are computed.

```py
§§§examples/high_level_aux_plots.py:1:37§§§
```


## Vertexing metrics vs a variable

Event level vertexing efficiency and purity plots vs a jet level variable can be produced as shown in
```py
§§§examples/high_level_aux_plots.py:39:42§§§
```
Here, efficiency represents the number of reconstructed vertices across all jets that were matched to a truth
vertex over the total number of truth vertices. The purity represents the number of matched vertices over the
total number of reconstructed vertices across all jets.


## Track to vertex association metrics vs a variable
One can also produce track to vertex association efficiency and purity plots vs a jet level variable
as follows:
```py
§§§examples/high_level_aux_plots.py:44:47§§§
```
These plots represent a way to measure vertex purity. Efficiency and purity are defined in analogy to
the corresponding jet level quantities, but in this case with regards to the correct and incorrect track
associations within each pair of matched vertices.
