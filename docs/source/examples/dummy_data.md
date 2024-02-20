# Dummy Data

To test/demonstrate the `puma` API, we just want to use dummy data.

There are three methods in puma to generate dummy data:

The first function returns directly a `pandas.DataFrame` including the following columns:

- `HadronConeExclTruthLabelID`
- `rnnip_pu`
- `rnnip_pc`
- `rnnip_pb`
- `dips_pu`
- `dips_pc`
- `dips_pb`

which can be used in the following manner:

```python
from puma.utils import get_dummy_2_taggers

df = get_dummy_2_taggers()

```

The second function is `get_dummy_multiclass_scores` which returns an output array
with shape `(size, 3)`, which is the usual output of our multi-class classifiers like
DIPS, and the labels conform with the `HadronConeExclTruthLabelID` variable.

```python
from puma.utils import get_dummy_multiclass_scores

output, labels = get_dummy_multiclass_scores()
```

Finally, the `get_dummy_tagger_aux` function returns a h5 file with both jet and
track collections (needed for aux task plots). These include the following columns
(aux task information is generated for both vertexing and track origin classification):

jets:
- `HadronConeExclTruthLabelID`
- `GN2_pu`
- `GN2_pc`
- `GN2_pb`
- `pt`
- `eta`
- `n_truth_promptLepton`

tracks:
- `ftagTruthVertexIndex`
- `GN2_VertexIndex`
- `ftagTruthOriginLabel`
- `GN2_TrackOrigin`

which can be used in the following manner:

```python
from puma.utils import get_dummy_tagger_aux

df = get_dummy_tagger_aux()

```

TODO: here we need to write more when we have the API more evolved.
