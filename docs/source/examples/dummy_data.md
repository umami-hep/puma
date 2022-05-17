# Dummy Data

Often to test the plotting, we just want to use dummy data.

There are two methods in umami to generate dummy data:

The first function returns directly a pandas DataFrame including the following columns:

* `HadronConeExclTruthLabelID`
* `rnnip_pu`
* `rnnip_pc`
* `rnnip_pb`
* `dips_pu`
* `dips_pc`
* `dips_pb`

which can be used in the following manner:
```python
from umami.plotting.utils import get_dummy_2_taggers

df = get_dummy_2_taggers()

```

The second function is `get_dummy_multiclass_scores` which returns an output array with shape `(size, 3)`, which is the usual output of our multi-class classifiers like DIPS, and the labels conform with the `HadronConeExclTruthLabelID` variable.

```python
from umami.plotting.utils import get_dummy_multiclass_scores

output, labels = get_dummy_multiclass_scores()
```


TODO: here we need to write more when we have the API more evolved.