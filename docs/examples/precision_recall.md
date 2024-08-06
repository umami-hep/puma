# Precision and Recall

The functions in `puma.utils.precision_score` and in `puma.utils.recall_score` compute the per-class precision and recall classification metrics, for a multiclass classification task with $N_c$ classes. The metrics are computed by comparing the classifier's predicted labels array with the target labels array. The functions return an array where the entry $i$ is the score related to class $i$. The scores are defined as follows.

## Precision

Fixed a class $i$ between the $N_c$ possible classes, the precision score (also called purity) for that class measures the ability of the classifier to not label as class $i$ a sample belonging to another class. It is defined as:

$$p = \frac{tp}{tp+fp}$$

where $tp$ is the true positives count and $fp$ is the false positives count in the test set.

## Recall

Fixed a class $i$ between the $N_c$ possible classes, the recall score for that class measures the ability of the classifier to detect every sample belonging to class $i$ in the set. It is defined as:

$$r = \frac{tp}{tp+fn}$$

where $tp$ is the true positives count and $fn$ is the false negatives count in the test set.

## Example


```py
--8<-- "examples/precision_recall_examples.py"
```