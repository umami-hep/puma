# Confusion Matrix

This function evaluates the (multiclass[^1]) Confusion Matrix (CM) associated to a classifier output predictions. The CM is a metric that measures the misclassification rates between all the classes in the classification task.

## Mathematical definition

Mathematically, if the classification task has $N_c$ target classes, the CM is an $N_c \times N_c$ matrix whose entry $C_{i,j}$ is the number of predictions known to be in group $i$ and predicted to be in group $j$. 
The matrix can then be normalized in different ways, obtaining rates of misclassifications instead of raw counts (more on that in [normalization](#normalization)).

## Implementation

The function `confusion_matrix` in `puma.utils.confusion_matrix` computes the CM from two arrays of target and predicted labels. The basic usage is:
```python
targets = np.array([2, 0, 2, 2, 0, 1])
predictions = np.array([0, 0, 2, 2, 0, 2])
confusion_matrix(targets, predictions)
```

Eventually, samples can be weighted by their relative importance by providing an array of weights $w_i \in [0,1]$:
```python
targets = np.array([2, 0, 2, 2, 0, 1])
predictions = np.array([0, 0, 2, 2, 0, 2])
weights = np.array([1, 0.5, 0.5, 1, 0.2, 1])
confusion_matrix(targets, predictions, sample_weights=weights)
```


### Normalization

There are four possible normalization choices, which can be selected through the `normalize` argument of the function: `None` to use raw counts; `"rownorm"` to normalize across the prediction class, i.e. such that the rows add to one (default); `"colnorm"` to normalize across the target class, i.e. such that the columns add to one; `"all" ` to normalize across all examples, i.e. such that all matrix entries add to one. Defaults to `"rownorm"`.

## Example


```py
--8<-- "examples/confusion_matrix_examples.py"
```


[^1] In a multiclass task, each sample belongs to one and only class (the true label, or target label).