from __future__ import annotations

import numpy as np
import scipy as scp


def confusion_matrix(
    targets: np.ndarray,
    predictions: np.ndarray,
    sample_weights: np.ndarray | None = None,
    normalize: str | None = "rownorm",
) -> np.ndarray:
    """
    Parameters
    ----------
    targets : np.ndarray
        target labels
    predictions : np.ndarray
        predicted labels (output of the classifier)
    sample_weights : np.ndarray, optional
        Weight of each sample; if None, each sample weights the same. Defaults to None.
    normalize : str | None, optional
        Normalization of the confusion matrix. Can be:
        None : Give raw counts;
        "rownorm": Normalize across the prediction class, i.e. such that the rows add to one;
        "colnorm": Normalize across the target class, i.e. such that the columns add to one;
        "all" : Normalize across all examples, i.e. such that all matrix entries add to one.
        Defaults to "rownorm".

    Returns
    -------
    np.ndarray : the confusion matrix,
    np.dnarray : the per-class efficiencies,
    np.ndarray : the per-class fake rates.

    Example
    --------
    >>> targets = np.array([2, 0, 2, 2, 0, 1])
    >>> predictions = np.array([0, 0, 2, 2, 0, 2])
    >>> weights = np.array([1, 0.5, 0.5, 1, 0.2, 1])
    >>> cm, eff, fake = confusion_matrix(targets, predictions, sample_weights=weights)
    >>> print(cm)
    np.array([[1.  0.  0. ]
        [0.  0.  1. ]
        [0.4 0.  0.6]])
    """
    # Checking that targets and predictions have the same sample size
    assert (
        targets.shape[0] == predictions.shape[0]
    ), "confusion_matrix: Predictions and targets must have the same sample size"
    # If user gives samples' weights, check that the sample size is consistent with the labels
    if sample_weights is not None:
        assert (
            sample_weights.shape[0] == targets.shape[0]
        ), "confusion_matrix: Mismatch between targets' and sample weights' size"

    if normalize is not None:
        assert normalize in {
            "rownorm",
            "colnorm",
            "all",
        }, "confusion_matrix: invalid normalization keyword"

    # Finding number of target classes
    # (i.e. max value of the categorical indexing plus one,
    # since categorical index starts from zero)
    n_classes = int(np.max(targets)) + 1

    # If no samples' weights are given, give to each sample weight = 1
    if sample_weights is None:
        sample_weights = np.ones_like(targets)

    # Calculate the raw count Confusion Matrix
    cm = scp.sparse.coo_matrix(
        (sample_weights, (targets, predictions)),
        shape=(n_classes, n_classes),
        dtype="float",
    ).toarray()

    # Calculate efficiency and fake rate
    efficiency = []
    fake_rate = []
    for i, row in enumerate(cm):
        N_this_class = np.sum(row)
        Npred_this_class = row[i]
        Npred_other_class = N_this_class - Npred_this_class
        efficiency.append(Npred_this_class / N_this_class)
        fake_rate.append(Npred_other_class / N_this_class)

    efficiency = np.array(efficiency)
    fake_rate = np.array(fake_rate)

    # Eventually normalize the Confusion Matrix
    with np.errstate(all="warn"):
        if normalize == "all":
            cm /= cm.sum()
        elif normalize == "rownorm":
            cm /= cm.sum(axis=1, keepdims=True)
        elif normalize == "colnorm":
            cm /= cm.sum(axis=0, keepdims=True)

    # Returning the CM with nan converted to zero
    return np.nan_to_num(cm), efficiency, fake_rate
