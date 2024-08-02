from __future__ import annotations

import numpy as np

from puma.utils.confusion_matrix import confusion_matrix


def precision_score_per_class(
    targets: np.ndarray,
    predictions: np.ndarray,
    sample_weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    The precision score is defined, for each class, as ``tp / (tp + fp)``,
    where ``tp`` is the number of true positives and ``fp`` is the number of false positives.

    Parameters
    ----------
    targets : 1d np.ndarray
        target labels
    predictions : 1d np.ndarray
        predicted labels (output of the classifier)
    sample_weights : np.ndarray, optional
        Weight of each sample; if None, each sample weights the same. Defaults to None.

    Returns
    -------
    np.ndarray : the per-class precision score.

    Example
    --------
    >>> targets = np.array([2, 0, 2, 2, 0, 1])
    >>> predictions = np.array([0, 0, 2, 2, 0, 2])
    >>> weights = np.array([1, 0.5, 0.5, 1, 0.2, 1])
    >>> precision_score_per_class(targets, predictions, sample_weights=weights)
    [0.41176471, 1.0, 1.0]
    """
    cm = confusion_matrix(targets, predictions, sample_weights=sample_weights, normalize=None)
    tp = np.diag(cm)
    with np.errstate(all="warn"):
        precision = tp / np.sum(cm, axis=0)

    return np.nan_to_num(precision)
