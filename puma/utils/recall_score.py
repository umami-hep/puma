from __future__ import annotations

import numpy as np

from puma.utils.confusion_matrix import confusion_matrix


def recall_score_per_class(
    targets: np.ndarray,
    predictions: np.ndarray,
    sample_weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    The recall score is defined, for each class, as ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` is the number of false negatives.

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
    np.ndarray : the per-class recall score.

    Example
    --------
    >>> targets = np.array([2, 0, 2, 2, 0, 1])
    >>> predictions = np.array([0, 0, 2, 2, 0, 2])
    >>> weights = np.array([1, 0.5, 0.5, 1, 0.2, 1])
    >>> recall_score_per_class(targets, predictions, sample_weights=weights)
    [1.0, 0.0, 0.6]
    """
    cm = confusion_matrix(targets, predictions, sample_weights=sample_weights, normalize=None)

    tp = np.diag(cm)
    with np.errstate(all="warn"):
        recall = tp / np.sum(cm, axis=1)

    return np.nan_to_num(recall)
