"""Support functions for general aux task related things."""

from __future__ import annotations


def get_aux_labels():
    """Get the truth labels for all aux tasks."""
    return {
        "vertexing": "ftagTruthVertexIndex",
        "track_origin": "ftagTruthOriginLabel",
    }


def get_trackOrigin_labels(tagger):
    """Get the Track Origin task target and predicted labels (i.e. an array of the true and predicted
    belonging class index for each track, input/output of the classifier).
    The arrays are flattened, i.e. tracks are not grouped by jets.
    If the sample has `Ntracks` in total, the shape of the returned arrays is `(Ntracks,)`.

    Parameters
    ----------
    tagger : the tagger object.

    Returns
    -------
    np.ndarray, np.ndarray
        `target, predictions`: the target and prediction labels' arrays.
    """
    target = tagger.aux_labels["track_origin"].reshape(-1)
    predictions = tagger.aux_scores["track_origin"].reshape(-1)
    return target, predictions
