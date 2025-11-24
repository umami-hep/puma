"""Support functions for general aux task related things."""

from __future__ import annotations


def get_aux_labels() -> dict:
    """Get the truth labels for all aux tasks.

    Returns
    -------
    dict
        Dict with the truth labels for the aux tasks
    """
    return {
        "vertexing": "ftagTruthVertexIndex",
        "track_origin": "ftagTruthOriginLabel",
    }


def get_trackOrigin_classNames() -> list[str]:
    """Get the Track Origin class names.

    Returns
    -------
    list[str]
        the class names
    """
    return [
        "Pileup",
        "Fake",
        "Primary",
        "FromB",
        "FromBC",
        "FromC",
        "FromTau",
        "OtherSecondary",
    ]
