"""Support functions for general aux task related things."""

from __future__ import annotations


def get_aux_labels():
    """Get the truth labels for all aux tasks."""
    return {
        "vertexing": "ftagTruthVertexIndex",
        "track_origin": "ftagTruthOriginLabel",
    }
