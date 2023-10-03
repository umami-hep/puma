"""Support functions for vertexing performance in flavour tagging."""
from __future__ import annotations

import numpy as np


def build_vertices(vertex_ids):
    """
    Vertex builder that outputs a list of vertices consisting of a
    list of track indices from vertex ids derived in athena or salt.

    Parameters
    ----------
    vertex_ids: np.ndarray
        Array containing vertex IDs for each track.

    Returns
    -------
    vertices: list (np.ndarray)
        List of vertices, each containing an array of track indices.
    """
    unique_ids, unique_counts = np.unique(vertex_ids, return_counts=True)
    unique_ids = unique_ids[unique_counts > 1] #remove vertices with only one track

    vertices = []
    for id in unique_ids:
        vertices.append(np.nonzero(vertex_ids == id)[0])

    return vertices


def associate_vertices(vertices1, vertices2):
    """
    Vertex associator that maps two collections of vertices onto
    each other 1-to-1 based on the highest overlap of track indices.
    Percentage of overlapping and non-overlapping tracks relative to
    vertex size are used as tiebreakers.

    Parameters
    ----------
    vertices1: list (np.ndarray)
        Vertex collection containing a list of vertices with an array of track indices each.
    vertices2: list (np.ndarray)
        Vertex collection containing a list of vertices with an array of track indices each.

    Returns
    -------
    associations: np.ndarray
        Matrix of vertex associations with shape (len(vertices1), len(vertices2)).
    """