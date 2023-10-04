"""Support functions for vertexing performance in flavour tagging."""
from __future__ import annotations

import numpy as np


def build_vertices(vertex_ids, minimum_id = None):
    """
    Vertex builder that outputs an array of vertex associations
    from vertex ids derived in athena or salt.

    Parameters
    ----------
    vertex_ids: np.ndarray
        Array containing vertex IDs for each track.
    minimum_id: int, optional
        Minimum vertex ID to consider, by default None.

    Returns
    -------
    vertices: np.ndarray
        Boolean array of shape(n_vertices, n_tracks) containing track-vertex
        associations for each vertex. Each track is associated with at most
        one vertex.
    """
    unique_ids, unique_counts = np.unique(vertex_ids, return_counts=True)
    unique_ids = unique_ids[unique_counts > 1] # remove vertices with only one track
    if minimum_id is not None: unique_ids = unique_ids[unique_ids >= minimum_id] # don't consider vertices with id < minimum_id

    vertices = np.tile(vertex_ids, (unique_ids.size, 1))
    comparison_ids = np.tile(unique_ids, (vertex_ids.size, 1)).T
    vertices = (vertices == comparison_ids)
    
    return vertices


def associate_vertices(test_vertices, ref_vertices):
    """
    Vertex associator that maps two collections of vertices onto
    each other 1-to-1 based on the highest overlap of track indices.
    Percentage of overlapping and non-overlapping tracks relative to
    vertex size are used as tiebreakers.

    Parameters
    ----------
    ref_vertices: np.ndarray
        Boolean array of shape(n_ref_vertices, n_tracks) containing track-vertex associations
        for vertex collection to use as reference (truth).
    test_vertices: np.ndarray
        Boolean array of shape(n_test_vertices, n_tracks) containing track-vertex associations
        for vertex collection to be tested (reco).

    Returns
    -------
    associations: np.ndarray
        Matrix of vertex associations with shape (len(ref_vertices), len(test_vertices)).
    """
    ref_sizes = ref_vertices.sum(axis=1)
    test_sizes = test_vertices.sum(axis=1)
    n_ref = ref_sizes.size
    n_test = test_sizes.size

    common_tracks = np.dot(test_vertices.astype(int), ref_vertices.astype(int).T)
    purity = common_tracks / np.tile(test_sizes, (n_ref, 1)).T
    fake_assoc_rate = common_tracks / np.tile(ref_sizes, (n_test, 1)) # actually 1 - fake association rate
    pair_index = np.arange(n_ref*n_test, 0, -1).reshape(n_test, n_ref) # unique number for each vertex pairing

    # calculate vertex associations based on maximum number of shared tracks
    # followed by maximum purity and minimum fake association rate; if there are two
    # equally good pairings for a vertex, the first in the list is chosen
    associations = np.ones_like(common_tracks, dtype=bool)
    for metric in [common_tracks, purity, fake_assoc_rate, pair_index]:
        metric[np.logical_not(associations)] = -1 # disregard pairs that have already been thrown out
        col_max = np.tile(np.amax(metric, axis=0), (n_test, 1))
        row_max = np.tile(np.amax(metric, axis=1), (n_ref, 1)).T
        associations = np.logical_and(metric == col_max, metric == row_max)

    return associations, purity, fake_assoc_rate