"""Support functions for vertexing performance in flavour tagging."""
from __future__ import annotations

import numpy as np


def build_vertices(vertex_ids, ignore_indices=None):
    """
    Vertex builder that outputs an array of vertex associations
    from vertex ids derived in athena or salt for a single jet.

    Parameters
    ----------
    vertex_ids: np.ndarray
        Array containing vertex IDs for each track.

    Returns
    -------
    vertices: np.ndarray
        Boolean array of shape (n_vertices, n_tracks) containing track-vertex
        associations for each vertex. Each track is associated with at most
        one vertex.
    """
    unique_ids, unique_counts = np.unique(vertex_ids, return_counts=True)
    unique_ids = unique_ids[unique_counts > 1]  # remove vertices with only one track
    unique_ids = unique_ids[
        np.logical_not(np.isin(unique_ids, ignore_indices))
    ]  # remove vertices with ignored indices

    vertices = np.tile(vertex_ids, (unique_ids.size, 1))
    comparison_ids = np.tile(unique_ids, (vertex_ids.size, 1)).T
    vertices = vertices == comparison_ids

    return vertices


def associate_vertices(test_vertices, ref_vertices):
    """
    Vertex associator that maps two collections of vertices onto
    each other 1-to-1 based on the highest overlap of track indices
    for a single jet. Percentage of overlapping and non-overlapping
    tracks relative to vertex size are used as tiebreakers.

    Parameters
    ----------
    ref_vertices: np.ndarray
        Boolean array of shape (n_ref_vertices, n_tracks) containing track-vertex associations
        for vertex collection to use as reference (truth).
    test_vertices: np.ndarray
        Boolean array of shape (n_test_vertices, n_tracks) containing track-vertex associations
        for vertex collection to be tested (reco).

    Returns
    -------
    associations: np.ndarray
        Boolean matrix of vertex associations with shape (n_test_vertices, n_ref_vertices).
    common_tracks: np.ndarray
        Matrix containing number of common tracks shared by each vertex pairing.
    """
    ref_sizes = ref_vertices.sum(axis=1)
    test_sizes = test_vertices.sum(axis=1)
    n_ref = ref_sizes.size
    n_test = test_sizes.size

    common_tracks = np.dot(test_vertices.astype(int), ref_vertices.astype(int).T)
    inv_ref_size = 1.0 / np.tile(ref_sizes, (n_test, 1))
    inv_test_size = 1.0 / np.tile(test_sizes, (n_ref, 1)).T

    pair_index = np.arange(n_ref * n_test, 0, -1).reshape(
        n_test, n_ref
    )  # unique number for each vertex pairing

    # calculate vertex associations based on maximum number of shared tracks
    # with ties broken by preferring the smallest true vertex (maximum purity score)
    # followed by the smallest reco vertex (minimum false association); if there are
    # two equally good pairings for a vertex, the first in the list is chosen
    associations = np.ones_like(common_tracks, dtype=bool)
    for metric in [common_tracks, inv_ref_size, inv_test_size, pair_index]:
        metric[np.logical_not(associations)] = -1
        col_max = np.tile(np.amax(metric, axis=0), (n_test, 1))
        row_max = np.tile(np.amax(metric, axis=1), (n_ref, 1)).T
        associations = np.logical_and(metric == col_max, metric == row_max)

    return associations, common_tracks


def calculate_vertex_metrics(
    test_indices, ref_indices, max_vertices=20, ignore_indices=None
):
    """
    Vertex metric calculator that outputs a set of metrics useful for evaluating
    vertexing performance for each jet.

    Parameters
    ----------
    ref_indices: np.ndarray
        Boolean array of shape (n_jets, n_tracks) containing vertex indices to use
        as reference (truth).
    test_indices: np.ndarray
        Boolean array of shape (n_jets, n_tracks) containing vertex indices to be
        tested (reco).
    max_vertices: int, optional
        Maximum number of matched vertices to write out, by default 20.
    ignore_indices: list, optional
        List of vertex IDs to ignore, by default None.

    Returns
    -------
    n_match: np.ndarray
        Array of shape (n_jets) containing the number of matched vertices per jet.
    n_test: np.ndarray
        Array of shape (n_jets) containing the number of reco vetices per jet.
    n_ref: np.ndarray
        Array of shape (n_jets) containing the number of truth vertices per jet.
    track_overlap: np.ndarray
        Array of shape (n_jets, max_vertices) containing the number of overlapping
        tracks between each matched vertex pair.
    test_vertex_size: np.ndarray
        Array of shape (n_jets, max_vertices) containing the number of tracks in each
        matched reco vertex.
    ref_vertex_size: np.ndarray
        Array of shape (n_jets, max_vertices) containing the number of tracks in each
        matched truth vertex.
    """

    assert (
        ref_indices.shape == test_indices.shape
    ), "Truth and reco vertex arrays must have the same shape."
    n_jets = ref_indices.shape[0]

    n_match = np.zeros(n_jets, dtype=int)
    n_test = np.zeros(n_jets, dtype=int)
    n_ref = np.zeros(n_jets, dtype=int)
    track_overlap = np.full((n_jets, max_vertices), -1)
    test_vertex_size = np.full((n_jets, max_vertices), -1)
    ref_vertex_size = np.full((n_jets, max_vertices), -1)

    for i in range(n_jets):
        ref_vertices = build_vertices(ref_indices[i], ignore_indices=ignore_indices)
        test_vertices = build_vertices(test_indices[i], ignore_indices=ignore_indices)

        # handle edge cases
        if not ref_vertices.any() and not test_vertices.any():
            continue
        elif not ref_vertices.any():
            n_test[i] = test_vertices.shape[0]
            continue
        elif not test_vertices.any():
            n_ref[i] = ref_vertices.shape[0]
            continue

        associations, common_tracks = associate_vertices(test_vertices, ref_vertices)

        # write out vertexing efficiency metrics
        n_match[i] = np.sum(associations)
        n_ref[i] = ref_vertices.shape[0]
        n_test[i] = test_vertices.shape[0]

        # write out vertexing purity metrics
        track_overlap[i, : n_match[i]] = common_tracks[associations]
        test_vertex_size[i, : n_match[i]] = test_vertices[
            associations.sum(axis=1).astype(bool)
        ].sum(axis=1)
        ref_vertex_size[i, : n_match[i]] = ref_vertices[
            associations.sum(axis=0).astype(bool)
        ].sum(axis=1)

    return n_match, n_test, n_ref, track_overlap, test_vertex_size, ref_vertex_size
