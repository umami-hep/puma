"""Support functions for vertexing performance in flavour tagging."""
from __future__ import annotations

import numpy as np


def build_vertices(vertex_ids, ignore_indices=[-1]):
    """
    Vertex builder that outputs an array of vertex associations
    from vertex ids derived in athena or salt for a single jet.

    Parameters
    ----------
    vertex_ids: np.ndarray
        Array containing vertex IDs for each track.
    ignore_indices: list, optional
        List of vertex IDs to ignore, by default [-1].

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
    purity: np.ndarray
        Matrix containing vertex purity scores (fraction of tracks in test vertex also found
        in ref vertex) for each pairing.
    true_assoc_rate: np.ndarray
        Matrix containing vertex true association rates (fraction of tracks in ref vertex also
        found in true vertex) for each pairing.
    """
    ref_sizes = ref_vertices.sum(axis=1)
    test_sizes = test_vertices.sum(axis=1)
    n_ref = ref_sizes.size
    n_test = test_sizes.size

    common_tracks = np.dot(test_vertices.astype(int), ref_vertices.astype(int).T)
    purity = common_tracks / np.tile(ref_sizes, (n_test, 1))
    true_assoc_rate = common_tracks / np.tile(test_sizes, (n_ref, 1)).T

    pair_index = np.arange(n_ref*n_test, 0, -1).reshape(
        n_test, n_ref
    ) # unique number for each vertex pairing

    # calculate vertex associations based on maximum number of shared tracks
    # followed by maximum purity and true association rate; if there are two
    # equally good pairings for a vertex, the first in the list is chosen
    associations = np.ones_like(common_tracks, dtype=bool)
    for metric in [common_tracks, purity, true_assoc_rate, pair_index]:
        metric[np.logical_not(associations)] = -1
        col_max = np.tile(np.amax(metric, axis=0), (n_test, 1))
        row_max = np.tile(np.amax(metric, axis=1), (n_ref, 1)).T
        associations = np.logical_and(metric == col_max, metric == row_max)

    return associations, purity, true_assoc_rate


def calculate_vertex_metrics(test_indices, ref_indices, max_vertices=20):
    """
    Vertex metric calculator that returns a list of vertex efficiencies, fake
    rates and purity information for each jet as well as the absolute number of
    tracks associated to truth and reco vertices.

    Parameters
    ----------
    ref_indices: np.ndarray
        Boolean array of shape (n_jets, n_tracks) containing vertex indices to use
        as reference (truth).
    test_indices: np.ndarray
        Boolean array of shape (n_jets, n_tracks) containing vertex indices to be
        tested (reco).
    max_vertices: int, optional
        Maximum number of vertices to write out, by default 20.

    Returns
    -------
    vertex_efficiency: np.ndarray
        Array of shape (n_jets) containing the vertex efficiency for each jet.
    vertex_fake_rate: np.ndarray
        Array of shape (n_jets) containing the vertex fake rate for each jet.
    purity: np.ndarray
        Array of shape (n_jets, max_vertices) containing the purity score for each
        matched vertex.
    true_assoc_rate: np.ndarray
        Array of shape (n_jets, max_vertices) containing the true association rate
        for each matched vertex.
    test_size: np.ndarray
        Array of shape (n_jets, max_vertices) containing the number of tracks in each
        test vertex.
    ref_size: np.ndarray
        Array of shape (n_jets, max_vertices) containing the number of tracks in each
        reference vertex.
    """

    assert ref_indices.shape == test_indices.shape, "Truth and reco vertex arrays must have the same shape."
    n_jets = ref_indices.shape[0]

    vertex_efficiency = np.zeros(n_jets)
    vertex_fake_rate = np.zeros(n_jets)
    purity = np.full((n_jets, max_vertices), -1, dtype=float)
    true_assoc_rate = np.full((n_jets, max_vertices), -1, dtype=float)
    test_size = np.full((n_jets, max_vertices), -1)
    ref_size = np.full((n_jets, max_vertices), -1)

    for i in range(n_jets):
        ref_vertices = build_vertices(ref_indices[i])
        test_vertices = build_vertices(test_indices[i])

        associations, purity_jet, true_assoc_rate_jet = associate_vertices(
            test_vertices, ref_vertices
        )

        n_match = np.sum(associations)

        # write out vertex metrics
        vertex_efficiency[i] = n_match / associations.shape[1]
        vertex_fake_rate[i] = associations.shape[0] / n_match - 1
        purity[i, : n_match] = purity_jet[associations]
        true_assoc_rate[i, : n_match] = true_assoc_rate_jet[associations]
        test_size[i, : n_match] = test_vertices[associations.sum(axis=1).astype(bool)].sum(axis=1)
        ref_size[i, : n_match] = ref_vertices[associations.sum(axis=0).astype(bool)].sum(axis=1)

    return vertex_efficiency, vertex_fake_rate, purity, true_assoc_rate, test_size, ref_size
