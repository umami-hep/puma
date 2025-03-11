"""Support functions for vertexing performance in flavour tagging."""

from __future__ import annotations

import numpy as np


def clean_indices(vertex_ids, condition, mode="remove"):
    """
    Vertex index cleaner that modifies vertex indices that fulfill
    a specified condition. The mode can be set to either "remove"
    or "merge". In the "remove" mode, the vertex indices are set to
    be ignored by the matching algorithm while in the "merge" mode
    the vertex indices are merged into a single vertex.

    Parameters
    ----------
    vertex_ids: np.ndarray
        Array containing vertex IDs for each track.
    condition: np.ndarray
        Boolean array of shape (n_jets, n_tracks) containing the condition
        to be applied.
    mode: str
        Mode to apply to indices that meet the specified condition.
        Options are "remove" and "merge".

    Returns
    -------
    vertex_ids: np.ndarray
        Array containing vertex IDs for each track.

    Raises
    ------
    ValueError
        If the mode is not recognized.
    """
    if mode == "remove":
        vertex_ids[condition] = -99
    elif mode == "merge":
        if len(set(vertex_ids[condition])) > 1:
            vertex_ids[condition] = np.max(vertex_ids) + 1
    else:
        raise ValueError(f"Mode {mode} not recognized.")

    return vertex_ids


def build_vertices(vertex_ids):
    """
    Vertex builder that outputs an array of vertex associations
    from vertex ids derived in athena or salt for a single jet.
    Negative indices and one-track vertices are ignored by
    default.

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
    unique_ids = unique_ids[unique_ids >= 0]  # remove tracks with negative indices

    vertices = np.tile(vertex_ids, (unique_ids.size, 1))
    comparison_ids = np.tile(unique_ids, (vertex_ids.size, 1)).T
    return vertices == comparison_ids


def associate_vertices(test_vertices, ref_vertices, eff_req, purity_req):
    """
    Vertex associator that maps two collections of vertices onto
    each other 1-to-1 based on the highest overlap of track indices
    for a single jet. Percentage of overlapping and non-overlapping
    tracks relative to vertex size are used as tiebreakers. The
    matching is performed in a greedy fashion and decisions are not
    revisited.

    Parameters
    ----------
    test_vertices: np.ndarray
        Boolean array of shape (n_test_vertices, n_tracks) containing track-vertex
        associations for vertex collection to be tested (reco).
    ref_vertices: np.ndarray
        Boolean array of shape (n_ref_vertices, n_tracks) containing track-vertex
        associations for vertex collection to use as reference (truth).
    eff_req: float, optional
        Minimum required efficiency for vertex matching.
    purity_req: float, optional
        Minimum required purity for vertex matching.

    Returns
    -------
    associations: np.ndarray
        Boolean matrix of vertex associations with shape (n_test_vertices,
        n_ref_vertices).
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
        associations[metric == -1] = False  # remove pairs that were already excluded
    associations[common_tracks == 0] = False  # remove leftover pairs with zero matches

    # enforce purity and efficiency requirements
    eff_cut = common_tracks * inv_ref_size >= eff_req
    purity_cut = common_tracks * inv_test_size >= purity_req
    associations = np.logical_and.reduce((associations, eff_cut, purity_cut))

    return associations, common_tracks


def calculate_vertex_metrics(
    test_indices,
    ref_indices,
    max_vertices=20,
    eff_req=0.65,
    purity_req=0.5,
):
    """
    Vertex metric calculator that outputs a set of metrics useful for evaluating
    vertexing performance for each jet.

    Parameters
    ----------
    test_indices: np.ndarray
        Boolean array of shape (n_jets, n_tracks) containing vertex indices to be
        tested (reco).
    ref_indices: np.ndarray
        Boolean array of shape (n_jets, n_tracks) containing vertex indices to use
        as reference (truth).
    max_vertices: int, optional
        Maximum number of matched vertices to write out, by default 20.
    eff_req: float, optional
        Minimum required efficiency for vertex matching, by default 0.65.
    purity_req: float, optional
        Minimum required purity for vertex matching, by default 0.5.

    Returns
    -------
    metrics: dict
        Dictionary containing the following metrics:
            n_match: np.ndarray
                Array of shape (n_jets) containing the number of matched vertices per
                jet.
            n_test: np.ndarray
                Array of shape (n_jets) containing the number of reco vetices per jet.
            n_ref: np.ndarray
                Array of shape (n_jets) containing the number of truth vertices per jet.
            track_overlap: np.ndarray
                Array of shape (n_jets, max_vertices) containing the number of
                overlapping tracks between each matched vertex pair.
            test_vertex_size: np.ndarray
                Array of shape (n_jets, max_vertices) containing the number of tracks
                in each matched reco vertex.
            ref_vertex_size: np.ndarray
                Array of shape (n_jets, max_vertices) containing the number of tracks
                in each matched truth vertex.
    """
    assert (
        ref_indices.shape == test_indices.shape
    ), "Truth and reco vertex arrays must have the same shape."
    n_jets = ref_indices.shape[0]

    metrics = {}
    metrics["n_match"] = np.zeros(n_jets, dtype=int)
    metrics["n_test"] = np.zeros(n_jets, dtype=int)
    metrics["n_ref"] = np.zeros(n_jets, dtype=int)
    metrics["track_overlap"] = np.full((n_jets, max_vertices), -1)
    metrics["test_vertex_size"] = np.full((n_jets, max_vertices), -1)
    metrics["ref_vertex_size"] = np.full((n_jets, max_vertices), -1)

    for i in range(n_jets):
        ref_vertices = build_vertices(ref_indices[i])
        test_vertices = build_vertices(test_indices[i])

        # handle edge cases
        if not ref_vertices.any() and not test_vertices.any():
            continue
        if not ref_vertices.any():
            metrics["n_test"][i] = test_vertices.shape[0]
            continue
        if not test_vertices.any():
            metrics["n_ref"][i] = ref_vertices.shape[0]
            continue

        associations, common_tracks = associate_vertices(
            test_vertices,
            ref_vertices,
            eff_req=eff_req,
            purity_req=purity_req,
        )

        # write out vertexing efficiency metrics
        metrics["n_match"][i] = np.sum(associations)
        metrics["n_ref"][i] = ref_vertices.shape[0]
        metrics["n_test"][i] = test_vertices.shape[0]

        # only save purity metrics for requested number of vertices
        max_index = min(metrics["n_match"][i], max_vertices)

        # write out vertexing purity metrics
        metrics["track_overlap"][i, :max_index] = common_tracks[associations][:max_index]
        metrics["test_vertex_size"][i, :max_index] = test_vertices[
            associations.sum(axis=1).astype(bool)
        ].sum(axis=1)[:max_index]
        metrics["ref_vertex_size"][i, :max_index] = ref_vertices[
            associations.sum(axis=0).astype(bool)
        ].sum(axis=1)[:max_index]

    return metrics


def clean_truth_vertices(truth_vertices, truth_track_origin, incl_vertexing=False):
    """
    Clean truth vertices for each track in a single jet. This function removes
    all truth vertices that are not entirely from HF. If inclusive vertexing
    is enabled, it also merges remaining vertices into a single vertex.

    Parameters
    ----------
    truth_vertices: np.ndarray
        Array containing truth vertex indices for each track in a jet.
    truth_track_origin: np.ndarray
        Array containing truth track origin labels for each track in a jet.
    incl_vertexing: bool, optional
        Whether to merge all vertex indices, by default False.

    Returns
    -------
    truth_vertices: np.ndarray
        Array containing cleaned truth vertex indices for each track in a jet.
    """
    # remove vertices that aren't purely HF
    removal_indices = np.unique(truth_vertices[np.isin(truth_track_origin, [3, 4, 5], invert=True)])
    truth_vertices = clean_indices(
        truth_vertices,
        np.isin(truth_vertices, removal_indices),
        mode="remove",
    )

    # merge truth vertices from HF for inclusive performance
    if incl_vertexing:
        truth_vertices = clean_indices(
            truth_vertices,
            truth_vertices > 0,
            mode="merge",
        )

    return truth_vertices


def clean_reco_vertices(
    reco_vertices, reco_track_origin=None, incl_vertexing=False, require_hf_track=True
):
    """
    Clean reconstructed vertices for each track in a single jet. This function
    removes the vertex with the most reco PV tracks if track origin classification
    information is available. If inclusive vertexing is enabled, it also merges all
    vertices with > 0 tracks from HF and removes all others for taggers with track
    origin classification while merging all vertices for taggers without it.

    Parameters
    ----------
    reco_vertices: np.ndarray
        Array containing reco vertex indices for each track in a jet.
    reco_track_origin: np.ndarray, optional
        Array containing reco track origin labels for each track in a jet.
    incl_vertexing: bool, optional
        Whether to merge all vertex indices, by default False.
    require_hf_track: bool, optional
        Whether to require at least one track from HF to keep a vertex, by default True.

    Returns
    -------
    reco_vertices: np.ndarray
        Array containing cleaned reco vertex indices for each track in a jet.
    """
    # elaborate cleaning if track origin predictions are available
    if reco_track_origin is not None:
        # remove vertex with most reco PV tracks
        pv_candidate_indices, pv_candidate_counts = np.unique(
            reco_vertices[reco_track_origin == 2], return_counts=True
        )
        if pv_candidate_indices.size > 0:
            pv_index = pv_candidate_indices[np.argmax(pv_candidate_counts)]
            reco_vertices = clean_indices(
                reco_vertices,
                reco_vertices == pv_index,
                mode="remove",
            )

        # remove vertices with no tracks from HF
        if require_hf_track:
            hf_vertex_indices = np.unique(reco_vertices[np.isin(reco_track_origin, [3, 4, 5])])
            reco_vertices = clean_indices(
                reco_vertices,
                np.isin(reco_vertices, hf_vertex_indices, invert=True),
                mode="remove",
            )

        # merge remaining vertices for inclusive performance
        if incl_vertexing:
            reco_vertices = clean_indices(
                reco_vertices,
                reco_vertices >= 0,
                mode="merge",
            )

    # merge all reco vertices if track origin predictions are not available
    elif incl_vertexing:
        reco_vertices = clean_indices(
            reco_vertices,
            reco_vertices >= 0,
            mode="merge",
        )

    return reco_vertices
