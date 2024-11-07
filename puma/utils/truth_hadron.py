# Hadron processing functions
from __future__ import annotations

import numpy as np


def GetOrderedHadrons(hadron_barcode, hadron_parent, n_max_showers=2):
    # This function orderes the hadron indices inside each jet in different showers
    # INPUTS:  f["truth_hadrons"]["barcode"], f["truth_hadrons"]["ftagTruthParentBarcode"]
    # n_showers max would be n_hadrons if there were 5 unrelated showers (not likely)

    # Output: Padded array of indices with shape (n_jets, n_showers, n_hadrons)

    n_jets, n_hadrons = hadron_barcode.shape

    set_parent_barcodes = [set(row[row > 0]) for row in hadron_parent]

    # Get indices for the parton shower within a family
    child_indices = [
        [
            [index for index, hadron in enumerate(hadron_list) if hadron == parent]
            for parent in parent_set
        ]
        for hadron_list, parent_set in zip(hadron_parent, set_parent_barcodes)
    ]
    parent_index = [
        [
            [index for index, hadron in enumerate(hadron_list) if hadron == parent]
            for parent in parent_set
        ]
        for hadron_list, parent_set in zip(hadron_barcode, set_parent_barcodes)
    ]
    family_indices = [
        [j + c[k] for k, j in enumerate(p)] for p, c in zip(parent_index, child_indices)
    ]

    # Now reshuffle them to keep always first the longest shower!
    #######################################################################################################
    # Calculate lengths for all family_indices
    lengths = np.array([len(family) for family in family_indices], dtype=object)

    # Create a mask for jets with more than one family member
    mask = lengths > 1

    # Create reshuffle indices only for those jets that have more than one family
    reshuffle = np.empty((n_jets, n_max_showers), dtype=object)  # n_max_shower

    reshuffle_indices = [
        np.argsort([len(elem) for elem in family_indices[i]])[::-1]
        for i in range(n_jets)
        if mask[i]
    ]
    reshuffle_indices = [idx[:n_max_showers] for idx in reshuffle_indices]
    if np.array(reshuffle_indices).size > 0:
        reshuffle[mask] = reshuffle_indices

    # Initialize sorted_family_indices
    sorted_family_indices = np.empty(n_jets, dtype=object)

    # Use the reshuffle to sort family_indices
    for i in range(n_jets):
        if mask[i]:  # Check if the jet has more than one family member
            sorted_indices = reshuffle[i]
            sorted_family_indices[i] = [family_indices[i][idx] for idx in sorted_indices]
        else:
            sorted_family_indices[i] = family_indices[i]  # Keep it unchanged if not applicable

    # Convert to numpy array if desired
    sorted_family_indices = np.array(sorted_family_indices, dtype=object)

    #######################################################################################################

    # Deal with the unrelated hadrons

    orphan_barcodes = np.where(
        (hadron_parent < 0) & (hadron_barcode > 0), hadron_barcode, 0
    )  # hadrons without parents
    unrelated_indices = [
        [
            [index]
            for index, hadron in enumerate(hadron_list)
            if (hadron not in parent_set) and (hadron > 0)
        ]
        for hadron_list, parent_set in zip(orphan_barcodes, set_parent_barcodes)
    ]  # remove parents
    extended_family_indices = [
        f + u for f, u in zip(sorted_family_indices.copy(), unrelated_indices)
    ]

    # Now select the more important (first) shower and pad the indices!

    # Initializethe padded array with -1
    padded_hadron_indices = np.full((n_jets, n_max_showers, n_hadrons), -1)

    # Fill the padded array
    for i, jet in enumerate(extended_family_indices):
        for j, shower in enumerate(jet):
            if j < n_max_showers:  # Limit the amount of showers
                hadrons_to_fill = shower[:n_hadrons]  # Limit to max number of hadrons
                padded_hadron_indices[i, j, : len(hadrons_to_fill)] = hadrons_to_fill

    return padded_hadron_indices


def AssociateTracksToHadron(track_parent, hadron_barcode, hadron_mask):
    # INPUTS #####
    # track_parent         shape (n_jets, n_tracks)
    # hadron_barcode       shape (n_jets, n_hadrons)
    # track_hadron_mask    shape (n_hadrons, n_jets, n_tracks)
    # OUTPUTS #####
    # track_to_hadron_array           shape (n_jets, n_hadrons, n_tracks)
    # inclusive_track_first_hadron    shape (n_jets, n_tracks)
    # inclusive_track_hadron          shape (n_jets, n_tracks)

    n_jets, n_tracks = track_parent.shape
    n_hadrons = hadron_barcode.shape[1]

    track_parent = np.where(
        track_parent < 0, np.nan, track_parent
    )  # use NAN so that they never match
    track_to_hadron_array = np.array([
        np.where(track_parent == hadron_barcode[:, k][:, np.newaxis], 1, 0)
        for k in range(n_hadrons)
    ])  # n_hadrons change to variable

    # build the inclusive vertex if needed
    inclusive_track_hadron = np.sum(track_to_hadron_array, axis=0)

    # Sum tracks from hadrons in the parton shower (applying the mask)
    track_hadron_mask = np.repeat(hadron_mask, n_tracks).reshape(n_hadrons, n_jets, n_tracks)
    inclusive_track_first_hadron = np.sum(
        np.where(track_hadron_mask, track_to_hadron_array, 0), axis=0
    )  # apply mask and sum across the first dimension

    # mask out hadrons with only one associated track
    mask_array = [
        np.repeat(
            np.where(np.sum(track_to_hadron_array[k], axis=1) >= 2, 1, 0)[:, np.newaxis], n_tracks
        ).reshape(n_jets, n_tracks)
        for k in range(n_hadrons)
    ]
    track_to_hadron_array = np.where(mask_array, track_to_hadron_array, mask_array)

    return np.array(track_to_hadron_array), inclusive_track_first_hadron, inclusive_track_hadron


def SelectHadron(truth_hadrons, hadron_index):
    invalid_jet_mask = hadron_index < 0

    # Select hadron with most tracks
    selected_hadron = truth_hadrons[np.arange(truth_hadrons.shape[0]), hadron_index.astype(int)]

    # Create a copy to preserve shape
    selected_hadron_copy = np.copy(selected_hadron)

    # Apply the mask and set invalid entries to np.nan (or you can set to 0)
    selected_hadron_copy[invalid_jet_mask] = -99  # Use np.nan or 0 based on preference

    return selected_hadron_copy


def select_tracks(track_hadron, index, element=0):
    rows = np.arange(track_hadron.shape[1])
    return track_hadron[index[rows, element], rows, :]
