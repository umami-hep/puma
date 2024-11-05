# Script for processing hadrons and getting fragmentation plots
from __future__ import annotations

import argparse
import os

import numpy as np

# Plotting
from ftag import Cuts
from ftag.hdf5 import H5Reader  # I use ftag tools to read the file

from puma import Histogram, HistogramPlot
from puma.utils.truth_hadron import (
    AssociateTracksToHadron,
    GetOrderedHadrons,
    SelectHadron,
    select_tracks,
)


def jet_flavour(jet, f=""):
    if f == "b":
        return jet["HadronConeExclTruthLabelID"] == 5
    if f == "c":
        return jet["HadronConeExclTruthLabelID"] == 4
    if f == "light":
        return jet["HadronConeExclTruthLabelID"] == 0
    return jet["HadronConeExclTruthLabelID"] >= 0


def LoadDataset(file_path, kinematic_cuts, n_jets=-1):
    track_var = [
        "dphi",
        "d0Uncertainty",
        "z0SinThetaUncertainty",
        "phiUncertainty",
        "thetaUncertainty",
        "qOverPUncertainty",
        "qOverP",
        "deta",
        "theta",
        "dphi",
    ]  # for vertex fit
    track_var += [
        "d0RelativeToBeamspot",
        "d0RelativeToBeamspotUncertainty",
        "z0RelativeToBeamspot",
        "z0RelativeToBeamspotUncertainty",
        "ftagTruthOriginLabel",
        "GN2v01_aux_TrackOrigin",
        "GN2v01_aux_VertexIndex",
        "ftagTruthVertexIndex",
        "ftagTruthParentBarcode",
    ]
    track_var += ["JFVertexIndex", "pt", "SV1VertexIndex", "valid"]

    jet_var = [
        "eventNumber",
        "GN2v01_pb",
        "GN2v01_pc",
        "GN2v01_pu",
        "phi",
        "eta",
        "HadronConeExclTruthLabelID",
        "HadronConeExclExtendedTruthLabelID",
    ]  # \phi is needed for vertex fit if track phi is not available # v00 instead of v01

    truth_hadrons = [
        "pt",
        "mass",
        "eta",
        "phi",
        "displacementZ",
        "Lxy",
        "pdgId",
        "barcode",
        "ftagTruthParentBarcode",
        "valid",
    ]

    event_var = [
        "primaryVertexToBeamDisplacementX",
        "primaryVertexToBeamDisplacementY",
        "primaryVertexToBeamDisplacementZ",
        "primaryVertexToTruthVertexDisplacementX",
        "primaryV\
ertexToTruthVertexDisplacementY",
        "primaryVertexToTruthVertexDisplacementZ",
    ]

    # read it!
    my_reader = H5Reader(file_path, precision="full", shuffle=False, batch_size=100)

    if n_jets == -1:
        my_data = my_reader.load(
            {
                "jets": jet_var,
                "tracks": track_var,
                "truth_hadrons": truth_hadrons,
                "eventwise": event_var,
            },
            cuts=kinematic_cuts,
        )
    else:
        my_data = my_reader.load(
            {
                "jets": jet_var,
                "tracks": track_var,
                "truth_hadrons": truth_hadrons,
                "eventwise": event_var,
            },
            num_jets=n_jets,
            cuts=kinematic_cuts,
        )

    return my_data


def ExtractHadronInfo(f, good_jets):
    tracks = f["tracks"][good_jets]
    # jets = f["jets"][good_jets]
    hadrons = f["truth_hadrons"][good_jets]

    # n_jets = jets.shape[0]
    # n_tracks = tracks.shape[1]
    # n_hadrons = hadrons.shape[1]

    ordered_hadron_indices = GetOrderedHadrons(
        hadrons["barcode"], hadrons["ftagTruthParentBarcode"], n_max_showers=2
    )

    # CHOICE:keep only the first shower
    shower_index = 0  # < n_max_showers --- Select the index most important shower
    hadron_indices_first_shower = ordered_hadron_indices[:, shower_index, :]

    # mask removing padded hadrons
    hadron_mask_first_shower = np.transpose(
        np.where(hadron_indices_first_shower >= 0, 1, 0)
    )  # (n_hadrons, n_jets)

    # Associate Tracks to Hadrons
    truth_SV_finding_exclusive, truth_SV_finding_first_shower, truth_SV_finding_inclusive = (
        AssociateTracksToHadron(
            tracks["ftagTruthParentBarcode"], hadrons["barcode"], hadron_mask_first_shower
        )
    )

    # Now re-order the exclusive ones using the hadron index
    tracks_associated_to_sv1 = select_tracks(
        truth_SV_finding_exclusive, hadron_indices_first_shower, element=0
    )
    n_tracks_sv1 = np.sum(tracks_associated_to_sv1, axis=1)

    tracks_associated_to_sv2 = select_tracks(
        truth_SV_finding_exclusive, hadron_indices_first_shower, element=1
    )
    n_tracks_sv2 = np.sum(tracks_associated_to_sv2, axis=1)

    tracks_associated_to_sv3 = select_tracks(
        truth_SV_finding_exclusive, hadron_indices_first_shower, element=2
    )
    n_tracks_sv3 = np.sum(tracks_associated_to_sv3, axis=1)

    # Calculate the number of tracks also for the inclusive vertex / showered from the first SV
    n_tracks_sv1_shower = np.sum(truth_SV_finding_first_shower, axis=1)
    n_tracks_inclusive_vertex = np.sum(truth_SV_finding_inclusive, axis=1)

    # Get the properties of each hadron
    sv_1 = SelectHadron(np.array(hadrons), hadron_indices_first_shower[:, 0])
    sv_2 = SelectHadron(np.array(hadrons), hadron_indices_first_shower[:, 1])
    sv_3 = SelectHadron(np.array(hadrons), hadron_indices_first_shower[:, 2])

    Exclusive_SV_1_2_3 = (
        (sv_1, n_tracks_sv1, tracks_associated_to_sv1),
        (sv_2, n_tracks_sv2, tracks_associated_to_sv2),
        (sv_3, n_tracks_sv3, tracks_associated_to_sv3),
    )

    SV1_with_shower = (sv_1, n_tracks_sv1_shower, truth_SV_finding_first_shower)

    InclusiveVertex = (sv_1, n_tracks_inclusive_vertex, truth_SV_finding_inclusive)

    return (Exclusive_SV_1_2_3, SV1_with_shower, InclusiveVertex)


def SV_Finding(
    vertex_index, track_origin, inclusive=False, remove_non_HF_tracks=True, min2tracks=True
):
    if remove_non_HF_tracks:
        raw_vertex_index = vertex_index.copy()
        # remove tracks not from HF
        # mask = np.where(track_origin > 2  , 1, 0)
        mask = np.where((track_origin > 2) & (track_origin < 6), 1, 0)
        vertex_index = np.where(mask, raw_vertex_index, -1)

    track_weights = np.array(
        [[[1 if r == i else 0 for r in row] for i in set(row[row >= 0])] for row in vertex_index],
        dtype="object",
    )

    max_jets = track_weights.shape[0]
    max_n_hadrons = max(len(inner_list) for inner_list in track_weights)
    n_tracks = vertex_index.shape[-1]
    padded_track_weights = np.zeros((max_jets, max_n_hadrons, n_tracks))

    for i, outer in enumerate(track_weights):
        for j, inner in enumerate(outer):
            padded_track_weights[i, j, : len(inner)] = (
                inner  # Only copy the actual values, pad with zeros
            )

    if inclusive:
        padded_track_weights = np.sum(padded_track_weights, axis=1)

        if min2tracks:
            # if only 1 track in vertex, set it off
            mask = np.sum(padded_track_weights, axis=1) == 1
            padded_track_weights[mask] = np.zeros(padded_track_weights.shape[1])

        return padded_track_weights

    if min2tracks:
        # if only 1 track in vertex, set it off
        mask = np.sum(padded_track_weights, axis=2) == 1
        padded_track_weights[mask] = np.zeros(padded_track_weights.shape[2])

    # Now remove empty vertices and repeat the padding
    track_weights = np.array(
        [[arr for arr in jet if np.sum(np.array(arr)) > 0] for jet in padded_track_weights],
        dtype="object",
    )  # > 1 drops 1 track vertices

    len_vertices = [len(inner_list) for inner_list in track_weights]
    max_n_hadrons = max(len_vertices)

    padded_track_weights = np.zeros((max_jets, max_n_hadrons, n_tracks))

    for i, outer in enumerate(track_weights):
        for j, inner in enumerate(outer):
            padded_track_weights[i, j, : len(inner)] = (
                inner  # Only copy the actual values, pad with zeros
            )

    return padded_track_weights


def main():
    # Set up argparse
    parser = argparse.ArgumentParser(description="Path to samples.")
    parser.add_argument("--path",  type=str, default="./", help="Path to sample list")
    parser.add_argument("--n_jets", type=int, default=300000, help="Number of jets to run")

    parser.add_argument("--sample", type=str, default="ttbar", help="Sample name")

    parser.add_argument("--mc", type=str, default="MC23a", help="Campaign name")

    parser.add_argument("--output", type=str, default="Plots/", help="Campaign name (MC23a)")

    # Parse the arguments
    args = parser.parse_args()

    path = args.path
    sample = args.sample
    mc = args.mc
    n_jets = args.n_jets

    path = path + "/" + mc + "_" + "new-" + sample + ".h5"

    output = args.output
    os.makedirs(output, exist_ok=True)

    if sample == "ttbar":
        sample_str = "$t\\overline{t}$"
        cuts = [
            ("pt", ">=", 20000),
            ("pt", "<=", 250000),
            ("eta", ">", -2.5),
            ("eta", "<", 2.5),
            ("HadronConeExclTruthLabelID", ">=", 0),
            ("HadronConeExclTruthLabelID", "<=", 5),
            ("n_truth_promptLepton", "==", 0),
        ]
        cut_str = "20 < $p_T$ < 250 GeV, $|\\eta| < 2.5$"

    elif sample == "zprime":
        sample_str = "Z'"
        cuts = [
            ("pt", ">=", 250000),
            ("pt", "<=", 6000000),
            ("eta", ">", -2.5),
            ("eta", "<", 2.5),
            ("HadronConeExclTruthLabelID", ">=", 0),
            ("HadronConeExclTruthLabelID", "<=", 5),
            ("n_truth_promptLepton", "==", 0),
        ]
        cut_str = "250 < $p_T$ < 6000 GeV, $|\\eta| < 2.5$"

    # Code starts here #########

    com = "13" if "MC20" in mc else "13.6"

    dataset = LoadDataset(path, Cuts.from_list(cuts), n_jets=n_jets)

    good_jets = np.where(dataset["jets"]["HadronConeExclExtendedTruthLabelID"] < 6, 1, 0).astype(
        bool
    )  # remove double b-jets

    _, sv1_shower, inclusive_vertex = ExtractHadronInfo(
        dataset, good_jets
    )  # process truth Hadron information (you will need it later)

    # select only jets with single b-tag label

    jets = dataset["jets"][good_jets]
    tracks = dataset["tracks"][good_jets]

    ############################
    # Reco (GN2)      ##
    ############################

    inclusive_vertex = True

    # Get Vertex Index, where are the secondary vertices?

    # this is the old function (it is slower and uses the FTAG clean vertex finding)

    # process SV finding ###

    vertex_index = tracks["GN2v01_aux_VertexIndex"]
    track_origin = tracks["GN2v01_aux_TrackOrigin"]

    tmp_track_weights = SV_Finding(
        vertex_index, track_origin, inclusive=False, remove_non_HF_tracks=False, min2tracks=False
    )

    # find vertex with highest number of tracks from PV (with origin = 2)

    origin_vertex_candidates = np.array([
        np.where(row, track_origin[r], 0) for r, row in enumerate(tmp_track_weights)
    ])
    pv_candidates = np.where(
        origin_vertex_candidates == 2, 1, 0
    )  # first check the amount of tracks with origin = 2 for each vertex candidate
    mask_tracks_pv = np.array([
        1 if np.sum(row) > 0 else 0 for row in np.where(np.sum(pv_candidates, axis=2) > 0, 1, 0)
    ])  # make a mask to only modify vertices with at least 1 track from the PV
    vertex_most_pv = np.argmax(
        np.sum(pv_candidates, axis=2), axis=1
    )  # find which vertex has most tracks from origin 2
    tracks_from_pv_candidate = origin_vertex_candidates[
        np.arange(pv_candidates.shape[0]), vertex_most_pv
    ]
    tracks_from_pv_candidate = np.where(mask_tracks_pv[:, np.newaxis], tracks_from_pv_candidate, 0)

    # drop vertex with highest number of tracks from PV

    clean_vertex_index = np.where(tracks_from_pv_candidate, -1, vertex_index)
    reco_track_weights = SV_Finding(
        clean_vertex_index,
        track_origin,
        inclusive=inclusive_vertex,
        remove_non_HF_tracks=True,
        min2tracks=True,
    )

    ############################
    # Fragmentation Plot ####
    ############################

    hadron_varaiables, _, tracks_hadron_family = sv1_shower

    truth_pt = hadron_varaiables["pt"] / 1000

    shower_tracks_pt = np.where(tracks_hadron_family, tracks["pt"] / 1000, 0)
    sum_track_pt = np.sum(shower_tracks_pt, axis=1)

    GN2_tracks_pt = np.where(reco_track_weights, tracks["pt"] / 1000, 0)
    GN2_sum_track_pt = np.sum(GN2_tracks_pt, axis=1)

    for flavour in ["bjets", "cjets"]:
        # Choose selection
        if flavour == "all":
            f = ""
            flav_str = ""

        if flavour == "bjets":
            f = "b"
            flav_str = "$b$-jets"

        if flavour == "cjets":
            f = "c"
            flav_str = "$c$-jets"

        if flavour == "light":
            f = "light"
            flav_str = "(light jets)"

        selection = jet_flavour(jets, f)

        h_hadron = Histogram(truth_pt[selection], label="Hadron (Truth)", histtype="step", alpha=1)
        h_tracks = Histogram(
            sum_track_pt[selection],
            label="$\\sum$ SV Hadron tracks (Reco)",
            histtype="step",
            alpha=1,
        )

        normalise = False
        y_axis = "Number of jets"
        if normalise:
            y_axis = "Normalised Arbitrary Units"

        # Initialise histogram plot
        plot_histo = HistogramPlot(
            ylabel=y_axis,
            xlabel=r"p$_{\mathrm{T}}$ [GeV]",
            logy=True,
            # bins=np.linspace(0, 5, 60),  # you can force a binning for the plot here
            bins=50,  # you can also define an integer number for the number of bins
            bins_range=(0, 250),  # only considered if bins is an integer
            norm=normalise,
            atlas_first_tag="Simulation Internal",
            atlas_second_tag="$\\sqrt{s}="
            + com
            + "$ TeV, "
            + mc
            + "\n"
            + sample_str
            + ", "
            + cut_str
            + " \n"
            + flav_str,
            figsize=(6, 5),
            y_scale=1.7,
            n_ratio_panels=1,
        )

        plot_histo.add(h_hadron, reference=True)
        plot_histo.add(h_tracks, reference=False)

        plot_histo.draw()
        plot_histo.savefig(output + "Fragmentation_Hadron_" + f + ".png", transparent=False)

        h_hadron = Histogram(truth_pt[selection], label="Hadron (Truth)", histtype="step", alpha=1)
        h_tracks = Histogram(
            GN2_sum_track_pt[selection], label="$\\sum$ tracks (GN2 SV)", histtype="step", alpha=1
        )

        normalise = False
        y_axis = "Number of jets"
        if normalise:
            y_axis = "Normalised Arbitrary Units"

        # Initialise histogram plot
        plot_histo = HistogramPlot(
            ylabel=y_axis,
            xlabel=r"p$_{\mathrm{T}}$ [GeV]",
            logy=True,
            # bins=np.linspace(0, 5, 60),  # you can force a binning for the plot here
            bins=50,  # you can also define an integer number for the number of bins
            bins_range=(0, 250),  # only considered if bins is an integer
            norm=normalise,
            atlas_first_tag="Simulation Internal",
            atlas_second_tag="$\\sqrt{s}="
            + com
            + "$ TeV, "
            + mc
            + "\n"
            + sample_str
            + ", "
            + cut_str
            + " \n"
            + flav_str,
            figsize=(6, 5),
            y_scale=1.7,
            n_ratio_panels=1,
        )

        plot_histo.add(h_hadron, reference=True)
        plot_histo.add(h_tracks, reference=False)

        plot_histo.draw()
        plot_histo.savefig(output + "Fragmentation_GN2_" + f + ".png", transparent=False)

    all_tracks_pt = np.sum(np.where(tracks["valid"], tracks["pt"] / 1000, 0), axis=1)

    var = sum_track_pt / all_tracks_pt

    h_b = Histogram(var[(jet_flavour(jets, "b"))], label="b-jets", histtype="step", alpha=1)
    h_c = Histogram(var[(jet_flavour(jets, "c"))], label="c-jets", histtype="step", alpha=1)
    h_l = Histogram(var[(jet_flavour(jets, "light"))], label="light jets", histtype="step", alpha=1)

    normalise = True
    y_axis = "Number of jets"
    if normalise:
        y_axis = "Normalised Arbitrary Units"

    # Initialise histogram plot
    pt = "$\\sum$ p$_\\mathrm{T}$"
    plot_histo = HistogramPlot(
        ylabel=y_axis,
        xlabel=pt + " tracks $\\in$ (SV Hadron) / " + pt + " tracks $\\in$ jet",
        logy=True,
        # bins=np.linspace(0, 5, 60),  # you can force a binning for the plot here
        bins=25,  # you can also define an integer number for the number of bins
        bins_range=(0, 1),  # only considered if bins is an integer
        norm=normalise,
        atlas_first_tag="Simulation Internal",
        atlas_second_tag="$\\sqrt{s}=" + com + "$ TeV, " + mc + "\n" + sample_str + ", " + cut_str,
        figsize=(6, 5),
        # y_scale=1.7,
        n_ratio_panels=0,
    )

    plot_histo.add(h_b, reference=False)
    plot_histo.add(h_c, reference=False)
    plot_histo.add(h_l, reference=False)

    plot_histo.draw()
    plot_histo.savefig(output + "Fragmentation_pT_Hadrons_all_flav.png", transparent=False)

    var = GN2_sum_track_pt / all_tracks_pt

    h_b = Histogram(var[(jet_flavour(jets, "b"))], label="b-jets", histtype="step", alpha=1)
    h_c = Histogram(var[(jet_flavour(jets, "c"))], label="c-jets", histtype="step", alpha=1)
    h_l = Histogram(var[(jet_flavour(jets, "light"))], label="light jets", histtype="step", alpha=1)

    normalise = True
    y_axis = "Number of jets"
    if normalise:
        y_axis = "Normalised Arbitrary Units"

    # Initialise histogram plot
    plot_histo = HistogramPlot(
        ylabel=y_axis,
        xlabel=pt + " tracks $\\in$ (SV Hadron) / " + pt + " tracks $\\in$ jet",
        logy=True,
        # bins=np.linspace(0, 5, 60),  # you can force a binning for the plot here
        bins=25,  # you can also define an integer number for the number of bins
        bins_range=(0, 1),  # only considered if bins is an integer
        norm=normalise,
        atlas_first_tag="Simulation Internal",
        atlas_second_tag="$\\sqrt{s}=" + com + "$ TeV, " + mc + "\n" + sample_str + ", " + cut_str,
        figsize=(6, 5),
        # y_scale=1.7,
        n_ratio_panels=0,
    )

    plot_histo.add(h_b, reference=False)
    plot_histo.add(h_c, reference=False)
    plot_histo.add(h_l, reference=False)

    plot_histo.draw()
    plot_histo.savefig(output + "Fragmentation_pT_GN2_all_flav.png", transparent=False)


if __name__ == "__main__":
    main()
