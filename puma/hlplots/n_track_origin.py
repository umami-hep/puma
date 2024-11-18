from __future__ import annotations

import os

import matplotlib as mpl
import numpy as np
from ftag import Label
from ftag.hdf5 import H5Reader

from puma.utils import get_good_colours, get_good_linestyles
from puma.var_vs_var import VarVsVar, VarVsVarPlot


def n_tracks_per_origin(
    flavour_list: list[Label],
    files: dict[dict[str]],
    plot_type: str,
    plot_path: str | None = None,
    track_origin_dict: dict | None = None,
    plot_format: str = "pdf",
    plot_name: str | None = None,
    **kwargs,
):
    """Plot number of tracks per track origin plot as a function of pT.

    Parameters
    ----------
    flavour_list : list[Flavour]
        List of Flavour objects to consider.
    files : dict[dict[str]]
        Dict of dicts with the samples to use. Each sub-dict has the following variables:
        filepath : str
            Path to the file.
        process_label : str
            Label of the process stored in the file for the legend.
        pt_bins : np.ndarray
            Numpy array with the pT bins to use.
        n_jets : int, optional
            Number of jets to load. By default load all jets.
        jets_name : str, optional
            Name of the jet collection in the h5 files. By default "jets".
        tracks_name : str, optional
            Name of the track collection in the h5 files. By default "tracks".
        jet_pt_variable : str, optional
            Name of the jet pT variable in the jet collection. By default "pt".
        track_truth_variable : str, optional
            Name of the track truth origin variable. By default "ftagTruthOriginLabel".
        flavour_label_variable : str, optional
            Name of the flavour label variable. By default "HadronConeExclTruthLabelID".
    plot_type : str
        Decide, which type of plot is wanted. Supported are "all_samples_one_flavour"
        or "one_sample_all_flavour".
    plot_path : str, optional
        Path to the folder where the plots will be stored. If None, plot is saved to current
        directory. By default None
    track_origin_dict : dict | None, optional
        Dict with the track origin names as keys and their origin value(s) as a list.
        With this, you can combine different origins and give them a new name. See the
        definition of the default track_origin_dict in the code. By default None
    plot_format : str, optional
        Plot format, by default "pdf"
    plot_name : str, optional
        Extra for the plot name, by default None

    Raises
    ------
    ValueError
        If the given plot_type is not supported.
    """
    # Check which type of plot is wanted
    if plot_type.lower() == "all_samples_one_flavour":
        all_flav_plot = False

    elif plot_type.lower() == "one_sample_all_flavour":
        all_flav_plot = True

    else:
        raise ValueError(
            'Only "all_samples_one_flavour" and "one_sample_all_flavour" are supported!'
        )

    # Get the grouping of the track origins
    if track_origin_dict is None:
        track_origin_dict = {
            "All": range(8),
            "HF decay": [3, 4, 5],
            # "Fragmentation": [1, 2],
            "From $\\tau$": [6],
            "Others": [0, 1, 2, 7],
        }

    # Init a default kwargs dict for the VarVsVarPlot
    VarVsVarPlot_kwargs = {
        "xlabel": "$p_T$ [GeV]",
        "ylabel": "Average $N_\\mathrm{Tracks}$ per Origin",
        "figsize": (7.0, 4.5),
        "y_scale": 1.3,
        "atlas_first_tag": "Simulation Internal",
        "atlas_second_tag": None,
        "logx": False,
        "logy": False,
    }

    # If kwargs are given, update the VarVsVarPlot_kwargs dict
    if kwargs is not None:
        VarVsVarPlot_kwargs.update(kwargs)

    # Init a dict for the plots
    var_plot_dict = {}

    # Check plot type and init plot if needed
    if all_flav_plot:
        for file_key, file_value in files.items():
            unmodified_second_tag = VarVsVarPlot_kwargs.get("atlas_second_tag")

            # Modify the second tag
            if unmodified_second_tag is None:
                VarVsVarPlot_kwargs["atlas_second_tag"] = file_value["process_label"] + " Jets"

            else:
                VarVsVarPlot_kwargs["atlas_second_tag"] += f'\n{file_value["process_label"]} Jets'

            # Init the plot
            var_plot_dict[file_key] = VarVsVarPlot(**VarVsVarPlot_kwargs)

            # Reverse the change for the next plot
            VarVsVarPlot_kwargs["atlas_second_tag"] = unmodified_second_tag

    else:
        for flavour in flavour_list:
            unmodified_second_tag = VarVsVarPlot_kwargs.get("atlas_second_tag")

            # Modify the second tag
            if unmodified_second_tag is None:
                VarVsVarPlot_kwargs["atlas_second_tag"] = flavour.label

            else:
                VarVsVarPlot_kwargs["atlas_second_tag"] += f"\n{flavour.label}"

            # Init the plot
            var_plot_dict[flavour.name] = VarVsVarPlot(**VarVsVarPlot_kwargs)

            # Reverse the change for the next plot
            VarVsVarPlot_kwargs["atlas_second_tag"] = unmodified_second_tag

    # Init lists for legend handles
    file_handles = []
    flavour_handles = []
    trk_origin_handles = []

    # Iterate over the given files
    for file_counter, (file_key, file_value) in enumerate(files.items()):
        # Get the variable names for the given file
        jet_pt_variable = file_value.get("jet_pt_variable", "pt")
        flavour_label_variable = file_value.get(
            "flavour_label_variable",
            "HadronConeExclTruthLabelID",
        )
        track_truth_variable = file_value.get(
            "track_truth_variable",
            "ftagTruthOriginLabel",
        )

        # Get the name of the jets and tracks in the h5 files
        jets_name = file_value.get("jets_name", "jets")
        tracks_name = file_value.get("tracks_name", "tracks")

        # Get the pT bins for the given file
        pt_bins = file_value["pt_bins"]

        # Init the reader for the file
        reader = H5Reader(
            fname=file_value["filepath"],
            shuffle=False,
        )

        # Iterate over the flavour and load them from the file
        for flavour_counter, flavour in enumerate(flavour_list):
            # Get the iterator to correctly choose the plot to add to
            plot_iterator = file_key if all_flav_plot else flavour.name

            # Load the data
            data = reader.load(
                variables={
                    jets_name: [jet_pt_variable, flavour_label_variable],
                    tracks_name: [track_truth_variable],
                },
                num_jets=file_value.get("n_jets", None),
                cuts=flavour.cuts,
            )

            # Loop over the different track origins
            for trk_origin_counter, (trk_origin_key, track_origin_value) in enumerate(
                track_origin_dict.items()
            ):
                # Get the number of tracks from this specific origin
                n_trks_tmp = np.sum(
                    np.isin(
                        data[tracks_name][track_truth_variable],
                        track_origin_value,
                    ),
                    axis=1,
                )

                # Bin it in pT
                bin_indices = np.digitize(
                    data[jets_name][jet_pt_variable],
                    pt_bins,
                )

                # Calculate mean n_trks for the given origin in pT bins
                n_trks_means = np.array([
                    np.mean(n_trks_tmp[bin_indices == i]) for i in range(len(pt_bins) - 1)
                ])

                # Calculate the std error
                n_trks_std = [
                    np.sqrt(
                        np.sum(
                            (n_trks_tmp[bin_indices == i] - np.mean(n_trks_tmp[bin_indices == i]))
                            ** 2
                        )
                        / len(n_trks_tmp[bin_indices == i] - 1)
                    )
                    / np.sqrt(len(n_trks_tmp[bin_indices == i]))
                    for i in range(len(pt_bins) - 1)
                ]

                # Plot the curve
                var_plot_dict[plot_iterator].add(
                    VarVsVar(
                        x_var=(0.5 * (pt_bins[1:] + pt_bins[:-1])) / 1000,
                        y_var_mean=n_trks_means,
                        x_var_widths=(
                            (0.5 * (pt_bins[1:] + pt_bins[:-1])) / 1000 - pt_bins[:-1] / 1000
                        )
                        * 2,
                        y_var_std=n_trks_std,
                        plot_y_std=False,
                        linestyle=(
                            get_good_linestyles()[flavour_counter]
                            if all_flav_plot
                            else get_good_linestyles()[file_counter]
                        ),
                        marker="o",
                        markersize="4",
                        colour=get_good_colours()[trk_origin_counter],
                        label=None if all_flav_plot else file_value["process_label"],
                    )
                )

                # Append the track origin labels only once
                if flavour_counter == 0 and file_counter == 0:
                    trk_origin_handles.append(
                        mpl.lines.Line2D(
                            [],
                            [],
                            color=get_good_colours()[trk_origin_counter],
                            label=trk_origin_key,
                            linestyle=get_good_linestyles()[flavour_counter],
                            marker="o",
                            markersize="4",
                        )
                    )

            # Append the flavour labels only once per file
            if file_counter == 0:
                # Add the flavour to the legend
                flavour_handles.append(
                    mpl.lines.Line2D(
                        [],
                        [],
                        color="#000000",
                        label=flavour.label,
                        linestyle=get_good_linestyles()[flavour_counter],
                    )
                )

        # Add the process to the legend
        file_handles.append(
            mpl.lines.Line2D(
                [],
                [],
                color="#000000",
                label=file_value["process_label"],
                linestyle=get_good_linestyles()[file_counter],
            )
        )

    # Get the extra for the plot_name
    tmp_plot_name = "" if plot_name is None else plot_name + "_"

    # Iterate over all plots
    for iter_plot_name in var_plot_dict:
        # Draw the actual plot
        var_plot_dict[iter_plot_name].draw()

        # Remove initial legend
        var_plot_dict[iter_plot_name].axis_top.get_legend().remove()

        # Remove the puma legend and call adjust layout
        var_plot_dict[iter_plot_name].fig.tight_layout()

        # Add the flavour or the process legend to the plots
        iter_handles = flavour_handles if all_flav_plot else file_handles
        var_plot_dict[iter_plot_name].axis_top.add_artist(
            var_plot_dict[iter_plot_name].axis_top.legend(
                handles=iter_handles,
                labels=[handle.get_label() for handle in iter_handles],
                loc="upper right",
                ncol=1,
                frameon=False,
            )
        )

        # Add the track origin legend to the plots
        var_plot_dict[iter_plot_name].axis_top.add_artist(
            var_plot_dict[iter_plot_name].axis_top.legend(
                handles=trk_origin_handles,
                labels=[handle.get_label() for handle in trk_origin_handles],
                loc="upper center",
                ncol=2,
                frameon=False,
            )
        )

        # Safe the figure
        var_plot_dict[iter_plot_name].savefig(
            plot_name=os.path.join(
                plot_path,
                (
                    f"{tmp_plot_name}{iter_plot_name}_all_flavour.{plot_format}"
                    if all_flav_plot
                    else f"{tmp_plot_name}{iter_plot_name}_all_samples.{plot_format}"
                ),
            )
        )
