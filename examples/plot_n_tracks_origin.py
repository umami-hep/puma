"""Plotting example of the n_tracks_per_origin function."""

from __future__ import annotations

from urllib.request import urlretrieve

import numpy as np
from ftag import Flavours

from puma.hlplots import n_tracks_per_origin

# Download test h5 sample
urlretrieve(
    "https://umami-ci-provider.web.cern.ch/plot_input_vars/plot_input_vars_r22_check.h5",
    "testfile.h5",
)

# Define the grouping of the track origins
track_origin_dict = {
    "All": range(8),
    # "Fragmentation": [1, 2],
    "HF decay": [3, 4, 5],
    # "From $\\tau$": [6],
    "Others": [0, 1, 2, 6, 7],
}

# Define a list with all flavours that should be plotted
flavour_list = [
    Flavours.bjets,
    Flavours.cjets,
    Flavours.ujets,
]

# Define the dict with all samples that are to be plotted
files_dict = {
    "ttbar": {
        "filepath": "testfile.h5",
        "n_jets": 10000,
        "tracks_name": "tracks_loose",
        "pt_bins": np.linspace(20_000, 250_000, 20),
        "process_label": "$t\\bar{t}$",
        "jet_pt_variable": "pt_btagJes",
        "track_truth_variable": "truthOriginLabel",
        "flavour_label_variable": "HadronConeExclTruthLabelID",
    },
}

# Plotting all samples for one flavour per plot
n_tracks_per_origin(
    flavour_list=flavour_list,
    files=files_dict,
    track_origin_dict=track_origin_dict,
    plot_type="all_samples_one_flavour",
    plot_path="./",
    plot_format="pdf",
    plot_name="Test_plots",
)

# Plotting all flavour for one sample per plot
n_tracks_per_origin(
    flavour_list=flavour_list,
    files=files_dict,
    track_origin_dict=track_origin_dict,
    plot_type="one_sample_all_flavour",
    plot_path="./",
    plot_format="pdf",
    plot_name="Test_plots",
)
