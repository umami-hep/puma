"""Unit test script for the functions in hlplots/n_track_origin.py."""

from __future__ import annotations

import os

# Uncomment line below to update expected image
# import shutil
import tempfile
import unittest
from urllib.request import urlretrieve

import numpy as np
from ftag import Flavours
from matplotlib.testing.compare import compare_images

from puma.hlplots import n_tracks_per_origin
from puma.utils import logger, set_log_level

set_log_level(logger, "DEBUG")


class NTrackOriginTestCase(unittest.TestCase):
    """Test class for the function."""

    def setUp(self):
        """Create a default dataset for testing."""
        self.tmp_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        self.actual_plots_dir = f"{self.tmp_dir.name}/"
        self.data_url = "https://umami-ci-provider.web.cern.ch/plot_input_vars/"
        self.r22_url = os.path.join(self.data_url, "plot_input_vars_r22_check.h5")
        self.r22_test_file = os.path.join(self.actual_plots_dir, "plot_input_vars_r22_check.h5")
        self.expected_plots_dir = os.path.join(os.path.dirname(__file__), "expected_plots")
        urlretrieve(self.r22_url, self.r22_test_file)  # noqa: S310

    def test_one_sample_all_flavour(self):
        """Test the one_sample_all_flavour plot."""
        n_tracks_per_origin(
            flavour_list=[
                Flavours.bjets,
                Flavours.cjets,
                Flavours.ujets,
            ],
            files={
                "ttbar": {
                    "filepath": self.r22_test_file,
                    "n_jets": 10000,
                    "tracks_name": "tracks_loose",
                    "pt_bins": np.linspace(20_000, 250_000, 20),
                    "process_label": "$t\\bar{t}$",
                    "jet_pt_variable": "pt_btagJes",
                    "track_truth_variable": "truthOriginLabel",
                    "flavour_label_variable": "HadronConeExclTruthLabelID",
                },
            },
            plot_path=self.actual_plots_dir,
            plot_type="one_sample_all_flavour",
            track_origin_dict=None,
            plot_format="png",
        )

        # Define a plot name
        name = "ttbar_all_flavour.png"

        # Uncomment line below to update expected image
        # shutil.copy(
        #     f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}"
        # )

        # Check
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{name}",
                f"{self.expected_plots_dir}/{name}",
                tol=1,
            )
        )

    def test_all_samples_one_flavour(self):
        """Test the all_samples_one_flavour plot."""
        n_tracks_per_origin(
            flavour_list=[
                Flavours.bjets,
                Flavours.cjets,
                Flavours.ujets,
            ],
            files={
                "ttbar_1": {
                    "filepath": self.r22_test_file,
                    "n_jets": 10000,
                    "tracks_name": "tracks_loose",
                    "pt_bins": np.linspace(20_000, 100_000, 20),
                    "process_label": "$t\\bar{t}$ 1",
                    "jet_pt_variable": "pt_btagJes",
                    "track_truth_variable": "truthOriginLabel",
                    "flavour_label_variable": "HadronConeExclTruthLabelID",
                },
                "ttbar_2": {
                    "filepath": self.r22_test_file,
                    "n_jets": 10000,
                    "tracks_name": "tracks_loose",
                    "pt_bins": np.linspace(100_000, 250_000, 20),
                    "process_label": "$t\\bar{t}$ 2",
                    "jet_pt_variable": "pt_btagJes",
                    "track_truth_variable": "truthOriginLabel",
                    "flavour_label_variable": "HadronConeExclTruthLabelID",
                },
            },
            plot_path=self.actual_plots_dir,
            plot_type="all_samples_one_flavour",
            track_origin_dict=None,
            plot_format="png",
        )

        # Define a plot name
        names = [
            "bjets_all_samples.png",
            "cjets_all_samples.png",
            "ujets_all_samples.png",
        ]

        # Check all plots
        for name in names:
            # Uncomment line below to update expected image
            # shutil.copy(
            #     f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}"
            # )

            # Check
            self.assertIsNone(
                compare_images(
                    f"{self.actual_plots_dir}/{name}",
                    f"{self.expected_plots_dir}/{name}",
                    tol=1,
                )
            )

    def test_extra_kwargs_all_flavour(self):
        """Test extra kwargs."""
        n_tracks_per_origin(
            flavour_list=[
                Flavours.bjets,
                Flavours.cjets,
                Flavours.ujets,
            ],
            files={
                "ttbar_test_kwargs": {
                    "filepath": self.r22_test_file,
                    "n_jets": 10000,
                    "tracks_name": "tracks_loose",
                    "pt_bins": np.linspace(20_000, 250_000, 20),
                    "process_label": "$t\\bar{t}$",
                    "jet_pt_variable": "pt_btagJes",
                    "track_truth_variable": "truthOriginLabel",
                    "flavour_label_variable": "HadronConeExclTruthLabelID",
                },
            },
            plot_path=self.actual_plots_dir,
            plot_type="one_sample_all_flavour",
            track_origin_dict=None,
            plot_format="png",
            atlas_second_tag="Test extra kwargs",
        )

        # Define a plot name
        name = "ttbar_test_kwargs_all_flavour.png"

        # Uncomment line below to update expected image
        # shutil.copy(
        #     f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}"
        # )

        # Check
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{name}",
                f"{self.expected_plots_dir}/{name}",
                tol=1,
            )
        )

    def test_extra_kwargs_all_samples(self):
        """Test extra kwargs."""
        n_tracks_per_origin(
            flavour_list=[
                Flavours.bjets,
                Flavours.cjets,
                Flavours.ujets,
            ],
            files={
                "ttbar_1": {
                    "filepath": self.r22_test_file,
                    "n_jets": 10000,
                    "tracks_name": "tracks_loose",
                    "pt_bins": np.linspace(20_000, 100_000, 20),
                    "process_label": "$t\\bar{t}$ 1",
                    "jet_pt_variable": "pt_btagJes",
                    "track_truth_variable": "truthOriginLabel",
                    "flavour_label_variable": "HadronConeExclTruthLabelID",
                },
                "ttbar_2": {
                    "filepath": self.r22_test_file,
                    "n_jets": 10000,
                    "tracks_name": "tracks_loose",
                    "pt_bins": np.linspace(100_000, 250_000, 20),
                    "process_label": "$t\\bar{t}$ 2",
                    "jet_pt_variable": "pt_btagJes",
                    "track_truth_variable": "truthOriginLabel",
                    "flavour_label_variable": "HadronConeExclTruthLabelID",
                },
            },
            plot_path=self.actual_plots_dir,
            plot_type="all_samples_one_flavour",
            plot_name="extra_kwargs",
            track_origin_dict=None,
            plot_format="png",
            atlas_second_tag="Test extra kwargs",
        )

        # Define a plot name
        names = [
            "extra_kwargs_bjets_all_samples.png",
            "extra_kwargs_cjets_all_samples.png",
            "extra_kwargs_ujets_all_samples.png",
        ]

        # Check all plots
        for name in names:
            # Uncomment line below to update expected image
            # shutil.copy(f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}")

            # Check
            self.assertIsNone(
                compare_images(
                    f"{self.actual_plots_dir}/{name}",
                    f"{self.expected_plots_dir}/{name}",
                    tol=1,
                )
            )

    def test_custom_track_origins(self):
        """Test custom grouping of track origins."""
        n_tracks_per_origin(
            flavour_list=[
                Flavours.bjets,
                Flavours.cjets,
                Flavours.ujets,
            ],
            files={
                "ttbar_test_custom_track_origins": {
                    "filepath": self.r22_test_file,
                    "n_jets": 10000,
                    "tracks_name": "tracks_loose",
                    "pt_bins": np.linspace(20_000, 250_000, 20),
                    "process_label": "$t\\bar{t}$",
                    "jet_pt_variable": "pt_btagJes",
                    "track_truth_variable": "truthOriginLabel",
                    "flavour_label_variable": "HadronConeExclTruthLabelID",
                },
            },
            plot_path=self.actual_plots_dir,
            plot_type="one_sample_all_flavour",
            track_origin_dict={
                "All": range(8),
                "Fragmentation": [1, 2],
                "HF decay": [3, 4, 5],
                "From $\\tau$": [6],
                "Others": [7],
            },
            plot_format="png",
        )

        # Define a plot name
        name = "ttbar_test_custom_track_origins_all_flavour.png"

        # Uncomment line below to update expected image
        # shutil.copy(f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}")

        # Check
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{name}",
                f"{self.expected_plots_dir}/{name}",
                tol=1,
            )
        )

    def test_wrong_plot_type(self):
        """Test wrong plot type error raising."""
        with self.assertRaises(ValueError):
            n_tracks_per_origin(
                flavour_list=[
                    Flavours.bjets,
                    Flavours.cjets,
                    Flavours.ujets,
                ],
                files={
                    "ttbar_test_custom_track_origins": {
                        "filepath": self.r22_test_file,
                        "n_jets": 10000,
                        "tracks_name": "tracks_loose",
                        "pt_bins": np.linspace(20_000, 250_000, 20),
                        "process_label": "$t\\bar{t}$",
                        "jet_pt_variable": "pt_btagJes",
                        "track_truth_variable": "truthOriginLabel",
                        "flavour_label_variable": "HadronConeExclTruthLabelID",
                    },
                },
                plot_path=self.actual_plots_dir,
                plot_type="crash",
                track_origin_dict=None,
                plot_format="png",
            )
