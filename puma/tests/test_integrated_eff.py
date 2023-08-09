#!/usr/bin/env python


"""Unit test script for the functions in roc.py."""

import os
import tempfile
import unittest

import numpy as np
from matplotlib.testing.compare import compare_images

from puma import IntegratedEfficiency, IntegratedEfficiencyPlot
from puma.utils import get_dummy_2_taggers
from puma.utils.logging import logger, set_log_level

set_log_level(logger, "DEBUG")


class IntegratedEfficiencyCase(unittest.TestCase):
    """Test class for the puma.roc functions."""

    def setUp(self):
        self.disc_sig = np.random.normal(0.7, 1, 100)
        self.disc_rej = np.random.normal(0.4, 1, 100)

    def test_int_eff_init(self):
        """Test init."""
        IntegratedEfficiency(self.disc_sig, self.disc_rej)

    def test_add_label_flavour(self):
        """Test both label and flavour"""
        int_eff = IntegratedEfficiency(
            self.disc_sig, self.disc_rej, label="b-jets", flavour="ujets"
        )
        self.assertEqual(int_eff.label, "b-jets")


class IntegratedEfficiencyPlotTestCase(unittest.TestCase):
    """Test class for the puma.IntegratedEfficiencyPlot."""

    def setUp(self):
        # Set up temp directory for comparison plots
        self.tmp_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        self.actual_plots_dir = f"{self.tmp_dir.name}/"
        self.expected_plots_dir = os.path.join(
            os.path.dirname(__file__), "expected_plots"
        )

        # Set up dummy data
        df = get_dummy_2_taggers(size=int(1e6))
        fc = 0.018
        df["disc_dips"] = np.log(
            df["dips_pb"] / (fc * df["dips_pc"] + (1 - fc) * df["dips_pu"])
        )
        df["disc_rnnip"] = np.log(
            df["rnnip_pb"] / (fc * df["rnnip_pc"] + (1 - fc) * df["rnnip_pu"])
        )
        is_light = df["HadronConeExclTruthLabelID"] == 0
        is_c = df["HadronConeExclTruthLabelID"] == 4
        is_b = df["HadronConeExclTruthLabelID"] == 5
        self.dips_int_effs = {
            "light": IntegratedEfficiency(
                df["disc_dips"][is_b],
                df["disc_dips"][is_light],
                n_vals=200,
                tagger="DIPS",
                flavour="ujets",
            ),
            "c": IntegratedEfficiency(
                df["disc_dips"][is_b],
                df["disc_dips"][is_c],
                n_vals=200,
                tagger="DIPS",
                flavour="cjets",
            ),
            "b": IntegratedEfficiency(
                df["disc_dips"][is_b],
                df["disc_dips"][is_b],
                n_vals=200,
                tagger="DIPS",
                flavour="bjets",
            ),
        }
        self.rnnip_int_effs = {
            "light": IntegratedEfficiency(
                df["disc_rnnip"][is_b],
                df["disc_rnnip"][is_light],
                n_vals=200,
                tagger="RNNIP",
                flavour="ujets",
            ),
            "c": IntegratedEfficiency(
                df["disc_rnnip"][is_b],
                df["disc_rnnip"][is_c],
                n_vals=200,
                tagger="RNNIP",
                flavour="cjets",
            ),
            "b": IntegratedEfficiency(
                df["disc_rnnip"][is_b],
                df["disc_rnnip"][is_b],
                n_vals=200,
                tagger="RNNIP",
                flavour="bjets",
            ),
        }

    def test_duplicate_key(self):
        """Test duplicate key."""
        plot = IntegratedEfficiencyPlot()
        plot.add(self.dips_int_effs["b"], key=1)
        with self.assertRaises(KeyError):
            plot.add(self.dips_int_effs["c"], key=1)

    def test_output_one_tagger(self):
        """Test with one tagger."""
        plot = IntegratedEfficiencyPlot(grid=True)
        for flav in ["b", "c", "light"]:
            plot.add(self.dips_int_effs[flav])
        plot.draw()
        plotname = "test_int_eff_one_tagger.png"
        plot.savefig(f"{self.actual_plots_dir}/{plotname}")
        # Uncomment line below to update expected image
        # plot.savefig(f"{self.expected_plots_dir}/{plotname}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{plotname}",
                f"{self.expected_plots_dir}/{plotname}",
                tol=1,
            )
        )

    def test_output_two_taggers(self):
        """Test with one tagger."""
        plot = IntegratedEfficiencyPlot(grid=True)
        for flav in ["b", "c", "light"]:
            plot.add(self.dips_int_effs[flav])
            plot.add(self.rnnip_int_effs[flav])
        plot.draw()
        plotname = "test_int_eff_two_taggers.png"
        plot.savefig(f"{self.actual_plots_dir}/{plotname}")
        # Uncomment line below to update expected image
        # plot.savefig(f"{self.expected_plots_dir}/{plotname}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{plotname}",
                f"{self.expected_plots_dir}/{plotname}",
                tol=1,
            )
        )
