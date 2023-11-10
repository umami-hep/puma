#!/usr/bin/env python


"""Unit test script for the functions in var_vs_var.py."""
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from puma.utils.logging import logger, set_log_level
from puma.var_vs_aux import VarVsAux, VarVsAuxPlot

set_log_level(logger, "DEBUG")


class VarVsAuxTestCase(unittest.TestCase):
    """Test class for the puma.var_vs_aux functions."""

    def setUp(self):
        np.random.seed(42)

        self.x_var = np.linspace(100, 250, 20)
        self.n_true = np.random.randint(1, 10, size=20)
        self.n_reco = np.random.randint(1, 10, size=20)
        self.n_match = np.random.randint(0, np.minimum(self.n_true, self.n_reco))

    def test_var_vs_aux_init_wrong_true_shape(self):
        """Test var_vs_aux init."""
        with self.assertRaises(ValueError):
            VarVsAux(
                x_var=np.ones(4),
                n_match=np.ones(4),
                n_true=np.ones(5),
                n_reco=np.ones(4),
            )

    def test_var_vs_aux_init_wrong_reco_shape(self):
        """Test var_vs_aux init."""
        with self.assertRaises(ValueError):
            VarVsAux(
                x_var=np.ones(4),
                n_match=np.ones(4),
                n_true=np.ones(4),
                n_reco=np.ones(5),
            )

    def test_var_vs_aux_init_wrong_match_shape(self):
        """Test var_vs_aux init."""
        with self.assertRaises(ValueError):
            VarVsAux(
                x_var=np.ones(4),
                n_match=np.ones(5),
                n_true=np.ones(4),
                n_reco=np.ones(4),
            )

    def test_var_vs_aux_init(self):
        """Test var_vs_aux init."""
        VarVsAux(
            x_var=np.ones(6),
            n_match=np.ones(6),
            n_true=np.ones(6),
            n_reco=np.ones(6),
            bins=10,
            key="test",
        )

    def test_var_vs_aux_set_bin_edges_list(self):
        """Test var_vs_aux _set_bin_edges."""
        var_plot = VarVsAux(
            x_var=[0, 1, 2],
            n_match=[3, 4, 5],
            n_true=[3, 4, 5],
            n_reco=[3, 4, 5],
            bins=[0, 1, 2],
        )
        np.testing.assert_array_almost_equal(var_plot.bin_edges, [0, 1, 2])

    def test_var_vs_aux_set_bin_edges(self):
        """Test var_vs_aux _set_bin_edges."""
        var_plot = VarVsAux(
            x_var=[0, 1, 2],
            n_match=[3, 4, 5],
            n_true=[3, 4, 5],
            n_reco=[3, 4, 5],
            bins=2,
        )
        np.testing.assert_array_almost_equal(var_plot.bin_edges, [0, 1, 2], decimal=4)

    def test_var_vs_aux_eq_different_classes(self):
        """Test var_vs_eff eq."""
        var_plot = VarVsAux(
            x_var=[0, 1, 2],
            n_match=[3, 4, 5],
            n_true=[3, 4, 5],
            n_reco=[3, 4, 5],
            bins=2,
        )
        self.assertNotEqual(var_plot, np.ones(6))

    def test_var_vs_eff_get(self):
        var_plot = VarVsAux(
            x_var=self.x_var,
            n_match=self.n_match,
            n_true=self.n_true,
            n_reco=self.n_reco,
            bins=1,
        )
        mode_options = ["efficiency", "fake_rate"]
        for mode in mode_options:
            var_plot.get(mode)


class VarVsAuxPlotTestCase(unittest.TestCase):
    """Test class for the puma.var_vs_aux_plot."""

    def setUp(self):
        # Set up temp directory for comparison plots
        self.tmp_dir = tempfile.TemporaryDirectory()  # pylint:disable=R1732
        self.actual_plots_dir = f"{self.tmp_dir.name}/"
        self.expected_plots_dir = os.path.join(
            os.path.dirname(__file__), "expected_plots"
        )
        np.random.seed(42)
        n_random = 10_000

        # truth variables - shared between taggers
        self.x_var = np.random.uniform(0, 250, size=n_random)
        self.n_true = np.random.randint(1, 10, size=n_random)

        # tagger 1
        self.n_reco_1 = np.random.randint(1, 10, size=n_random)
        self.n_match_1 = np.random.randint(0, np.minimum(self.n_true, self.n_reco_1))

        # tagger 2
        self.n_reco_2 = np.random.randint(1, 10, size=n_random)
        self.n_match_2 = np.random.randint(0, np.minimum(self.n_true, self.n_reco_2))

        # Define pT bins
        self.bins = [20, 30, 40, 60, 85, 110, 140, 175, 250]

    def test_var_vs_aux_plot_mode_wrong_option(self):
        with self.assertRaises(ValueError):
            VarVsAuxPlot(
                mode="test",
                ylabel="Aux fake rate",
                xlabel=r"$p_{T}$ [GeV]",
                logy=True,
                atlas_second_tag="test",
                y_scale=1.5,
                n_ratio_panels=1,
                figsize=(9, 6),
            )

    def test_var_vs_aux_plot_mode_efficiency(self):
        """Test aux output plot - efficiency."""
        # define the curves
        tagger1 = VarVsAux(
            x_var=self.x_var,
            n_true=self.n_true,
            n_reco=self.n_reco_1,
            n_match=self.n_match_1,
            bins=self.bins,
            label="tagger 1",
        )
        tagger2 = VarVsAux(
            x_var=self.x_var,
            n_true=self.n_true,
            n_reco=self.n_reco_2,
            n_match=self.n_match_2,
            bins=self.bins,
            label="tagger 2",
        )

        plot_eff = VarVsAuxPlot(
            mode="efficiency",
            ylabel="Aux fake rate",
            xlabel=r"$p_{T}$ [GeV]",
            logy=True,
            y_scale=1.5,
            n_ratio_panels=1,
            figsize=(9, 6),
        )
        plot_eff.add(tagger1, reference=True)
        plot_eff.add(tagger2)

        plot_eff.draw()

        plotname = "test_aux_efficiency.png"
        plot_eff.savefig(f"{self.actual_plots_dir}/{plotname}")

    def test_var_vs_aux_plot_mode_fake_rate(self):
        """Test aux output plot - fake rate."""
        # define the curves
        tagger1 = VarVsAux(
            x_var=self.x_var,
            n_true=self.n_true,
            n_reco=self.n_reco_1,
            n_match=self.n_match_1,
            bins=self.bins,
            label="tagger 1",
        )
        tagger2 = VarVsAux(
            x_var=self.x_var,
            n_true=self.n_true,
            n_reco=self.n_reco_2,
            n_match=self.n_match_2,
            bins=self.bins,
            label="tagger 2",
        )

        plot_fr = VarVsAuxPlot(
            mode="fake_rate",
            ylabel="Aux fake rate",
            xlabel=r"$p_{T}$ [GeV]",
            logy=True,
            y_scale=1.5,
            n_ratio_panels=1,
            figsize=(9, 6),
        )
        plot_fr.add(tagger1, reference=True)
        plot_fr.add(tagger2)

        plot_fr.draw()

        plotname = "test_aux_fake_rate.png"
        plot_fr.savefig(f"{self.actual_plots_dir}/{plotname}")
