"""Unit test script for the functions in var_vs_var.py."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
from matplotlib.testing.compare import compare_images

# from matplotlib.testing.compare import compare_images
from puma import VarVsVtx, VarVsVtxPlot
from puma.utils.logging import logger, set_log_level

set_log_level(logger, "DEBUG")


class VarVsVtxTestCase(unittest.TestCase):
    """Test class for the puma.var_vs_vtx functions."""

    def setUp(self):
        np.random.seed(42)

        self.x_var = np.linspace(100, 250, 20)
        self.n_true = np.random.randint(1, 10, size=20)
        self.n_reco = np.random.randint(1, 10, size=20)
        self.n_match = np.random.randint(0, np.minimum(self.n_true, self.n_reco))

    def test_var_vs_vtx_init_wrong_true_shape(self):
        """Test var_vs_vtx init."""
        with self.assertRaises(ValueError):
            VarVsVtx(
                x_var=np.ones(4),
                n_match=np.ones(4),
                n_true=np.ones(5),
                n_reco=np.ones(4),
            )

    def test_var_vs_vtx_init_wrong_reco_shape(self):
        """Test var_vs_vtx init."""
        with self.assertRaises(ValueError):
            VarVsVtx(
                x_var=np.ones(4),
                n_match=np.ones(4),
                n_true=np.ones(4),
                n_reco=np.ones(5),
            )

    def test_var_vs_vtx_init_wrong_match_shape(self):
        """Test var_vs_vtx init."""
        with self.assertRaises(ValueError):
            VarVsVtx(
                x_var=np.ones(4),
                n_match=np.ones(5),
                n_true=np.ones(4),
                n_reco=np.ones(4),
            )

    def test_var_vs_vtx_init(self):
        """Test var_vs_vtx init."""
        VarVsVtx(
            x_var=np.ones(6),
            n_match=np.ones(6),
            n_true=np.ones(6),
            n_reco=np.ones(6),
            bins=10,
            key="test",
        )

    def test_var_vs_vtx_set_bin_edges_list(self):
        """Test var_vs_vtx set_bin_edges."""
        var_plot = VarVsVtx(
            x_var=[0, 1, 2],
            n_match=[3, 4, 5],
            n_true=[3, 4, 5],
            n_reco=[3, 4, 5],
            bins=[0, 1, 2],
        )
        np.testing.assert_array_almost_equal(var_plot.bin_edges, [0, 1, 2])

    def test_var_vs_vtx_set_bin_edges(self):
        """Test var_vs_vtx set_bin_edges."""
        var_plot = VarVsVtx(
            x_var=[0, 1, 2],
            n_match=[3, 4, 5],
            n_true=[3, 4, 5],
            n_reco=[3, 4, 5],
            bins=2,
        )
        np.testing.assert_array_almost_equal(var_plot.bin_edges, [0, 1, 2], decimal=4)

    def test_var_vs_vtx_get_perf_ratio_zero(self):
        """Test var_vs_vtx get_performance_ratio with zero efficiency case."""
        var_plot = VarVsVtx(
            x_var=[0, 1, 2],
            n_match=[0, 0, 0],
            n_true=[3, 4, 5],
            n_reco=[3, 4, 5],
            bins=2,
        )
        pm, pm_error = var_plot.get_performance_ratio(var_plot.n_match, var_plot.n_true)
        self.assertEqual(pm, 0)
        self.assertEqual(pm_error, 0)

    def test_var_vs_vtx_get_perf_ratio_infinity(self):
        """Test var_vs_vtx get_performance_ratio with undefined efficiency case."""
        var_plot = VarVsVtx(
            x_var=[0, 1, 2],
            n_match=[0, 0, 0],
            n_true=[0, 0, 0],
            n_reco=[0, 0, 0],
            bins=2,
        )
        pm, pm_error = var_plot.get_performance_ratio(var_plot.n_match, var_plot.n_true)
        np.testing.assert_equal(pm, np.nan)
        np.testing.assert_equal(pm_error, np.nan)

    def test_var_vs_vtx_eq_different_classes(self):
        """Test var_vs_vtx eq."""
        var_plot = VarVsVtx(
            x_var=[0, 1, 2],
            n_match=[3, 4, 5],
            n_true=[3, 4, 5],
            n_reco=[3, 4, 5],
            bins=2,
        )
        self.assertNotEqual(var_plot, np.ones(6))

    def test_var_vs_vtx_get(self):
        var_plot = VarVsVtx(
            x_var=self.x_var,
            n_match=self.n_match,
            n_true=self.n_true,
            n_reco=self.n_reco,
            bins=1,
        )
        mode_options = ["efficiency", "purity", "fakes"]
        for mode in mode_options:
            var_plot.get(mode)


class VarVsVtxPlotTestCase(unittest.TestCase):
    """Test class for the puma.var_vs_vtx_plot."""

    def setUp(self):
        # Set up temp directory for comparison plots
        self.tmp_dir = tempfile.TemporaryDirectory()  # pylint:disable=R1732
        self.actual_plots_dir = f"{self.tmp_dir.name}/"
        self.expected_plots_dir = os.path.join(os.path.dirname(__file__), "expected_plots")
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

    def test_var_vs_vtx_plot_mode_wrong_option(self):
        with self.assertRaises(ValueError):
            VarVsVtxPlot(
                mode="test",
                ylabel="Vtx purity",
                xlabel=r"$p_{T}$ [GeV]",
                logy=True,
                atlas_second_tag="test",
                y_scale=1.5,
                n_ratio_panels=1,
                figsize=(9, 6),
            )

    def test_var_vs_vtx_plot_mode_efficiency(self):
        """Test vtx output plot - efficiency."""
        # define the curves
        tagger1 = VarVsVtx(
            x_var=self.x_var,
            n_true=self.n_true,
            n_reco=self.n_reco_1,
            n_match=self.n_match_1,
            bins=self.bins,
            label="tagger 1",
        )
        tagger2 = VarVsVtx(
            x_var=self.x_var,
            n_true=self.n_true,
            n_reco=self.n_reco_2,
            n_match=self.n_match_2,
            bins=self.bins,
            label="tagger 2",
        )

        plot_eff = VarVsVtxPlot(
            mode="efficiency",
            ylabel="Vtx efficiency",
            xlabel=r"$p_{T}$ [GeV]",
            logy=True,
            y_scale=1.5,
            n_ratio_panels=1,
            figsize=(9, 6),
        )
        plot_eff.add(tagger1, reference=True)
        plot_eff.add(tagger2)

        plot_eff.draw()

        plotname = "test_vtx_efficiency.png"
        plot_eff.savefig(f"{self.actual_plots_dir}/{plotname}")
        # Uncomment line below to update expected image
        # plot_eff.savefig(f"{self.expected_plots_dir}/{plotname}")

        # Investigate small shifts in labels/scale causing this test to fail
        self.assertEqual(
            None,
            compare_images(
                f"{self.actual_plots_dir}/{plotname}",
                f"{self.expected_plots_dir}/{plotname}",
                tol=5,
            ),
        )

    def test_var_vs_vtx_plot_mode_purity(self):
        """Test vtx output plot - purity."""
        # define the curves
        tagger1 = VarVsVtx(
            x_var=self.x_var,
            n_true=self.n_true,
            n_reco=self.n_reco_1,
            n_match=self.n_match_1,
            bins=self.bins,
            label="tagger 1",
        )
        tagger2 = VarVsVtx(
            x_var=self.x_var,
            n_true=self.n_true,
            n_reco=self.n_reco_2,
            n_match=self.n_match_2,
            bins=self.bins,
            label="tagger 2",
        )

        plot_pur = VarVsVtxPlot(
            mode="purity",
            ylabel="Vtx purity",
            xlabel=r"$p_{T}$ [GeV]",
            logy=True,
            y_scale=1.5,
            n_ratio_panels=1,
            figsize=(9, 6),
        )
        plot_pur.add(tagger1, reference=True)
        plot_pur.add(tagger2)

        plot_pur.draw()

        plotname = "test_vtx_purity.png"
        plot_pur.savefig(f"{self.actual_plots_dir}/{plotname}")
        # Uncomment line below to update expected image
        # plot_pur.savefig(f"{self.expected_plots_dir}/{plotname}")

        # Investigate small shifts in labels/scale causing this test to fail
        self.assertEqual(
            None,
            compare_images(
                f"{self.actual_plots_dir}/{plotname}",
                f"{self.expected_plots_dir}/{plotname}",
                tol=5,
            ),
        )

    def test_var_vs_vtx_plot_mode_fake_rate(self):
        """Test aux output plot - fake_rate."""
        # define the curves
        tagger1 = VarVsVtx(
            x_var=self.x_var,
            n_true=self.n_true,
            n_reco=self.n_reco_1,
            n_match=self.n_match_1,
            bins=self.bins,
            label="tagger 1",
        )
        tagger2 = VarVsVtx(
            x_var=self.x_var,
            n_true=self.n_true,
            n_reco=self.n_reco_2,
            n_match=self.n_match_2,
            bins=self.bins,
            label="tagger 2",
        )

        plot_fakes = VarVsVtxPlot(
            mode="fakes",
            ylabel="Vtx fake rate",
            xlabel=r"$p_{T}$ [GeV]",
            logy=True,
            y_scale=1.5,
            n_ratio_panels=1,
            figsize=(9, 6),
        )
        plot_fakes.add(tagger1, reference=True)
        plot_fakes.add(tagger2)

        plot_fakes.draw()

        plotname = "test_vtx_fake_rate.png"
        plot_fakes.savefig(f"{self.actual_plots_dir}/{plotname}")
        # Uncomment line below to update expected image
        # plot_fakes.savefig(f"{self.expected_plots_dir}/{plotname}")

        # Investigate small shifts in labels/scale causing this test to fail
        self.assertEqual(
            None,
            compare_images(
                f"{self.actual_plots_dir}/{plotname}",
                f"{self.expected_plots_dir}/{plotname}",
                tol=5,
            ),
        )
