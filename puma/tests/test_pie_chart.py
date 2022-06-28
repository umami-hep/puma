#!/usr/bin/env python

"""
Unit test script for the functions in histogram.py
"""

import os
import tempfile
import unittest

import numpy as np
from matplotlib.testing.compare import compare_images

from puma import Histogram, HistogramPlot, PiePlot
from puma.utils import logger, set_log_level

set_log_level(logger, "DEBUG")


class histogram_plot_TestCase(unittest.TestCase):
    """Test class for puma.histogram_plot"""

    def setUp(self):
        np.random.seed(42)
        n_random = 10_000
        self.discrete_vals = [0, 4, 5, 15]
        self.pie = Histogram(
            np.random.choice(self.discrete_vals, size=n_random), label=f"N={n_random:_}"
        )

        # Set up directories for comparison plots
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.actual_plots_dir = f"{self.tmp_dir.name}/"
        self.expected_plots_dir = os.path.join(
            os.path.dirname(__file__), "expected_plots"
        )

    def test_invalid_bins_type(self):
        """check if ValueError is raised when using invalid type in `bins` argument"""
        hist_plot = HistogramPlot(bins=1.1, plot_pie=True)
        hist_plot.add(self.pie, reference=True)
        with self.assertRaises(ValueError):
            hist_plot.plot_pie_chart()

    def test_invalid_binwidth(self):
        """check if ValueError is raised when using a bin width larger than 1"""
        hist_plot = HistogramPlot(
            bins=5, bins_range=(0, 16), discrete_vals=self.discrete_vals, plot_pie=True
        )
        hist_plot.add(self.pie)
        with self.assertRaises(ValueError):
            hist_plot.plot_pie_chart()

    def test_no_discrete_vals(self):
        """check if ValueError is raised when no discrete values are provided"""
        hist_plot = HistogramPlot(
            bins=5, bins_range=(0, 16), plot_pie=True, vertical_split=True
        )
        hist_plot.add(self.pie)
        with self.assertRaises(ValueError):
            hist_plot.plot_pie_chart()

    def test_no_vertical_split(self):
        hist_plot = HistogramPlot(
            bins=5, bins_range=(0, 16), discrete_vals=[0, 4, 5, 15], plot_pie=True
        )
        hist_plot.add(self.pie)
        with self.assertRaises(ValueError):
            hist_plot.plot_pie_chart()

    def test_plot_pie_chart(self):
        """check if plot chart is plotted correctly"""
        hist_plot = HistogramPlot(
            bins=16,
            bins_range=(0, 16),
            discrete_vals=[0, 4, 5, 15],
            plot_pie=True,
            pie_labels=["light-flavour jets", "c-jets", "b-jets", "tau-jets"],
            vertical_split=True,
            title="PartonTruthLabelID",
        )
        hist_plot.add(self.pie)
        hist_plot.draw()
        plotname = "test_pie_chart.png"
        hist_plot.savefig(f"{self.actual_plots_dir}/{plotname}")
        # Uncomment line below to update expected image
        # hist_plot.savefig(f"{self.expected_plots_dir}/{plotname}")

        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{plotname}",
                f"{self.expected_plots_dir}/{plotname}",
                tol=1,
            )
        )

    def test_plot_pie_chart_default_style(self):
        """check if pie chart is plotted correctly (using default style)"""
        pie_plot = PiePlot(
            fracs=[20, 40, 30, 10],
            labels=["light-flavour jets", "c-jets", "b-jets", "tau-jets"],
        )
        plotname = "test_pie_chart_default_style.png"
        pie_plot.savefig(f"{self.actual_plots_dir}/{plotname}")
        # Uncomment line below to update expected image
        # pie_plot.savefig(f"{self.expected_plots_dir}/{plotname}")

        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{plotname}",
                f"{self.expected_plots_dir}/{plotname}",
                tol=1,
            )
        )
